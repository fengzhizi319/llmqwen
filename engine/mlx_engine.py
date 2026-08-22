"""
AI Code Service - 高性能 Apple Silicon MLX 模型推理引擎
集成 Metal 统一内存优化、自动本地缓存解析、采样器调优与实时 TPS/显存监控
"""

import functools
import glob
import os
import threading
import time
from typing import Generator, Optional, List, Union, Dict, Any
from .base import BaseModelEngine


def resolve_local_model_path(model_path: str) -> str:
    """
    智能解析模型本地路径：
    1. 若是有效绝对/相对路径或以 ~ 开头，展开后存在则直接返回
    2. 若是仓库名 (如 Qwen/Qwen3.8-27B 或 lmstudio-community/Qwen3.8-27B-MLX-8bit)，
       优先搜索 ModelScope 缓存目录 (~/.cache/modelscope/models/.../snapshots/*)
    3. 搜索 HuggingFace 缓存目录 (~/.cache/huggingface/hub/models--.../snapshots/*)
    4. 搜索 LM Studio 缓存目录 (~/.cache/lm-studio/models/...)
    5. 若未在本地缓存中找到，则返回原路径供下游尝试在线加载
    """
    expanded = os.path.expanduser(model_path)
    if os.path.exists(expanded):
        return expanded

    repo_id_normalized = model_path.replace("/", "--")
    search_patterns = [
        f"~/.cache/modelscope/models/{repo_id_normalized}/snapshots/*",
        f"~/.cache/modelscope/models/{repo_id_normalized}",
        f"~/.cache/modelscope/hub/{model_path}",
        f"~/.cache/huggingface/hub/models--{repo_id_normalized}/snapshots/*",
        f"~/.cache/lm-studio/models/*/{model_path}*",
    ]
    for pattern in search_patterns:
        matches = glob.glob(os.path.expanduser(pattern))
        for match in sorted(matches, reverse=True):
            if os.path.isdir(match):
                # 检查该目录是否包含模型配置文件或权重
                if any(os.path.exists(os.path.join(match, f)) for f in ("config.json", "params.json", "configuration.json")):
                    return match

    return expanded


class MLXModelEngine(BaseModelEngine):
    """基于 Apple MLX 硬件加速的高性能 LLM 推理引擎"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        engine_type: str = "auto",
        metal_cache_limit_mb: int = 4096,
        clear_cache_after_generation: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.engine_type = engine_type
        self.metal_cache_limit_mb = metal_cache_limit_mb
        self.clear_cache_after_generation = clear_cache_after_generation

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.generate_fn = None
        self.stream_generate_fn = None
        self.lock = threading.Lock()
        self._loaded = False
        self.resolved_path = None

        # 性能统计指标
        self._total_requests = 0
        self._total_prompt_tokens = 0
        self._total_generation_tokens = 0
        self._last_prompt_tps = 0.0
        self._last_generation_tps = 0.0
        self._generation_times: List[float] = []

        # 配置 MLX Metal 内存参数
        self._init_metal_runtime()

    def _init_metal_runtime(self):
        """初始化 MLX Metal 运行时显存与缓存限制"""
        try:
            import mlx.core as mx
            limit_bytes = self.metal_cache_limit_mb * 1024 * 1024
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(limit_bytes)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
                mx.metal.set_cache_limit(limit_bytes)
        except Exception:
            pass

    def load_model(self):
        """延迟加载模型与 Tokenizer（优先从本地 ModelScope/HuggingFace 缓存加载）"""
        if self._loaded:
            return

        with self.lock:
            if self._loaded:
                return

            self.resolved_path = resolve_local_model_path(self.model_path)
            print(f"[MLXEngine] 正在从本地路径 '{self.resolved_path}' 加载模型 '{self.model_name}'...")

            # 尝试优先使用 mlx_lm
            if self.engine_type in ("auto", "mlx_lm"):
                try:
                    import mlx_lm
                    self.model, self.tokenizer = mlx_lm.load(self.resolved_path)
                    self.generate_fn = mlx_lm.generate
                    self.stream_generate_fn = mlx_lm.stream_generate
                    self._loaded = True
                    print(f"[MLXEngine] 成功通过 mlx_lm 从本地加载模型: {self.model_name}")
                    return
                except Exception as e:
                    print(f"[MLXEngine] mlx_lm 加载未成功: {e}，尝试使用 mlx_vlm...")

            # 尝试使用 mlx_vlm
            if self.engine_type in ("auto", "mlx_vlm"):
                try:
                    import mlx_vlm
                    self.model, self.processor = mlx_vlm.load(self.resolved_path)
                    self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    self.generate_fn = mlx_vlm.generate
                    self.stream_generate_fn = getattr(mlx_vlm, "stream_generate", None)
                    self._loaded = True
                    print(f"[MLXEngine] 成功通过 mlx_vlm 从本地加载模型: {self.model_name}")
                    return
                except Exception as e:
                    print(f"[MLXEngine] mlx_vlm 加载未成功: {e}")

            raise RuntimeError(
                f"无法加载模型 '{self.model_name}' (解析路径: {self.resolved_path})，请先运行 download.py 下载模型权重或开启 mock 模式。"
            )

    @functools.lru_cache(maxsize=4096)
    def _cached_count_tokens(self, text: str) -> int:
        """带 LRU 缓存的高速 Token 计数"""
        if not text:
            return 0
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_tokens(self, text: str) -> int:
        return self._cached_count_tokens(text)

    def _build_sampler_kwargs(self, temperature: float, top_p: float, **kwargs) -> Dict[str, Any]:
        """构建优化采样参数（完美适配 mlx_lm 的 make_sampler 与 mlx_vlm）"""
        gen_kwargs: Dict[str, Any] = {}

        # mlx_lm (基于 make_sampler 与 make_logits_processors)
        if self.processor is None:
            try:
                from mlx_lm.sample_utils import make_sampler, make_logits_processors
                temp_val = max(0.0, float(temperature))
                top_p_val = max(0.0, min(1.0, float(top_p))) if top_p is not None else 0.0
                top_k_val = int(kwargs["top_k"]) if kwargs.get("top_k") else 0
                min_p_val = float(kwargs["min_p"]) if kwargs.get("min_p") else 0.0

                gen_kwargs["sampler"] = make_sampler(
                    temp=temp_val,
                    top_p=top_p_val,
                    top_k=top_k_val,
                    min_p=min_p_val,
                )

                if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
                    rep_ctx = int(kwargs.get("repetition_context_size", 20))
                    gen_kwargs["logits_processors"] = make_logits_processors(
                        repetition_penalty=float(kwargs["repetition_penalty"]),
                        repetition_context_size=rep_ctx,
                    )
            except Exception:
                gen_kwargs["temp"] = temperature
                gen_kwargs["top_p"] = top_p
        else:
            # mlx_vlm
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
                gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]

        return gen_kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        self.load_model()
        gen_kwargs = self._build_sampler_kwargs(temperature, top_p, **kwargs)
        gen_kwargs["max_tokens"] = max_tokens

        start_t = time.time()
        with self.lock:
            if self.processor:
                raw_res = self.generate_fn(self.model, self.processor, prompt, **gen_kwargs)
            else:
                raw_res = self.generate_fn(self.model, self.tokenizer, prompt, **gen_kwargs)

            # 确保提取纯文本字符串（兼容 mlx_lm 与 mlx_vlm 返回的 GenerationResult 对象）
            if hasattr(raw_res, "text"):
                result = raw_res.text
            elif isinstance(raw_res, str):
                result = raw_res
            else:
                result = str(raw_res)

            # Stop 字符截断
            if stop:
                stop_list = [stop] if isinstance(stop, str) else stop
                for s in stop_list:
                    if s in result:
                        result = result.split(s)[0]

            duration = time.time() - start_t
            p_tokens = self.count_tokens(prompt)
            c_tokens = self.count_tokens(result)

            self._total_requests += 1
            self._total_prompt_tokens += p_tokens
            self._total_generation_tokens += c_tokens
            if duration > 0 and c_tokens > 0:
                self._last_generation_tps = round(c_tokens / duration, 2)
                self._generation_times.append(duration)

            if self.clear_cache_after_generation:
                self._clear_metal_cache()

            return result

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        self.load_model()
        gen_kwargs = self._build_sampler_kwargs(temperature, top_p, **kwargs)
        gen_kwargs["max_tokens"] = max_tokens

        if not self.stream_generate_fn:
            full_res = self.generate(prompt, max_tokens, temperature, top_p, stop, **kwargs)
            for char in full_res:
                yield char
            return

        with self.lock:
            arg_target = self.processor if self.processor else self.tokenizer
            stop_list = [stop] if isinstance(stop, str) else (stop or [])
            accumulated = ""
            stopped = False

            for response in self.stream_generate_fn(self.model, arg_target, prompt, **gen_kwargs):
                if stopped:
                    break

                text_chunk = response.text if hasattr(response, "text") else str(response)
                
                # 记录 MLX 引擎内部报告的 TPS 指标
                if hasattr(response, "prompt_tps") and response.prompt_tps:
                    self._last_prompt_tps = round(float(response.prompt_tps), 2)
                if hasattr(response, "generation_tps") and response.generation_tps:
                    self._last_generation_tps = round(float(response.generation_tps), 2)

                accumulated += text_chunk

                # 检查 stop 截断
                for s in stop_list:
                    if s in accumulated:
                        stopped = True
                        cutoff_index = accumulated.find(s)
                        # 计算当前 chunk 需要保留的部分
                        text_chunk = text_chunk[:len(text_chunk) - (len(accumulated) - cutoff_index)]
                        break

                if text_chunk:
                    yield text_chunk

            self._total_requests += 1
            if self.clear_cache_after_generation:
                self._clear_metal_cache()

    def _clear_metal_cache(self):
        """主动清理 Metal 显存缓存"""
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取 Apple Metal 显存状态"""
        try:
            import mlx.core as mx
            if hasattr(mx, "metal") and mx.metal.is_available():
                active_mb = round(mx.metal.get_active_memory() / (1024 * 1024), 2)
                cache_mb = round(mx.metal.get_cache_memory() / (1024 * 1024), 2)
                peak_mb = round(mx.metal.get_peak_memory() / (1024 * 1024), 2)
                return {
                    "active_memory_mb": active_mb,
                    "cache_memory_mb": cache_mb,
                    "peak_memory_mb": peak_mb,
                }
        except Exception:
            pass
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """返回引擎性能统计指标"""
        stats = {
            "model_name": self.model_name,
            "engine_type": self.engine_type,
            "loaded": self._loaded,
            "resolved_path": self.resolved_path,
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_generation_tokens": self._total_generation_tokens,
            "last_prompt_tps": self._last_prompt_tps,
            "last_generation_tps": self._last_generation_tps,
        }
        stats.update(self.get_memory_stats())
        return stats

    def health_check(self) -> bool:
        return self._loaded
