"""
AI Code Service - 高性能 Apple Silicon MLX 模型推理引擎
集成 Metal 统一内存优化、采样器调优、LRU Token 计数缓存与实时 TPS/显存监控
"""

import functools
import threading
import time
from typing import Generator, Optional, List, Union, Dict, Any
from .base import BaseModelEngine


class MLXModelEngine(BaseModelEngine):
    """基于 Apple MLX 硬件加速的高性能 LLM 推理引擎"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        engine_type: str = "auto",
        metal_cache_limit_mb: int = 2048,
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
            if mx.metal.is_available():
                limit_bytes = self.metal_cache_limit_mb * 1024 * 1024
                mx.metal.set_cache_limit(limit_bytes)
        except Exception:
            pass

    def load_model(self):
        """延迟加载模型与 Tokenizer"""
        if self._loaded:
            return

        with self.lock:
            if self._loaded:
                return

            print(f"[MLXEngine] 正在从 {self.model_path} 加载模型 {self.model_name}...")

            # 尝试优先使用 mlx_lm
            if self.engine_type in ("auto", "mlx_lm"):
                try:
                    import mlx_lm
                    self.model, self.tokenizer = mlx_lm.load(self.model_path)
                    self.generate_fn = mlx_lm.generate
                    self.stream_generate_fn = mlx_lm.stream_generate
                    self._loaded = True
                    print(f"[MLXEngine] 成功通过 mlx_lm 加载模型: {self.model_name}")
                    return
                except Exception as e:
                    print(f"[MLXEngine] mlx_lm 加载未成功: {e}，尝试使用 mlx_vlm...")

            # 尝试使用 mlx_vlm
            if self.engine_type in ("auto", "mlx_vlm"):
                try:
                    import mlx_vlm
                    self.model, self.processor = mlx_vlm.load(self.model_path)
                    self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    self.generate_fn = mlx_vlm.generate
                    self.stream_generate_fn = getattr(mlx_vlm, "stream_generate", None)
                    self._loaded = True
                    print(f"[MLXEngine] 成功通过 mlx_vlm 加载模型: {self.model_name}")
                    return
                except Exception as e:
                    print(f"[MLXEngine] mlx_vlm 加载未成功: {e}")

            raise RuntimeError(
                f"无法加载模型 '{self.model_name}' (路径: {self.model_path})，请先运行 download.py 下载模型权重或开启 mock 模式。"
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
        """构建优化采样参数（支持 top_k, min_p, repetition_penalty 等）"""
        gen_kwargs = {
            "temp": temperature,
            "top_p": top_p,
        }
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            gen_kwargs["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        if "repetition_context_size" in kwargs and kwargs["repetition_context_size"] is not None:
            gen_kwargs["repetition_context_size"] = kwargs["repetition_context_size"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            gen_kwargs["seed"] = kwargs["seed"]
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
                result = self.generate_fn(self.model, self.processor, prompt, **gen_kwargs)
            else:
                result = self.generate_fn(self.model, self.tokenizer, prompt, **gen_kwargs)

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
            if mx.metal.is_available():
                mx.metal.clear_cache()
        except Exception:
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取 Apple Metal 显存状态"""
        try:
            import mlx.core as mx
            if mx.metal.is_available():
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
