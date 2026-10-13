"""
AI Code Service - Qwen4Exp (Flash-Next) 高性能推理引擎

核心能力:
  - Qwen4Exp 架构支持: 混合线性/全注意力层 (linear_attention + full_attention)、
    PLE 渐进层嵌入、HyperConnection 超连接、SwitchGLU MoE 专家混合
  - oQ4e 混合精度量化: 3-bit/4-bit/8-bit 自适应量化，大幅压缩模型体积
  - 内置 MTP 推测解码: 模型自带 MoE MTP 层，本引擎执行位置对齐验证与完整状态重放
  - 自定义模型代码注册: 通过 omlx_support 补丁将 qwen4_exp 模型类型注入 mlx_lm
  - Thinking 模式处理: 自动识别并可选剥离 <think>...</think> 思考链输出
  - 128K 超长上下文: 支持 131,072 tokens 上下文窗口

架构层级:
  _register_custom_model_code()  → omlx_support 自定义模型代码注册 (全局单例)
  Qwen4ExpEngine                → 推理引擎核心 (继承 BaseModelEngine)
    ├── load_model()             → 注册自定义代码 → mlx_lm.load 加载模型
    ├── generate()               → 非流式生成 (支持 thinking 模式控制)
    ├── stream_generate()        → 流式生成 (mlx_lm 原生 stream_generate)
    └── get_stats()              → 性能指标采集
"""

import functools
import glob
import os
import sys
import threading
import time
from typing import Generator, Optional, List, Union, Dict, Any

from .base import BaseModelEngine
from .prefix_cache import PrefixSnapshotCache


# ── 全局自定义模型代码注册标志 (单例模式，仅注册一次) ──
_custom_code_registered = False
_custom_code_lock = threading.Lock()


def _register_custom_model_code(omlx_support_path: str) -> None:
    """注册 Qwen4Exp 自定义模型代码到 mlx_lm 模型注册表

    Qwen4Exp (qwen4_exp) 是 mlx_lm 尚未内置的新架构类型，
    需要通过 omlx_support/qwen4_exp.py 将模型定义注入到 mlx_lm.models 命名空间。

    注册流程 (参考 omlx_support/sitecustomize.py):
      1. 将 omlx_support 目录插入 mlx_lm.models.__path__ 头部
      2. 使 mlx_lm 的模型加载器能发现 qwen4_exp.Model / ModelArgs

    注意: 此操作为全局单例，多次调用仅首次生效。
    缓存集成 (qwen4_cache_integration) 和 MTP 集成 (qwen4_mtp_integration)
    依赖 omlx 运行时，本引擎不依赖这些组件——
    mlx_lm 原生 KV Cache 机制已能满足基本推理需求。

    Args:
        omlx_support_path: omlx_support 目录的绝对路径
    """
    global _custom_code_registered
    if _custom_code_registered:
        return

    with _custom_code_lock:
        if _custom_code_registered:
            return

        # 注入 omlx.patches.mlx_lm_mtp 兼容垫片，激活 Qwen4Exp 原生 MTP (Multi-Token Prediction) 模块
        import types
        if "omlx" not in sys.modules:
            m_omlx = types.ModuleType("omlx")
            m_patches = types.ModuleType("omlx.patches")
            m_mtp = types.ModuleType("omlx.patches.mlx_lm_mtp")
            m_mtp.is_mtp_active = lambda: True
            m_omlx.patches = m_patches
            m_patches.mlx_lm_mtp = m_mtp
            sys.modules["omlx"] = m_omlx
            sys.modules["omlx.patches"] = m_patches
            sys.modules["omlx.patches.mlx_lm_mtp"] = m_mtp

        # 将 omlx_support 目录添加到 Python 路径 (供 import 发现)
        if omlx_support_path not in sys.path:
            sys.path.insert(0, omlx_support_path)

        # 注入到 mlx_lm.models 命名空间，使 qwen4_exp 模型类型可被发现
        import mlx_lm.models
        if omlx_support_path not in mlx_lm.models.__path__:
            mlx_lm.models.__path__.insert(0, omlx_support_path)

        _custom_code_registered = True
        print(f"[Qwen4ExpEngine] 自定义模型代码与 MTP 支持已注册: {omlx_support_path}")


def resolve_local_model_path(model_path: str) -> str:
    """
    智能解析模型本地路径：
    1. 若是有效绝对/相对路径或以 ~ 开头，展开后存在则直接返回
    2. 若是仓库名，优先搜索 ModelScope 缓存目录
    3. 搜索 HuggingFace / LM Studio 缓存目录
    4. 若未在本地缓存中找到，则返回原路径供下游尝试在线加载
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
        f"~/.cache/lm-studio/models/{model_path}",
        f"~/.cache/lm-studio/models/*/{model_path}*",
        f"~/.cache/lm-studio/models/*/{model_path.split('/')[-1]}*",
        f"~/.lmstudio/models/{model_path}",
        f"~/.lmstudio/models/*/{model_path}*",
        f"~/.lmstudio/models/*/{model_path.split('/')[-1]}*",
    ]
    for pattern in search_patterns:
        matches = glob.glob(os.path.expanduser(pattern))
        for match in sorted(matches, reverse=True):
            if os.path.isdir(match):
                if any(os.path.exists(os.path.join(match, f)) for f in ("config.json", "params.json", "configuration.json")):
                    return match

    return expanded


class Qwen4ExpEngine(BaseModelEngine):
    """基于 Qwen4Exp 架构的高性能 LLM 推理引擎

    专为 Qwen3.8-Flash-Next-oQ4e-MTP-128k 模型设计，
    该模型采用全新的 Qwen4Exp 架构 (混合线性/全注意力 + PLE + HyperConnection + MoE)，
    需要通过 omlx_support 自定义代码注册才能被 mlx_lm 正确加载。

    与 MLXModelEngine 的关键差异:
      - 架构类型: qwen4_exp (非 qwen3_5)，需要自定义模型代码注入
      - MTP 策略: 模型内置 MoE MTP 层，本引擎进行贪婪验证，无需外部 MTP 头
      - Thinking 模式: 模型默认输出 <think>...</think> 思考链，支持可选剥离
      - 内存占用: ~93GB 峰值内存 (18 分片 + MTP 专家权重)

    线程安全: 通过 self.lock 保证模型加载/卸载/生成的串行化。
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        metal_cache_limit_mb: int = 8192,       # Qwen4Exp 需要更大 Metal 缓存 (默认 8GB)
        clear_cache_after_generation: bool = False,
        kv_bits: Optional[int] = 8,             # KV Cache 量化位数
        kv_group_size: int = 64,                # KV 量化分组大小
        prefill_step_size: int = 2048,          # 分块预填充步长
        enable_prompt_cache: bool = True,       # 是否启用 prompt KV cache 复用
        enable_mtp: bool = True,                # 🚀 是否启用 MTP (Multi-Token Prediction) 多 Token 投机推测
        mtp_depth: int = 3,                     # MTP 草稿深度 (1~3)，收益取决于接受率
        code_agent_mode: bool = True,           # 🚀 Code Agent 专项加速模式 (低温确定性、前缀持久化、工具截断)
        strip_thinking: bool = False,           # 是否剥离 <think>...</think> 思考链输出
        enable_thinking: bool = True,           # 是否启用 thinking 模式 (传入 chat template)
        prompt_cache_max_entries: int = 4,
        prompt_cache_max_tokens: int = 8192,
    ):
        # ── 配置参数 ──
        self.model_name = model_name
        self.model_path = model_path
        self.metal_cache_limit_mb = metal_cache_limit_mb
        self.clear_cache_after_generation = clear_cache_after_generation
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.prefill_step_size = prefill_step_size
        self.enable_prompt_cache = enable_prompt_cache
        self.enable_mtp = enable_mtp
        self.mtp_depth = mtp_depth
        self.code_agent_mode = code_agent_mode
        self.strip_thinking = strip_thinking
        self.enable_thinking = enable_thinking

        # ── 模型与 Tokenizer (延迟加载) ──
        self.model = None               # MLX 模型实例 (qwen4_exp.Model)
        self.tokenizer = None           # 分词器
        self.generate_fn = None         # 非流式生成函数引用
        self.stream_generate_fn = None  # 流式生成函数引用
        self.lock = threading.Lock()    # 全局串行锁 (加载/卸载/生成)
        self._loaded = False            # 模型是否已加载
        self.has_mtp = False            # 是否具有并已加载 MTP 辅助模块
        self.resolved_path = None       # 解析后的本地模型绝对路径
        self.omlx_support_path = None   # omlx_support 自定义代码路径

        # ── Code Agent 会话级持久化 Prompt 前缀 KV Cache ──
        self._prefix_cache = PrefixSnapshotCache(
            prompt_cache_max_entries, prompt_cache_max_tokens
        )
        self._last_reused_tokens = 0
        self._last_prefill_tokens = 0
        self._last_prompt_token_count = 0
        self._last_prefill_seconds = 0.0
        self._mtp_drafted_tokens = 0
        self._mtp_accepted_tokens = 0

        # ── 性能统计指标 ──
        self._total_requests = 0
        self._total_prompt_tokens = 0
        self._total_generation_tokens = 0
        self._last_prompt_tps = 0.0
        self._last_generation_tps = 0.0
        self._generation_times: List[float] = []

        # 配置 MLX Metal 运行时显存限制
        self._init_metal_runtime()

    def _init_metal_runtime(self):
        """初始化 MLX Metal 运行时显存与缓存限制

        Qwen4Exp 模型峰值内存约 93GB，需要较大的 Metal 缓存池。
        默认 8192 MB (8 GB)，可根据设备内存调整。
        """
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
        """延迟加载模型与 Tokenizer (双重检查锁保证线程安全)

        加载流程:
          1. 解析本地模型路径 (支持 ModelScope/HuggingFace/LM Studio 缓存)
          2. 定位并注册 omlx_support 自定义模型代码 (qwen4_exp 架构) + MTP 激活补丁
          3. 通过 mlx_lm.load 加载模型权重 (包含全部 62 个 MTP 专家与投机权重) 与 Tokenizer
          4. 检查并激活 MTP (Multi-Token Prediction) 投机推测模块
        """
        if self._loaded:
            return

        with self.lock:
            if self._loaded:
                return

            # 智能解析本地模型路径
            self.resolved_path = resolve_local_model_path(self.model_path)
            print(f"[Qwen4ExpEngine] 正在从本地路径 '{self.resolved_path}' 加载模型 '{self.model_name}'...")

            # ── Step 1: 定位 omlx_support 目录 ──
            omlx_candidate = os.path.join(self.resolved_path, "omlx_support")
            if os.path.isdir(omlx_candidate):
                self.omlx_support_path = omlx_candidate
            else:
                omlx_candidate = os.path.join(self.resolved_path, "omlx_support")
                if os.path.isdir(omlx_candidate):
                    self.omlx_support_path = omlx_candidate

            # ── Step 2: 注册自定义模型代码与 MTP 支持 ──
            if self.omlx_support_path and os.path.isdir(self.omlx_support_path):
                _register_custom_model_code(self.omlx_support_path)
                print(f"[Qwen4ExpEngine] 已定位 omlx_support: {self.omlx_support_path}")
            else:
                print(f"[Qwen4ExpEngine] ⚠️ 未找到 omlx_support 目录，模型加载可能失败")

            # ── Step 3: 通过 mlx_lm 加载模型 ──
            try:
                import mlx_lm
                self.model, self.tokenizer = mlx_lm.load(self.resolved_path)
                self.generate_fn = mlx_lm.generate
                self.stream_generate_fn = mlx_lm.stream_generate
                self.has_mtp = hasattr(self.model, "mtp") and self.model.mtp is not None
                self._loaded = True
                print(f"[Qwen4ExpEngine] ✅ 成功加载 Qwen4Exp 模型: {self.model_name}")
                print(f"[Qwen4ExpEngine] 模型类型: {type(self.model).__module__}.{type(self.model).__name__}")
                if self.has_mtp and self.enable_mtp:
                    print(f"[Qwen4ExpEngine] 🚀 MTP 多 Token 投机推测已就绪 (深度: {self.mtp_depth}，完整状态验证已启用)")
                else:
                    print(f"[Qwen4ExpEngine] ℹ️ 标准串行解码模式运行")
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 Qwen4Exp 模型 '{self.model_name}' (路径: {self.resolved_path}): {e}\n"
                    f"请确认 omlx_support 自定义模型代码可用，或先运行 download.py 下载模型权重。"
                )

    @functools.lru_cache(maxsize=4096)
    def _cached_count_tokens(self, text: str) -> int:
        """带 LRU 缓存的高速 Token 计数"""
        if not text:
            return 0
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            try:
                encoded = self.tokenizer.encode(text)
                count = len(encoded)
                if count > max(len(text), 10):
                    return max(1, len(text) // 4)
                return count
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_tokens(self, text: str) -> int:
        return self._cached_count_tokens(text)

    def _prepare_prefix(self, prompt, tokenizer):
        """Restore an exact prefill checkpoint, then process only the suffix.

        Must run under self.lock on the MLX generation stream. Snapshots are
        taken before decoding, including all four ArraysCache states and QSA
        index keys. No generated token or speculative state enters the cache.
        """
        import mlx.core as mx

        add_bos = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        tokens = tokenizer.encode(prompt, add_special_tokens=add_bos)
        if not tokens:
            raise ValueError("Prompt must contain at least one token")
        started = time.perf_counter()
        state, reused = self._prefix_cache.restore(tokens) if self.enable_prompt_cache else (None, 0)
        state = state if state is not None else self.model.make_cache()
        target = len(tokens) - 1
        # Retain the stable system/tool declaration boundary as well as the
        # most recent bounded prompt. This also helps branched conversations.
        boundaries = {target}
        cap = self._prefix_cache.max_tokens
        if self.enable_prompt_cache and 0 < cap < target:
            boundaries.add(cap)
        marker = "<|im_end|>"
        if self.enable_prompt_cache and prompt.startswith("<|im_start|>system\n") and marker in prompt:
            system_text = prompt[:prompt.index(marker) + len(marker)]
            system_tokens = tokenizer.encode(system_text, add_special_tokens=add_bos)
            if tokens[:len(system_tokens)] == system_tokens and len(system_tokens) <= min(cap, target):
                boundaries.add(len(system_tokens))
        processed = reused
        for boundary in sorted(boundaries):
            if boundary <= processed:
                continue
            while processed < boundary:
                end = min(boundary, processed + max(1, self.prefill_step_size))
                self.model(mx.array(tokens[processed:end], dtype=mx.uint32)[None], cache=state)
                mx.eval([layer.state for layer in state])
                processed = end
            if self.enable_prompt_cache:
                self._prefix_cache.store(tokens[:processed], state)
        self._last_prompt_token_count = len(tokens)
        self._last_reused_tokens = reused
        self._last_prefill_tokens = target - reused
        self._last_prefill_seconds = time.perf_counter() - started
        return state, tokens[processed:], tokens

    def _prepare_native_stream(self, prompt):
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        tokenizer.detokenizer.reset()
        with mx.stream(generation_stream):
            state, remaining, _ = self._prepare_prefix(prompt, tokenizer)
        return tokenizer, remaining, state

    def _build_sampler_kwargs(self, temperature: float, top_p: float, **kwargs) -> Dict[str, Any]:
        """构建采样器参数

        适配 mlx_lm 的 make_sampler + make_logits_processors API。

        注意: Qwen4Exp 架构使用自定义缓存 (QSAKVCache / PLE 嵌入缓存)，
        不支持标准 KV Cache 量化 (kv_bits/kv_group_size)。
        因此不注入 kv_bits/kv_group_size 参数，分块预填充由本引擎处理，
        避免 generate_step 尝试量化自定义缓存导致输出异常。
        """
        gen_kwargs: Dict[str, Any] = {}

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

            # 自定义缓存不注入 kv_bits/kv_group_size；分块预填充由 _prepare_prefix 处理。

        except Exception:
            gen_kwargs["temp"] = temperature
            gen_kwargs["top_p"] = top_p

        return gen_kwargs

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """剥离 <think>...</think> 思考链标签及其内容

        Qwen4Exp 模型默认启用 thinking 模式，输出格式为:
          <think>
          ...思考过程...
</think>

        最终回答内容

        当 strip_thinking=True 时，移去思考链部分，仅保留最终回答。
        """
        import re
        # 匹配完整的 <think>...</think> 块 (含换行)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """非流式生成: 一次性返回完整生成文本

        流程: 加载模型 → 构建采样参数 → 调用 mlx_lm.generate → 提取文本
              → thinking 标签处理 → stop 截断 → 更新统计
        """
        self.load_model()
        # Use the same accelerated path and stop detection as streaming agents.
        if self.code_agent_mode and self.stream_generate_fn is not None:
            return "".join(self.stream_generate(
                prompt, max_tokens, temperature, top_p, stop, **kwargs
            ))
        gen_kwargs = self._build_sampler_kwargs(temperature, top_p, **kwargs)
        gen_kwargs["max_tokens"] = max_tokens

        start_t = time.time()
        with self.lock:
            raw_res = self.generate_fn(self.model, self.tokenizer, prompt, **gen_kwargs)

            # 兼容 GenerationResult 对象与纯字符串返回
            if hasattr(raw_res, "text"):
                result = raw_res.text
            elif isinstance(raw_res, str):
                result = raw_res
            else:
                result = str(raw_res)

            # Thinking 模式: 可选剥离 <think>...</think> 思考链
            if self.strip_thinking:
                result = self._strip_thinking_tags(result)

            # Stop 字符串截断
            if stop:
                stop_list = [stop] if isinstance(stop, str) else stop
                for s in stop_list:
                    if s in result:
                        result = result.split(s)[0]

            # 更新性能统计指标
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

    def _stream_generate_mtp(
        self, prompt, max_tokens, temperature, stop_list, gen_kwargs,
    ) -> Generator[str, None, None]:
        """Greedy MTP with correctly aligned verification and full-state replay.

        Verification inputs are [current, draft_0, ...], so logits[j] predicts
        draft[j]. Rejected speculative states are discarded in their entirety;
        replay restores recurrent, PLE and QSA state without partial trimming.
        """
        import copy
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        with self.lock, mx.stream(generation_stream):
            started = time.perf_counter()
            main_cache, remaining, prompt_tokens = self._prepare_prefix(prompt, tokenizer)
            logits, hidden = self.model(
                mx.array(remaining, dtype=mx.uint32)[None],
                cache=main_cache, return_hidden=True,
            )
            current = mx.argmax(logits[:, -1, :], axis=-1)
            current_hidden = hidden[:, -1:, :]
            mx.eval(current, current_hidden)
            self._last_prompt_tps = round(
                (self._last_prefill_tokens + len(remaining)) / max(time.perf_counter() - started, 1e-9), 2
            )
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            accumulated = ""
            generated = 0
            stopped = False
            gen_started = time.perf_counter()

            def emit(token_id):
                nonlocal accumulated, generated, stopped
                if token_id in tokenizer.eos_token_ids:
                    stopped = True
                    return ""
                generated += 1
                detokenizer.add_token(token_id)
                text = detokenizer.last_segment
                accumulated += text
                positions = [accumulated.find(s) for s in stop_list if s and s in accumulated]
                if positions:
                    stopped = True
                    cutoff = min(positions)
                    text = text[:max(0, len(text) - (len(accumulated) - cutoff))]
                return text if not self.strip_thinking else ""

            text = emit(current.item())
            if text:
                yield text
            while generated < max_tokens and not stopped:
                # Draft head state is speculative and is never persisted across
                # cycles or requests. Only backbone-approved tokens are emitted.
                draft_cache = self.model.make_mtp_cache()
                draft_hidden, draft_token = current_hidden, current
                drafts = []
                for _ in range(min(max(1, min(self.mtp_depth, 3)), max_tokens - generated)):
                    d_logits, draft_hidden = self.model.mtp_forward(
                        draft_hidden, draft_token.reshape(1, 1),
                        mtp_cache=draft_cache, return_hidden=True,
                    )
                    draft_token = mx.argmax(d_logits[:, -1, :], axis=-1)
                    mx.eval(draft_token)
                    drafts.append(draft_token.item())
                    if drafts[-1] in tokenizer.eos_token_ids:
                        break

                before = copy.deepcopy(main_cache)
                inputs = [current.item()] + drafts
                v_logits, v_hidden = self.model(
                    mx.array(inputs, dtype=mx.uint32)[None],
                    cache=main_cache, return_hidden=True,
                )
                predictions = mx.argmax(v_logits, axis=-1)
                mx.eval(predictions, v_hidden)
                predictions = predictions[0].tolist()
                accepted = 0
                for draft, predicted in zip(drafts, predictions):
                    if draft != predicted:
                        break
                    accepted += 1
                self._mtp_drafted_tokens += len(drafts)
                self._mtp_accepted_tokens += accepted

                if accepted < len(drafts):
                    main_cache = before
                    _, replay_hidden = self.model(
                        mx.array(inputs[:accepted + 1], dtype=mx.uint32)[None],
                        cache=main_cache, return_hidden=True,
                    )
                    mx.eval([layer.state for layer in main_cache], replay_hidden)
                    current_hidden = replay_hidden[:, -1:, :]
                else:
                    current_hidden = v_hidden[:, -1:, :]
                # Accepted drafts plus one authoritative backbone prediction.
                approved = drafts[:accepted] + [predictions[accepted]]
                current = mx.array([approved[-1]], dtype=mx.uint32)
                for token_id in approved:
                    if generated >= max_tokens or stopped:
                        break
                    text = emit(token_id)
                    if text:
                        yield text

            detokenizer.finalize()
            tail = detokenizer.last_segment
            if tail and not stopped:
                accumulated += tail
                if not self.strip_thinking:
                    yield tail
            if self.strip_thinking:
                final_text = self._strip_thinking_tags(accumulated)
                for stop in stop_list:
                    if stop:
                        final_text = final_text.split(stop)[0]
                if final_text:
                    yield final_text
            duration = time.perf_counter() - gen_started
            self._last_generation_tps = round(generated / max(duration, 1e-9), 2)
            self._total_requests += 1
            self._total_prompt_tokens += len(prompt_tokens)
            self._total_generation_tokens += generated
            if self.clear_cache_after_generation:
                self._clear_metal_cache()

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成: 支持 MTP 多 Token 投机推测、提示词前缀 KV Cache 复用与 Code Agent 极速采样"""
        self.load_model()

        # Code Agent 模式优化: 当采样温度 <= 0.2 时启用确定性贪婪采样
        eff_temp = float(temperature)
        if self.code_agent_mode and eff_temp <= 0.2:
            eff_temp = 0.0

        gen_kwargs = self._build_sampler_kwargs(eff_temp, top_p, **kwargs)
        gen_kwargs["max_tokens"] = max_tokens
        stop_list = [stop] if isinstance(stop, str) else (stop or [])

        # ── 路径 1: 贪婪 MTP，多 Token 验证与完整状态重放 ──
        # Stochastic speculative sampling needs acceptance/rejection correction;
        # use native sampling until that algorithm is implemented.
        if (self.has_mtp and self.enable_mtp and self.mtp_depth > 0
                and eff_temp == 0.0 and not gen_kwargs.get("logits_processors")):
            yield from self._stream_generate_mtp(
                prompt, max_tokens, eff_temp, stop_list, gen_kwargs
            )
            return

        # ── 降级路径: 无流式生成函数 → 非流式 → 逐字符 yield ──
        if not self.stream_generate_fn:
            full_res = self.generate(prompt, max_tokens, eff_temp, top_p, stop, **kwargs)
            for char in full_res:
                yield char
            return

        # ── 路径 2: mlx_lm 原生流式生成 ──
        with self.lock:
            tokenizer, remaining_tokens, main_cache = self._prepare_native_stream(prompt)
            gen_kwargs["prompt_cache"] = main_cache
            gen_kwargs["prefill_step_size"] = max(1, self.prefill_step_size)
            stop_list = [stop] if isinstance(stop, str) else (stop or [])

            accumulated = ""
            stopped = False

            generation_tokens = 0

            for response in self.stream_generate_fn(self.model, tokenizer, remaining_tokens, **gen_kwargs):
                if stopped:
                    break

                # 兼容 GenerationResponse 对象与字符串返回
                text_chunk = response.text if hasattr(response, "text") else str(response)

                # 采集 MLX 引擎内部报告的 TPS 指标
                if hasattr(response, "prompt_tps") and response.prompt_tps:
                    prefill_duration = self._last_prefill_seconds + len(remaining_tokens) / float(response.prompt_tps)
                    self._last_prompt_tps = round(
                        (self._last_prefill_tokens + len(remaining_tokens)) / max(prefill_duration, 1e-9), 2
                    )
                if hasattr(response, "generation_tps") and response.generation_tps:
                    self._last_generation_tps = round(float(response.generation_tps), 2)

                generation_tokens = getattr(response, "generation_tokens", generation_tokens)
                accumulated += text_chunk

                # Stop 字符串截断检测
                for s in stop_list:
                    if s in accumulated:
                        stopped = True
                        cutoff_index = accumulated.find(s)
                        text_chunk = text_chunk[:len(text_chunk) - (len(accumulated) - cutoff_index)]
                        break

                if text_chunk and not self.strip_thinking:
                    yield text_chunk
                if stopped:
                    break

            # ── Thinking 模式: 后处理剥离 ──
            if self.strip_thinking:
                final_text = self._strip_thinking_tags(accumulated)
                # Stop 截断
                for s in stop_list:
                    if s in final_text:
                        final_text = final_text.split(s)[0]
                if final_text:
                    yield final_text
            else:
                # 最终 Stop 截断 (流式中未触发的情况)
                pass

            self._total_requests += 1
            self._total_prompt_tokens += self._last_prompt_token_count
            self._total_generation_tokens += generation_tokens
            if self.clear_cache_after_generation:
                self._clear_metal_cache()

    def _clear_metal_cache(self):
        """主动清理 Metal 显存缓存，释放 MLX 内存池中的空闲分配"""
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取 Apple Metal 显存状态 (单位: MB)"""
        try:
            import mlx.core as mx
            active_fn = getattr(mx, "get_active_memory", getattr(getattr(mx, "metal", None), "get_active_memory", None))
            cache_fn = getattr(mx, "get_cache_memory", getattr(getattr(mx, "metal", None), "get_cache_memory", None))
            peak_fn = getattr(mx, "get_peak_memory", getattr(getattr(mx, "metal", None), "get_peak_memory", None))
            if active_fn and cache_fn and peak_fn:
                active_mb = round(active_fn() / (1024 * 1024), 2)
                cache_mb = round(cache_fn() / (1024 * 1024), 2)
                peak_mb = round(peak_fn() / (1024 * 1024), 2)
                return {
                    "active_memory_mb": active_mb,
                    "cache_memory_mb": cache_mb,
                    "peak_memory_mb": peak_mb,
                }
        except Exception:
            pass
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """返回引擎综合性能统计指标 (供 /metrics 端点使用)"""
        stats = {
            "model_name": self.model_name,
            "engine_type": "qwen4_exp",
            "loaded": self._loaded,
            "has_mtp": self.has_mtp,
            "enable_mtp": self.enable_mtp,
            "mtp_depth": self.mtp_depth,
            "mtp_drafted_tokens": self._mtp_drafted_tokens,
            "mtp_accepted_tokens": self._mtp_accepted_tokens,
            "code_agent_mode": self.code_agent_mode,
            "prefix_cache_hits": self._prefix_cache.hits,
            "prefix_cache": self._prefix_cache.stats(),
            "last_reused_tokens": self._last_reused_tokens,
            "last_prefill_tokens": self._last_prefill_tokens,
            "last_prefill_seconds": round(self._last_prefill_seconds, 6),
            "resolved_path": self.resolved_path,
            "kv_bits": self.kv_bits,
            "prefill_step_size": self.prefill_step_size,
            "enable_thinking": self.enable_thinking,
            "strip_thinking": self.strip_thinking,
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

    def unload_model(self) -> None:
        """完整释放模型资源，恢复为未加载状态

        释放顺序:
          1. 置空所有模型/Tokenizer 引用 (触发 Python GC 回收)
          2. 重置加载标志与统计缓存
          3. 清理 Metal 显存缓存
          4. 强制 Python GC 回收
        """
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.generate_fn = None
            self.stream_generate_fn = None
            self._loaded = False
            self._prefix_cache.clear()
            self._last_reused_tokens = self._last_prefill_tokens = 0
            self._last_prompt_token_count = 0
            self._last_prefill_seconds = 0.0
            self._cached_count_tokens.cache_clear()
        self._clear_metal_cache()
        import gc
        gc.collect()
        print(f"[Qwen4ExpEngine] 模型 '{self.model_name}' 已卸载，显存已释放")
