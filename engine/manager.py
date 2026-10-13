"""
AI Code Service - 模型统一管理器
管理多个 LLM 模型引擎生命周期、Prompt 构建、响应缓存与性能指标聚合
"""

import asyncio
from typing import Dict, List, Optional, Any

from fastapi import HTTPException

from config import AppConfig, ModelSpec
from .base import BaseModelEngine
from .mock_engine import MockModelEngine
from .mlx_engine import MLXModelEngine
from .qwen4_exp_engine import Qwen4ExpEngine
from .cache import ResponseCache


class ModelManager:
    """全局模型管理器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.engines: Dict[str, BaseModelEngine] = {}
        self._generation_semaphore: Optional[asyncio.Semaphore] = None

        # 初始化响应与 Prompt 缓存器
        perf = getattr(config, "performance", None)
        enable_cache = perf.enable_cache if perf else True
        max_size = perf.cache_max_size if perf else 1024
        ttl = perf.cache_ttl_seconds if perf else 3600
        self.cache = ResponseCache(max_size=max_size, ttl_seconds=ttl, enabled=enable_cache)

    @property
    def generation_semaphore(self) -> asyncio.Semaphore:
        """限制同时进行的生成调用并发数"""
        if self._generation_semaphore is None:
            max_concurrency = 4
            if self.config.performance:
                max_concurrency = self.config.performance.max_concurrency
            self._generation_semaphore = asyncio.Semaphore(max_concurrency)
        return self._generation_semaphore

    def get_model_names(self) -> List[str]:
        return list(self.config.models.keys())

    def get_model_info(self, model_name: str) -> Optional[ModelSpec]:
        return self.config.models.get(model_name)

    def check_prompt_length(self, model_name: str, prompt_tokens: int) -> None:
        """校验 prompt token 数是否超过模型上下文长度"""
        spec = self.get_model_info(model_name)
        if spec is None:
            return
        if prompt_tokens > spec.context_length:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt too long ({prompt_tokens} tokens) exceeds model "
                    f"'{model_name}' context length ({spec.context_length})"
                ),
            )

    def get_engine(self, model_name: Optional[str] = None) -> BaseModelEngine:
        """获取或创建指定模型的推理引擎，带有自动 fallback"""
        target_name = model_name or self.config.default_model
        if target_name not in self.config.models:
            # 如果请求的模型不在配置中，使用默认模型或 Mock 引擎
            if self.config.default_model in self.config.models:
                target_name = self.config.default_model
            else:
                return MockModelEngine(model_name=target_name or "default-mock")

        if target_name in self.engines:
            return self.engines[target_name]

        # 如果开启了全局 use_mock，直接使用 MockEngine
        if self.config.use_mock:
            engine = MockModelEngine(model_name=target_name)
            self.engines[target_name] = engine
            return engine

        spec = self.config.models[target_name]
        perf = getattr(self.config, "performance", None)
        metal_limit = perf.metal_cache_limit_mb if perf else 8192
        clear_cache = perf.clear_cache_after_generation if perf else False
        kv_bits = perf.kv_bits if perf else 8
        kv_group_size = perf.kv_group_size if perf else 64
        prefill_step = perf.prefill_step_size if perf else 2048
        enable_prompt_cache = perf.enable_prompt_cache if perf else True
        enable_mtp = getattr(perf, "enable_mtp", True) if perf else True
        mtp_depth = getattr(perf, "mtp_depth", 3) if perf else 3
        code_agent_mode = getattr(perf, "code_agent_mode", True) if perf else True

        try:
            # 根据 engine_type 选择引擎实现
            # qwen4_exp 架构使用专用引擎，其他架构使用通用 MLX 引擎
            if spec.engine_type == "qwen4_exp":
                engine = Qwen4ExpEngine(
                    model_name=target_name,
                    model_path=spec.path,
                    metal_cache_limit_mb=metal_limit,
                    clear_cache_after_generation=clear_cache,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    prefill_step_size=prefill_step,
                    enable_prompt_cache=enable_prompt_cache,
                    enable_mtp=enable_mtp,
                    mtp_depth=mtp_depth,
                    code_agent_mode=code_agent_mode,
                    prompt_cache_max_entries=perf.prompt_cache_max_entries if perf else 4,
                    prompt_cache_max_tokens=perf.prompt_cache_max_tokens if perf else 8192,
                )
            else:
                engine = MLXModelEngine(
                    model_name=target_name,
                    model_path=spec.path,
                    engine_type=spec.engine_type,
                    metal_cache_limit_mb=metal_limit,
                    clear_cache_after_generation=clear_cache,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    prefill_step_size=prefill_step,
                    enable_prompt_cache=enable_prompt_cache,
                    enable_mtp=enable_mtp,
                    mtp_depth=mtp_depth,
                    code_agent_mode=code_agent_mode,
                )
            self.engines[target_name] = engine
            return engine
        except Exception as e:
            print(f"[ModelManager] Warning: 无法初始化 MLX 引擎 ({e})，将回退到 Mock 引擎。")
            engine = MockModelEngine(model_name=target_name)
            self.engines[target_name] = engine
            return engine

    GUARDRAIL = (
        "\n\n【核心约束】请直接输出解答与完整的带注释代码，禁止输出工具调用标签。"
        "如果未收到代码正文，请直接引导用户提供代码或在编辑器中选中代码。"
    )

    def build_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """根据消息构建标准 Chat Prompt (符合 Qwen/ChatML 格式)

        - tools 非空时注入模型原生工具声明块 (见 engine/tool_format.py)，并跳过防工具标签约束
        - assistant 历史消息中的 tool_calls 会序列化回模型原生调用格式
        - role=tool 的工具执行结果按模板约定包装为 user 消息的 tool_result 块
        """
        from . import tool_format

        formatted_messages = [dict(m) for m in messages]

        user_system = ""
        rest: List[Dict[str, Any]] = []
        for msg in formatted_messages:
            if msg.get("role") == "system" and not user_system:
                user_system = msg.get("content") or ""
            else:
                rest.append(msg)

        if tools:
            system_content = tool_format.render_tools_block(tools)
            if user_system.strip():
                system_content += "\n\n" + user_system
        else:
            if not user_system:
                system_content = system_prompt or self.config.system_prompt
            elif "【核心约束】" in user_system:
                system_content = user_system
            else:
                system_content = user_system + self.GUARDRAIL

        prompt_parts = []
        if system_content:
            prompt_parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")

        # 合并连续的 tool 消息为一个 user 消息块 (严格符合 Qwen 官方 chat_template.jinja)
        grouped_messages: List[Dict[str, Any]] = []
        for msg in rest:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "tool":
                if grouped_messages and grouped_messages[-1].get("role") == "tool":
                    grouped_messages[-1]["tool_items"].append(content)
                else:
                    grouped_messages.append({"role": "tool", "tool_items": [content]})
            else:
                grouped_messages.append(msg)

        for gmsg in grouped_messages:
            role = gmsg.get("role", "user")
            if role == "assistant":
                content = tool_format.clean_think_tags(gmsg.get("content") or "")
                if gmsg.get("tool_calls"):
                    raw_calls = [
                        tc if isinstance(tc, dict) else tc.model_dump()
                        for tc in gmsg["tool_calls"]
                    ]
                    block = tool_format.serialize_assistant_tool_calls(raw_calls)
                    content = f"{content}\n\n{block}" if content.strip() else block
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                tool_blocks = [tool_format.render_tool_message(c) for c in gmsg.get("tool_items", [])]
                inner = "\n".join(tool_blocks)
                prompt_parts.append(f"<|im_start|>user\n{inner}<|im_end|>")
            else:
                content = gmsg.get("content") or ""
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        tail = "<|im_start|>assistant\n"
        if enable_thinking is False:
            tail += f"{tool_format.THINK_OPEN}\n\n{tool_format.THINK_CLOSE}\n\n"
        else:
            tail += f"{tool_format.THINK_OPEN}\n"
        prompt_parts.append(tail)
        return "\n".join(prompt_parts)

    def build_fim_prompt(self, prefix: str, suffix: Optional[str] = None) -> str:
        """根据前缀和后缀构建 Qwen 标准 Fill-In-The-Middle (FIM) 代码补全 Prompt"""
        if not suffix:
            return prefix
        
        # Qwen / Code LLM 标准 FIM 标记格式
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

    def get_metrics_report(self) -> Dict[str, Any]:
        """获取综合性能与运行状态报告"""
        report: Dict[str, Any] = {
            "cache": self.cache.get_stats(),
            "engines": {},
        }
        for name, engine in self.engines.items():
            if hasattr(engine, "get_stats"):
                report["engines"][name] = engine.get_stats()
        return report

    def unload_engine(self, model_name: str) -> bool:
        """卸载指定模型的引擎，释放模型权重与显存"""
        engine = self.engines.pop(model_name, None)
        if engine is None:
            return False
        engine.unload_model()
        return True
