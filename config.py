"""
AI Code Service - 配置加载与管理模块

配置优先级: 系统环境变量 > .env 文件 > config.yaml > 代码默认值
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 自动加载项目根目录的 .env 文件 (override=False 保留系统环境变量)
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 1235
    reload: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None


class PerformanceConfig(BaseModel):
    """LLM 推理与服务性能优化配置"""
    enable_cache: bool = True               # 是否启用响应与 Prompt LRU 缓存
    cache_max_size: int = 1024              # 缓存最大条目数
    cache_ttl_seconds: int = 3600           # 缓存过期时间（秒，默认 1 小时）
    max_concurrency: int = 3                # 最大并发推理任务数 (256K 上下文推荐 2~3 并发)
    metal_cache_limit_mb: int = 4096        # Apple Silicon MLX Metal 缓存上限 (MB, 默认 4GB)
    clear_cache_after_generation: bool = False  # 是否在每次生成后显式清理 Metal 缓存
    stream_chunk_size: int = 1              # 流式响应 Token 聚合块大小
    kv_bits: Optional[int] = 8              # KV Cache 显存量化位数 (8bit 节省 50% KV 显存，长文本提速 30%~50%)
    kv_group_size: int = 64                 # KV Cache 量化分组大小
    prefill_step_size: int = 2048           # 分块预填充大小 (Chunked Prefill，平抑 256K 超大文本峰值显存)
    enable_prompt_cache: bool = True        # 是否启用前缀/系统提示词 KV Cache 复用 (<5ms 首字延迟)


class ModelSpec(BaseModel):
    path: str
    description: str = ""
    engine_type: str = "auto"  # auto, mlx_lm, mlx_vlm, mock
    context_length: int = 262144  # 默认 256K (262,144 Tokens) 超长上下文


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    default_model: str = "qwen3.8-flash-next-oq4e-mtp-128k"
    use_mock: bool = False
    system_prompt: str = """你是一个专业的编程助手。你的职责是：
1. 帮助开发者编写、调试和优化代码
2. 解释代码逻辑和技术概念
3. 提供最佳实践和设计模式建议
4. 支持多种编程语言和框架

回答时请：
- 代码示例要完整、可运行
- 解释要清晰简洁
- 遵循行业最佳实践
- 必要时提供多种解决方案"""
    models: Dict[str, ModelSpec] = Field(default_factory=dict)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """加载并解析 YAML 配置文件，支持环境变量覆盖"""
    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    # 解析模型字典
    raw_models = config_dict.get("models", {})
    parsed_models = {}
    for name, spec in raw_models.items():
        if isinstance(spec, dict):
            parsed_models[name] = ModelSpec(**spec)
        elif isinstance(spec, ModelSpec):
            parsed_models[name] = spec

    server_data = config_dict.get("server", {})
    if "api_key" not in server_data and os.getenv("API_KEY"):
        server_data["api_key"] = os.getenv("API_KEY")

    if os.getenv("HOST"):
        server_data["host"] = os.getenv("HOST")
    if os.getenv("PORT"):
        server_data["port"] = int(os.getenv("PORT"))

    perf_data = config_dict.get("performance", {})
    if os.getenv("ENABLE_CACHE"):
        perf_data["enable_cache"] = os.getenv("ENABLE_CACHE").lower() in ("true", "1")

    return AppConfig(
        server=ServerConfig(**server_data),
        performance=PerformanceConfig(**perf_data),
        default_model=config_dict.get("default_model", os.getenv("DEFAULT_MODEL", "qwen3.8-flash-next-oq4e-mtp-128k")),
        use_mock=config_dict.get("use_mock", os.getenv("USE_MOCK", "").lower() in ("true", "1")),
        system_prompt=config_dict.get("system_prompt", AppConfig().system_prompt),
        models=parsed_models,
    )
