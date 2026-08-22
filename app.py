"""
AI Code Service - 基于 LLM (MLX/Qwen) 的高性能编程助手服务
提供 OpenAI 兼容 API、FIM 代码自动补全及专用编程助手工具接口
"""

import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import load_config, AppConfig
from engine import ModelManager
from routers import (
    chat_router,
    completions_router,
    code_router,
    models_router,
    health_router,
)


def create_app(config_path: str = "config.yaml") -> FastAPI:
    config: AppConfig = load_config(config_path)
    model_manager = ModelManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("=" * 60)
        print("  🚀 AI Code Service 正在启动...")
        print(f"  📌 默认模型: {config.default_model}")
        print(f"  📌 配置模型列表: {model_manager.get_model_names()}")
        print(f"  📌 Mock 状态: {'开启 (Mock 仿真模式)' if config.use_mock else '关闭 (实际模型推理模式)'}")
        print("=" * 60)

        # 预热默认模型，降低首个请求冷启动延迟
        if not config.use_mock and config.default_model in model_manager.get_model_names():
            try:
                default_engine = model_manager.get_engine(config.default_model)
                if hasattr(default_engine, "load_model"):
                    default_engine.load_model()
                    print(f"[Startup] 默认模型 {config.default_model} 已预热")
            except Exception as e:
                print(f"[Startup] 默认模型预热失败: {e}")

        yield
        print("  👋 AI Code Service 已安全关闭")

    app = FastAPI(
        title="AI Code Service",
        description="基于编程 LLM (Qwen/MLX) 的 API 服务，完全兼容 OpenAI 接口与 FIM 代码补全",
        version="1.1.0",
        lifespan=lifespan,
    )

    # 存储全局 ModelManager 引用
    app.state.config = config
    app.state.model_manager = model_manager

    # CORS 跨域配置 (支持 IDE 插件与 Web 客户端)
    origins = config.server.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key 鉴权与 Request ID 中间件
    @app.middleware("http")
    async def security_and_tracing_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:10]}"
        start_time = time.time()

        # 校验 API Key (如果配置了 api_key)
        api_key = config.server.api_key
        public_paths = {"/", "/health", "/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
        
        if api_key and request.url.path not in public_paths:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing or invalid Authorization header. Expected Bearer token."},
                )
            token = auth_header.split(" ", 1)[1]
            if token != api_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid API Key."},
                )

        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response

    # 注册路由
    app.include_router(chat_router)
    app.include_router(completions_router)
    app.include_router(code_router)
    app.include_router(models_router)
    app.include_router(health_router)

    @app.get("/", tags=["Root"])
    async def root():
        """服务根路径节点信息"""
        return {
            "service": "AI Code Service",
            "version": "1.1.0",
            "description": "基于 LLM 的高可用 AI 编程服务",
            "endpoints": {
                "chat": "/v1/chat/completions",
                "completions_fim": "/v1/completions",
                "code_explain": "/v1/code/explain",
                "code_refactor": "/v1/code/refactor",
                "code_generate_tests": "/v1/code/generate-tests",
                "code_fix_bugs": "/v1/code/fix-bugs",
                "models": "/v1/models",
                "health": "/health",
                "docs": "/docs",
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    server_cfg = load_config().server
    uvicorn.run(
        "app:app",
        host=server_cfg.host,
        port=server_cfg.port,
        reload=server_cfg.reload,
    )
