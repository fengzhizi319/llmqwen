"""
AI Code Service - 健康检查与服务度量路由
提供系统运行状态、Metal 显存、LRU 缓存命中率与 TPS 性能监控
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, Request
from engine import ModelManager

router = APIRouter(tags=["Health & Status"])

START_TIME = time.time()


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("/health")
@router.get("/v1/health")
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """健康状态检查"""
    active_engines = {}
    for name, engine in manager.engines.items():
        active_engines[name] = engine.health_check()

    return {
        "status": "ok",
        "service": "AI Code Service",
        "version": "1.1.0",
        "default_model": manager.config.default_model,
        "available_models": manager.get_model_names(),
        "active_engines": active_engines,
        "use_mock": manager.config.use_mock,
    }


@router.get("/metrics")
async def get_metrics(manager: ModelManager = Depends(get_model_manager)):
    """服务运行指标、显存与缓存性能监控"""
    uptime_seconds = int(time.time() - START_TIME)
    report = manager.get_metrics_report()
    
    # 尝试获取全局 Metal 显存状态
    metal_memory = {}
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            metal_memory = {
                "active_memory_mb": round(mx.metal.get_active_memory() / (1024 * 1024), 2),
                "cache_memory_mb": round(mx.metal.get_cache_memory() / (1024 * 1024), 2),
                "peak_memory_mb": round(mx.metal.get_peak_memory() / (1024 * 1024), 2),
            }
    except Exception:
        pass

    return {
        "uptime_seconds": uptime_seconds,
        "configured_models_count": len(manager.get_model_names()),
        "loaded_engines_count": len(manager.engines),
        "loaded_engines": list(manager.engines.keys()),
        "server_host": manager.config.server.host,
        "server_port": manager.config.server.port,
        "performance": {
            "cache": report.get("cache", {}),
            "metal_memory": metal_memory,
            "engine_stats": report.get("engines", {}),
        },
    }
