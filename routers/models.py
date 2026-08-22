"""
AI Code Service - 模型查询 API 路由
符合 OpenAI /v1/models 规范
"""

import time
from fastapi import APIRouter, HTTPException, Depends, Request
from schemas import ModelListResponse, ModelInfo, ModelPermission
from engine import ModelManager

router = APIRouter(tags=["Models"])


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """列出所有已配置和可用的模型"""
    models_list = []
    current_time = int(time.time())

    for model_name in manager.get_model_names():
        spec = manager.get_model_info(model_name)
        desc = spec.description if spec else ""
        ctx_len = spec.context_length if spec else 262144
        models_list.append(
            ModelInfo(
                id=model_name,
                created=current_time,
                description=desc,
                context_length=ctx_len,
                permission=[ModelPermission(created=current_time)],
            )
        )

    # 如果配置为空或开启 mock
    if not models_list and manager.config.use_mock:
        models_list.append(
            ModelInfo(
                id="qwen3.8-27b",
                created=current_time,
                description="Mock Qwen3.8 27B Model (256K Context)",
                context_length=262144,
            )
        )

    return ModelListResponse(object="list", data=models_list)


@router.get("/v1/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str, manager: ModelManager = Depends(get_model_manager)):
    """获取单个模型详细信息"""
    spec = manager.get_model_info(model_id)
    if not spec and model_id not in manager.get_model_names() and not manager.config.use_mock:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    desc = spec.description if spec else "Model Information"
    ctx_len = spec.context_length if spec else 262144
    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        description=desc,
        context_length=ctx_len,
        permission=[ModelPermission()],
    )
