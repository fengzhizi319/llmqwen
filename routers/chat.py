"""
AI Code Service - Chat Completions API 路由
符合 OpenAI /v1/chat/completions 标准规范，支持异步非阻塞流式、LRU 缓存与性能追踪
"""

import json
import time
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    UsageInfo,
)
from engine import ModelManager
from engine.cache import ResponseCache

router = APIRouter(tags=["Chat Completions"])


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/v1/chat/completions")
async def create_chat_completion(
    req: ChatCompletionRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """OpenAI 兼容的对话补全接口"""
    model_name = req.model or manager.config.default_model

    # 验证模型合法性
    if model_name not in manager.get_model_names() and not manager.config.use_mock:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {manager.get_model_names()}",
        )

    # 转换为 dict 列表并构建 Prompt
    msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = manager.build_chat_prompt(msg_dicts)

    engine = manager.get_engine(model_name)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 计算 Prompt Tokens 并校验上下文长度
    prompt_tokens = engine.count_tokens(prompt)
    manager.check_prompt_length(model_name, prompt_tokens)

    # 生成缓存 Key
    cache_key = ResponseCache.generate_key(
        prefix="chat",
        model=model_name,
        prompt=prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        seed=req.seed,
    )

    stream_chunk_size = (
        manager.config.performance.stream_chunk_size
        if manager.config.performance
        else 1
    )

    # 流式 SSE 响应（通过异步生成器输出，不阻塞事件循环）
    if req.stream:
        async def async_stream_generator():
            try:
                async with manager.generation_semaphore:
                    tokens_gen = engine.async_stream_generate(
                        prompt=prompt,
                        max_tokens=req.max_tokens or 2048,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        stop=req.stop,
                        seed=req.seed,
                        chunk_size=stream_chunk_size,
                    )
                    async for chunk_text in tokens_gen:
                        chunk_payload = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"

                # 结束 Chunk
                done_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_payload = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500,
                    }
                }
                yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(async_stream_generator(), media_type="text/event-stream")

    # 非流式响应（优先命中缓存，未命中则异步非阻塞推理）
    else:
        cached_result = manager.cache.get(cache_key)
        if cached_result is not None:
            output_text, completion_tokens = cached_result
            if response is not None:
                response.headers["X-Cache"] = "HIT"
        else:
            try:
                async with manager.generation_semaphore:
                    output_text = await engine.async_generate(
                        prompt=prompt,
                        max_tokens=req.max_tokens or 2048,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        stop=req.stop,
                        seed=req.seed,
                    )
                completion_tokens = engine.count_tokens(output_text)
                manager.cache.set(cache_key, (output_text, completion_tokens))
                if response is not None:
                    response.headers["X-Cache"] = "MISS"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

        if response is not None:
            response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
            response.headers["X-Completion-Tokens"] = str(completion_tokens)

        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created_time,
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
