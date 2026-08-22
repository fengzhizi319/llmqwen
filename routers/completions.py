"""
AI Code Service - Standard Text/Code Completions API 路由
支持 OpenAI /v1/completions 以及 Fill-In-The-Middle (FIM) 代码补全、异步流式与响应缓存
"""

import json
import time
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse

from schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    UsageInfo,
)
from engine import ModelManager
from engine.cache import ResponseCache

router = APIRouter(tags=["Completions & FIM Autocomplete"])


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/v1/completions")
@router.post("/completions")
async def create_completion(
    req: CompletionRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """OpenAI 兼容的代码补全接口，支持 FIM (Fill-In-The-Middle)"""
    model_name = req.model or manager.config.default_model

    if model_name not in manager.get_model_names() and not manager.config.use_mock:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    # 提取 prompt 文本
    if isinstance(req.prompt, list):
        prefix_prompt = req.prompt[0] if req.prompt else ""
    else:
        prefix_prompt = req.prompt or ""

    # 如果提供了 suffix，使用 FIM 格式构建 Prompt
    if req.suffix:
        prompt = manager.build_fim_prompt(prefix_prompt, req.suffix)
    else:
        prompt = prefix_prompt

    engine = manager.get_engine(model_name)
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 计算 Prompt Tokens 并校验上下文长度
    prompt_tokens = engine.count_tokens(prompt)
    manager.check_prompt_length(model_name, prompt_tokens)

    cache_key = ResponseCache.generate_key(
        prefix="fim" if req.suffix else "cmpl",
        model=model_name,
        prompt=prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        echo=req.echo,
        seed=req.seed,
    )

    stream_chunk_size = (
        manager.config.performance.stream_chunk_size
        if manager.config.performance
        else 1
    )

    # 流式响应（异步非阻塞生成器）
    if req.stream:
        async def async_stream_generator():
            try:
                async with manager.generation_semaphore:
                    tokens_gen = engine.async_stream_generate(
                        prompt=prompt,
                        max_tokens=req.max_tokens or 512,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        stop=req.stop,
                        seed=req.seed,
                        chunk_size=stream_chunk_size,
                    )
                    async for chunk_text in tokens_gen:
                        chunk_payload = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": chunk_text,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"

                done_payload = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_payload = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(async_stream_generator(), media_type="text/event-stream")

    # 非流式响应
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
                        max_tokens=req.max_tokens or 512,
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
                raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")

        final_text = (prefix_prompt + output_text) if (req.echo and not req.suffix) else output_text
        if response is not None:
            response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
            response.headers["X-Completion-Tokens"] = str(completion_tokens)

        return CompletionResponse(
            id=request_id,
            object="text_completion",
            created=created_time,
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=final_text,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
