"""
AI Code Service - Chat Completions & Responses API 路由
符合 OpenAI /v1/chat/completions 与 /v1/responses 标准规范，支持异步非阻塞流式、LRU 缓存与性能追踪
"""

import json
from contextlib import aclosing
import time
import uuid
from typing import Dict, Any, List, Union
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from engine import ModelManager
from engine.cache import ResponseCache
from engine import tool_format

router = APIRouter(tags=["Chat Completions & Responses"])


def normalize_content(content: Any) -> str:
    """将 OpenAI 多模态 content (str 或 parts 列表) 归一化为纯文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
@router.post("/v1/responses")
@router.post("/responses")
async def create_chat_completion(
    req: ChatCompletionRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """OpenAI 兼容的对话补全与 Responses 接口 (支持 /v1/chat/completions 与 /v1/responses)"""
    model_name = req.model or manager.config.default_model

    # 验证模型合法性
    if model_name not in manager.get_model_names() and not manager.config.use_mock:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {manager.get_model_names()}",
        )

    # 标准化解析 messages 与 input/instructions
    msg_dicts = []
    if req.instructions:
        msg_dicts.append({"role": "system", "content": req.instructions})

    if req.messages:
        for m in req.messages:
            entry: Dict[str, Any] = {"role": m.role, "content": normalize_content(m.content)}
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
            msg_dicts.append(entry)
    elif req.input is not None:
        if isinstance(req.input, str):
            msg_dicts.append({"role": "user", "content": req.input})
        elif isinstance(req.input, list):
            for item in req.input:
                if isinstance(item, dict):
                    msg_dicts.append({
                        "role": item.get("role", "user"),
                        "content": item.get("content", str(item)),
                    })
                elif isinstance(item, str):
                    msg_dicts.append({"role": "user", "content": item})
                elif hasattr(item, "content"):
                    msg_dicts.append({"role": getattr(item, "role", "user"), "content": str(item.content)})

    if not msg_dicts:
        msg_dicts = [{"role": "user", "content": ""}]

    use_tools = req.tools if req.tools and req.tool_choice != "none" else None

    # 思考模式控制：在 Code Agent 模式或工具调用场景下，如果客户端未显式要求 thinking，默认关闭以减少首字延迟和不必要的多轮思考 token
    enable_thinking = req.enable_thinking
    if enable_thinking is None and (use_tools or (manager.config.performance and manager.config.performance.code_agent_mode)):
        enable_thinking = False

    prompt = manager.build_chat_prompt(msg_dicts, tools=use_tools, enable_thinking=enable_thinking)

    # 温度与采样参数：Code Agent 模式下默认 0.0 (argmax)，大幅提高 MTP 投机推测命中率与代码生成稳定性
    if req.temperature is not None:
        effective_temp = req.temperature
    elif manager.config.performance and manager.config.performance.code_agent_mode:
        effective_temp = manager.config.performance.default_temperature
    else:
        effective_temp = 0.7

    # 工具调用模式下注入原生结束标签作为 stop 序列，避免多余生成
    stop_seqs: List[str] = []
    if req.stop:
        stop_seqs += [req.stop] if isinstance(req.stop, str) else list(req.stop)
    if use_tools:
        for t_stop in [tool_format.TC_CLOSE, "<|im_end|>"]:
            if t_stop not in stop_seqs:
                stop_seqs.append(t_stop)
    stop_param = stop_seqs or None

    engine = manager.get_engine(model_name)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    max_toks = req.max_output_tokens or req.max_tokens or 2048

    # 计算 Prompt Tokens 并校验上下文长度
    prompt_tokens = engine.count_tokens(prompt)
    manager.check_prompt_length(model_name, prompt_tokens)

    # 生成缓存 Key
    cache_key = ResponseCache.generate_key(
        prefix="chat",
        model=model_name,
        prompt=prompt,
        temperature=effective_temp,
        top_p=req.top_p,
        stop=req.stop,
        seed=req.seed,
        max_tokens=max_toks,
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
                        max_tokens=max_toks,
                        temperature=effective_temp,
                        top_p=req.top_p,
                        stop=stop_param,
                        seed=req.seed,
                        chunk_size=stream_chunk_size,
                    )
                    full_output_text = ""
                    region_filter = tool_format.StreamRegionFilter() if use_tools else None
                    raw_output_text = ""
                    async with aclosing(tokens_gen):
                        async for chunk_text in tokens_gen:
                            if region_filter is not None:
                                raw_output_text += chunk_text
                                visible = region_filter.feed(chunk_text)
                                if not visible:
                                    continue
                                full_output_text += visible
                            else:
                                visible = chunk_text
                                full_output_text += visible
                            chunk_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": visible},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"

                    # 流结束: 若已进入工具/思考区域则吞掉残余文本，否则冲刷尾部可见内容。
                    # 思考未闭合时保留尾部 (无 tools 的普通回答常被 max_tokens 截断于思考区，
                    # 客户端依赖累积的 delta 展示进度，解析工具调用仍使用 raw 文本)。
                    tool_calls: List[Dict[str, Any]] = []
                    if region_filter is not None:
                        if region_filter.has_tool_region():
                            region_filter.flush()
                        elif region_filter.in_thinking:
                            keep_tail = region_filter.flush()
                            full_output_text += keep_tail
                            if keep_tail:
                                tail_payload = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": keep_tail},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(tail_payload, ensure_ascii=False)}\n\n"
                        else:
                            tail_visible = region_filter.flush()
                            full_output_text += tail_visible
                            if tail_visible:
                                flush_payload = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": tail_visible},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(flush_payload, ensure_ascii=False)}\n\n"
                        _, tool_calls = tool_format.parse_tool_calls(raw_output_text)

                finish_reason = "tool_calls" if tool_calls else "stop"
                # 计算实际生成 token 数
                completion_tokens = engine.count_tokens(raw_output_text if region_filter is not None else full_output_text)

                done_delta: Dict[str, Any] = {}
                if tool_calls:
                    indexed_calls = []
                    for idx, tc in enumerate(tool_calls):
                        tc_dict = dict(tc)
                        tc_dict["index"] = idx
                        indexed_calls.append(tc_dict)
                    done_delta["tool_calls"] = indexed_calls
                done_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": done_delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
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

    # 非流式响应（优先命中缓存，未命中则异步非阻塞推理；工具调用请求不缓存以保证 agent 循环实时性）
    else:
        cached_result = None if use_tools else manager.cache.get(cache_key)
        if cached_result is not None:
            output_text, completion_tokens = cached_result
            if response is not None:
                response.headers["X-Cache"] = "HIT"
        else:
            try:
                async with manager.generation_semaphore:
                    output_text = await engine.async_generate(
                        prompt=prompt,
                        max_tokens=max_toks,
                        temperature=effective_temp,
                        top_p=req.top_p,
                        stop=stop_param,
                        seed=req.seed,
                    )
                completion_tokens = engine.count_tokens(output_text)
                if not use_tools:
                    manager.cache.set(cache_key, (output_text, completion_tokens))
                if response is not None:
                    response.headers["X-Cache"] = "MISS"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

        if response is not None:
            response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
            response.headers["X-Completion-Tokens"] = str(completion_tokens)

        visible_text = output_text
        tool_call_dicts: List[Dict[str, Any]] = []
        if use_tools:
            visible_text, tool_call_dicts = tool_format.parse_tool_calls(output_text)
            if tool_call_dicts:
                visible_text = visible_text.replace(tool_format.THINK_CLOSE, "").strip()
        if not tool_call_dicts:
            visible_text = tool_format.clean_think_tags(visible_text)

        message = ChatMessage(
            role="assistant",
            content=visible_text,
            tool_calls=[ToolCall(**tc) for tc in tool_call_dicts] or None,
        )
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created_time,
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=message,
                    finish_reason="tool_calls" if tool_call_dicts else "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
