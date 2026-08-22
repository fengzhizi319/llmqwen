"""
AI Code Service - API 数据结构定义
符合 OpenAI 规范及扩展的编程服务模型
"""

import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息文本内容")
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=262144)
    max_output_tokens: Optional[int] = Field(default=None, ge=1, le=262144)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = Field(default=None, description="随机种子，用于可复现生成")
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Optional[Union[str, List[str]]] = ""
    suffix: Optional[str] = Field(default=None, description="FIM 补全的后缀代码 (Suffix)")
    max_tokens: Optional[int] = Field(default=512, ge=1, le=262144)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = Field(default=None, description="随机种子，用于可复现生成")
    echo: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ModelPermission(BaseModel):
    id: str = "modelperm-1"
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ai-code-service"
    description: Optional[str] = ""
    context_length: int = Field(default=262144, description="最大上下文长度 (Tokens)")
    permission: List[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# 编程专属快捷接口 Request Schemas

class BaseCodeRequest(BaseModel):
    model: Optional[str] = None
    language: Optional[str] = Field(default="python", description="编程语言名称")
    stream: bool = Field(default=False, description="是否使用流式 SSE 实时输出")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=262144)


class CodeExplainRequest(BaseCodeRequest):
    code: str = Field(..., description="待解释的代码片段")


class CodeRefactorRequest(BaseCodeRequest):
    code: str = Field(..., description="待重构的代码")
    instruction: str = Field(..., description="重构需求说明，例如：提升性能、转换为异步代码")


class CodeTestGenerateRequest(BaseCodeRequest):
    code: str = Field(..., description="需要生成单测的目标代码")
    framework: Optional[str] = Field(default="pytest", description="测试框架，如 pytest, unittest, jest")


class CodeFixBugsRequest(BaseCodeRequest):
    code: str = Field(..., description="包含 Bug 的代码段")
    error_message: Optional[str] = Field(default=None, description="错误日志或报错信息")


class CodeEditRequest(BaseCodeRequest):
    code: str = Field(..., description="待编辑的代码片段")
    instruction: str = Field(..., description="编辑要求，例如：将函数改写为异步")


class CodeReviewRequest(BaseCodeRequest):
    code: str = Field(..., description="待审查的代码片段")


class CodeDocstringRequest(BaseCodeRequest):
    code: str = Field(..., description="需要生成文档字符串的代码片段")
