"""
AI Code Service - 编程助手专有快捷 API 路由
提供代码解释、代码重构、单测生成、Bug 修复、行内编辑、代码审查与文档字符串生成接口
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from schemas import (
    CodeExplainRequest,
    CodeRefactorRequest,
    CodeTestGenerateRequest,
    CodeFixBugsRequest,
    CodeEditRequest,
    CodeReviewRequest,
    CodeDocstringRequest,
    ChatCompletionResponse,
    ChatCompletionRequest,
    ChatMessage,
)
from engine import ModelManager
from .chat import create_chat_completion

router = APIRouter(prefix="/v1/code", tags=["Code Assistant Tools"])


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/explain", response_model=ChatCompletionResponse)
async def explain_code(
    req: CodeExplainRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """代码解释接口"""
    content = f"请详细解释以下 {req.language} 代码的作用、逻辑与关键步骤：\n\n```{req.language}\n{req.code}\n```"
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.3,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/refactor", response_model=ChatCompletionResponse)
async def refactor_code(
    req: CodeRefactorRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """代码重构接口"""
    content = f"请根据以下要求重构代码：\n重构要求：{req.instruction}\n\n原始 {req.language} 代码：\n```{req.language}\n{req.code}\n```"
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.2,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/generate-tests", response_model=ChatCompletionResponse)
async def generate_tests(
    req: CodeTestGenerateRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """单测生成接口"""
    content = f"请为以下 {req.language} 代码编写完整的单元测试（使用 {req.framework} 框架）：\n\n```{req.language}\n{req.code}\n```"
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.2,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/fix-bugs", response_model=ChatCompletionResponse)
async def fix_bugs(
    req: CodeFixBugsRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """Bug 修复接口"""
    err_info = f"\n报错信息/现象：{req.error_message}" if req.error_message else ""
    content = f"请分析以下 {req.language} 代码中的 Bug，解释原因并给出修改后的正确代码：{err_info}\n\n```{req.language}\n{req.code}\n```"
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.2,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/edit", response_model=ChatCompletionResponse)
async def edit_code(
    req: CodeEditRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """行内代码编辑接口"""
    content = (
        f"请按照以下要求修改代码：\n"
        f"编辑要求：{req.instruction}\n\n"
        f"原始 {req.language} 代码：\n```{req.language}\n{req.code}\n```"
    )
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.2,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/review", response_model=ChatCompletionResponse)
async def review_code(
    req: CodeReviewRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """代码审查接口"""
    content = (
        f"请对以下 {req.language} 代码进行代码审查，指出潜在问题、性能隐患、可读性问题，"
        f"并给出具体的修改建议：\n\n```{req.language}\n{req.code}\n```"
    )
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.3,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)


@router.post("/docstring", response_model=ChatCompletionResponse)
async def generate_docstring(
    req: CodeDocstringRequest,
    response: Response,
    manager: ModelManager = Depends(get_model_manager),
):
    """文档字符串/注释生成接口"""
    content = (
        f"请为以下 {req.language} 代码生成清晰、完整的文档字符串/注释：\n\n"
        f"```{req.language}\n{req.code}\n```"
    )
    chat_req = ChatCompletionRequest(
        model=req.model or manager.config.default_model,
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.2,
    )
    return await create_chat_completion(chat_req, response=response, manager=manager)
