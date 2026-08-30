"""
本地真实模型集成 UT — 全链路推理验证。

运行方式 (在 llmqwen conda 环境中):
    RUN_REAL_MODEL_TESTS=1 python -m pytest tests/test_real_model_integration.py -v

测试覆盖四个层级:
  1. Engine 层   — 直接调用引擎 generate / stream_generate
  2. Manager 层  — Prompt 构建、引擎获取、配置校验
  3. Route 层    — 通过路由函数直接调用 (不经过 HTTP)，覆盖 Chat / Completions / Code 全链路
  4. 高级特性    — 流式 SSE、缓存命中、seed 可复现、stop 截断、thinking 模式、长上下文、代码正确性

模型名称与路径均从 config.yaml 动态读取，添加新模型时无需修改本文件。
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import load_config
from engine.mlx_engine import MLXModelEngine, resolve_local_model_path
from engine.base import BaseModelEngine
from engine import ModelManager
from engine.mock_engine import MockModelEngine
from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CodeExplainRequest,
    CodeRefactorRequest,
    CodeTestGenerateRequest,
    CodeFixBugsRequest,
    CodeEditRequest,
    CodeReviewRequest,
    CodeDocstringRequest,
)
from routers.chat import create_chat_completion
from routers.completions import create_completion
from routers.code import (
    explain_code,
    refactor_code,
    generate_tests,
    fix_bugs,
    edit_code,
    review_code,
    generate_docstring,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# 从 config.yaml 动态读取默认模型名称与路径
_real_config = load_config(str(CONFIG_PATH))
MODEL_NAME = _real_config.default_model
MODEL_PATH_RAW = _real_config.models[MODEL_NAME].path
# 支持本地相对路径与 ModelScope/HuggingFace 仓库 ID 两种格式
if MODEL_PATH_RAW.startswith("./models/"):
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RAW.removeprefix("./")
else:
    MODEL_PATH = Path(resolve_local_model_path(MODEL_PATH_RAW))

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS") != "1",
        reason="set RUN_REAL_MODEL_TESTS=1 to load the local model",
    ),
    pytest.mark.skipif(
        not MODEL_PATH.is_dir(),
        reason=f"local model directory not found: {MODEL_PATH}",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re

def _strip_thinking(text: str) -> str:
    """剥离 thinking 标签，返回纯回答内容"""
    think_open = "<" + "think>"
    think_close = "<" + "/think>"
    pattern = re.escape(think_open) + r".*?" + re.escape(think_close)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def _run_async(coro):
    """在同步测试中运行异步协程。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_response():
    """创建轻量 Mock Response 对象，供路由函数设置 header。"""
    resp = MagicMock()
    resp.headers = {}
    return resp


def _collect_stream(streaming_response) -> str:
    """从 StreamingResponse 中收集全部 SSE chunk 的 content，拼接为完整文本。

    同时返回 usage 信息 (来自最终 [DONE] 前的 chunk)。
    """
    from fastapi.responses import StreamingResponse
    assert isinstance(streaming_response, StreamingResponse), \
        f"预期 StreamingResponse，实际为 {type(streaming_response)}"

    full_text = ""
    usage_info = None

    # 同步迭代 body_iterator 收集 chunk
    async def _collect():
        nonlocal full_text, usage_info
        async for raw_line in streaming_response.body_iterator:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            for segment in line.strip().split("\n"):
                segment = segment.strip()
                if segment.startswith("data: ") and segment != "data: [DONE]":
                    try:
                        payload = json.loads(segment[6:])
                        choices = payload.get("choices", [{}])
                        if choices:
                            # Chat 格式: delta.content
                            delta = choices[0].get("delta", {})
                            text_chunk = delta.get("content", "")
                            # Completions 格式: text 字段
                            if not text_chunk:
                                text_chunk = choices[0].get("text", "")
                            if text_chunk:
                                full_text += text_chunk
                        # 最终 chunk 可能携带 usage
                        if "usage" in payload:
                            usage_info = payload["usage"]
                    except json.JSONDecodeError:
                        pass

    _run_async(_collect())
    return full_text, usage_info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_engine():
    """通过 ModelManager 创建并加载正确类型的引擎 (module 级单例，避免重复加载)。"""
    config = load_config(str(CONFIG_PATH))
    assert config.default_model == MODEL_NAME
    assert config.use_mock is False

    manager = ModelManager(config)
    engine = manager.get_engine(MODEL_NAME)
    if hasattr(engine, "load_model"):
        engine.load_model()
    assert engine._loaded is True
    # mlx_vlm 有 processor，mlx_lm/qwen4_exp 有 tokenizer
    assert getattr(engine, "processor", None) is not None or getattr(engine, "tokenizer", None) is not None
    yield engine


@pytest.fixture(scope="module")
def real_manager(real_engine):
    """复用已加载引擎的 ModelManager，避免双倍内存占用。"""
    config = load_config(str(CONFIG_PATH))
    manager = ModelManager(config)
    manager.engines[MODEL_NAME] = real_engine
    return manager


# ===========================================================================
# 1. Engine 层 — 直接调用引擎 generate / stream_generate
# ===========================================================================


def test_engine_chat(real_engine):
    """引擎层: 基础对话 — 模型能生成有意义的非空回答。"""
    prompt = "<|im_start|>user\n用一句话解释什么是递归。<|im_end|>\n<|im_start|>assistant\n"
    result = real_engine.generate(prompt, max_tokens=256, temperature=0.0, top_p=0.9)
    answer = _strip_thinking(result)

    assert answer.strip(), "模型回答不应为空"
    assert real_engine.count_tokens(answer) > 0
    # 回答应包含递归相关关键词
    assert any(kw in answer.lower() for kw in ("递归", "recursi", "自身", "调用", "函数", "定义"))


def test_engine_programming(real_engine):
    """引擎层: 编程 — 模型能解释代码并输出关键信息。"""
    prompt = (
        "<|im_start|>user\n解释以下代码的功能:\n"
        "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```<|im_end|>\n<|im_start|>assistant\n"
    )
    result = real_engine.generate(prompt, max_tokens=256, temperature=0.0, top_p=0.9)
    answer = _strip_thinking(result)

    assert answer.strip()
    assert any(kw in answer.lower() for kw in ("fibonacci", "斐波那契", "数列", "迭代", "fib", "循环"))


def test_engine_stream(real_engine):
    """引擎层: 流式生成 — 应产生多个非空 chunk。"""
    prompt = "<|im_start|>user\n1+1等于几？<|im_end|>\n<|im_start|>assistant\n"
    chunks = list(real_engine.stream_generate(prompt, max_tokens=128, temperature=0.0, top_p=0.9))
    result = "".join(chunks)

    assert len(chunks) >= 1, "流式生成应至少产生一个 chunk"
    assert _strip_thinking(result).strip()


def test_engine_stop_sequence(real_engine):
    """引擎层: stop 序列 — 生成应在遇到 stop 字符串时截断。"""
    prompt = "<|im_start|>user\n从1数到10<|im_end|>\n<|im_start|>assistant\n"
    result = real_engine.generate(prompt, max_tokens=256, temperature=0.0, top_p=0.9, stop="5")

    # 生成结果不应包含 stop 标记之后的内容
    assert "5" not in result or result.index("5") == len(result) - 1 or result.endswith("5")


def test_engine_thinking_output(real_engine):
    """引擎层: thinking 模式 — 模型默认输出 <think> 标签。"""
    prompt = "<|im_start|>user\n计算 15 * 7<|im_end|>\n<|im_start|>assistant\n"
    result = real_engine.generate(prompt, max_tokens=256, temperature=0.0, top_p=0.9)

    # Qwen4Exp 模型默认启用 thinking，输出应包含 <think> 标签
    assert "<think>" in result, "thinking 模型应输出 <think> 标签"
    # 剥离思考后应有实质内容
    answer = _strip_thinking(result)
    assert answer.strip()
    assert "105" in answer


# ===========================================================================
# 2. Manager 层 — Prompt 构建与引擎获取
# ===========================================================================


def test_manager_build_chat_prompt(real_manager):
    """Manager: ChatML 格式 Prompt 构建正确。"""
    prompt = real_manager.build_chat_prompt([{"role": "user", "content": "你好"}])

    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "你好" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_manager_build_chat_prompt_with_system(real_manager):
    """Manager: 带 system 消息时注入防伪造约束。"""
    prompt = real_manager.build_chat_prompt([
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "写代码"},
    ])

    assert "你是一个助手" in prompt
    assert "【核心约束】" in prompt


def test_manager_build_fim_prompt(real_manager):
    """Manager: FIM Prompt 包含 <|fim_prefix|>/<|fim_middle|> 标记。"""
    prompt = real_manager.build_fim_prompt("def add(a, b):\n", "\n    return result")

    assert "<|fim_suffix|>" in prompt
    assert "<|fim_middle|>" in prompt
    assert "def add(a, b):" in prompt
    assert "return result" in prompt


def test_manager_build_fim_prompt_no_suffix(real_manager):
    """Manager: 无 suffix 时 FIM 直接返回 prefix。"""
    assert real_manager.build_fim_prompt("print('hi')", None) == "print('hi')"


def test_manager_get_engine_returns_real_engine(real_manager):
    """Manager: 返回真实引擎而非 Mock。"""
    engine = real_manager.get_engine(MODEL_NAME)
    assert isinstance(engine, BaseModelEngine)
    assert not isinstance(engine, MockModelEngine)
    assert engine._loaded is True


def test_manager_model_names(real_manager):
    """Manager: 能列出所有配置的模型名称。"""
    names = real_manager.get_model_names()
    assert MODEL_NAME in names
    assert len(names) >= 1


# ===========================================================================
# 3. Chat 路由层 — 通过路由函数直接调用
# ===========================================================================


def test_chat_basic(real_manager):
    """Chat: 基础问答 — 返回完整 OpenAI 兼容结构。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="1+1等于多少？回答数字即可。")],
        temperature=0.0, max_tokens=256, stream=False,
    )
    result = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))

    assert isinstance(result, ChatCompletionResponse)
    assert result.model == MODEL_NAME
    assert len(result.choices) == 1
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
    answer = _strip_thinking(result.choices[0].message.content)
    assert "2" in answer


def test_chat_programming(real_manager):
    """Chat: 编程问题 — 返回有意义的编程知识。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="Python 中 list 和 tuple 有什么区别？简要回答。")],
        temperature=0.0, max_tokens=256, stream=False,
    )
    result = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content).lower()

    assert content.strip()
    assert any(kw in content for kw in ("可变", "不可变", "list", "tuple", "列表", "元组", "修改"))


def test_chat_with_system_prompt(real_manager):
    """Chat: 自定义 system prompt 生效。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            ChatMessage(role="system", content="你是一个只输出 JSON 的助手。"),
            ChatMessage(role="user", content="说 hello"),
        ],
        temperature=0.0, max_tokens=256, stream=False,
    )
    result = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    assert _strip_thinking(result.choices[0].message.content).strip()


def test_chat_multi_turn(real_manager):
    """Chat: 多轮对话上下文 — 模型能回忆前文信息。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            ChatMessage(role="user", content="请记住这个数字：42"),
            ChatMessage(role="assistant", content="好的，我记住了：42。"),
            ChatMessage(role="user", content="我刚才让你记住的数字是多少？"),
        ],
        temperature=0.0, max_tokens=256, stream=False,
    )
    result = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    assert "42" in _strip_thinking(result.choices[0].message.content)


def test_chat_stop_sequence(real_manager):
    """Chat: stop 参数截断生成。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="从1开始数数，数到10")],
        temperature=0.0, max_tokens=256, stream=False,
        stop="5",
    )
    result = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    answer = _strip_thinking(result.choices[0].message.content)
    # stop 截断后不应包含 "5" 之后的数字
    assert "6" not in answer


def test_chat_seed_reproducibility(real_manager):
    """Chat: 相同 seed 产生相同输出。"""
    messages = [ChatMessage(role="user", content="用一句话描述天空的颜色")]
    common = dict(model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=256, stream=False, seed=42)

    r1 = _run_async(create_chat_completion(ChatCompletionRequest(**common), response=_make_mock_response(), manager=real_manager))
    r2 = _run_async(create_chat_completion(ChatCompletionRequest(**common), response=_make_mock_response(), manager=real_manager))

    c1 = _strip_thinking(r1.choices[0].message.content)
    c2 = _strip_thinking(r2.choices[0].message.content)
    assert c1 == c2, "相同 seed + temperature=0 应产生确定性输出"


def test_chat_streaming(real_manager):
    """Chat: 流式 SSE — 返回 StreamingResponse 且能收集完整内容。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="用一句话解释什么是 API")],
        temperature=0.0, max_tokens=256, stream=True,
    )
    streaming_resp = _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    full_text, usage_info = _collect_stream(streaming_resp)
    answer = _strip_thinking(full_text)

    assert answer.strip(), "流式收集的内容不应为空"
    assert usage_info is not None, "流式最终 chunk 应包含 usage 信息"
    assert usage_info["prompt_tokens"] > 0
    assert usage_info["completion_tokens"] > 0


def test_chat_cache_hit(real_manager):
    """Chat: 缓存 — 相同请求第二次应命中缓存 (X-Cache: HIT)。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="缓存测试：1+1=?")],
        temperature=0.0, max_tokens=256, stream=False,
    )
    # 第一次请求: MISS
    resp1 = _make_mock_response()
    _run_async(create_chat_completion(req, response=resp1, manager=real_manager))
    assert resp1.headers.get("X-Cache") == "MISS"

    # 第二次请求: HIT
    resp2 = _make_mock_response()
    _run_async(create_chat_completion(req, response=resp2, manager=real_manager))
    assert resp2.headers.get("X-Cache") == "HIT"


def test_chat_unknown_model_404(real_manager):
    """Chat: 请求不存在的模型应返回 404。"""
    from fastapi import HTTPException
    req = ChatCompletionRequest(
        model="nonexistent-model-xyz",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=16,
    )
    with pytest.raises(HTTPException) as exc_info:
        _run_async(create_chat_completion(req, response=_make_mock_response(), manager=real_manager))
    assert exc_info.value.status_code == 404


# ===========================================================================
# 4. Completions 路由层
# ===========================================================================


def test_completion_basic(real_manager):
    """Completions: 基础文本补全。"""
    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        max_tokens=256, temperature=0.0, stream=False,
    )
    result = _run_async(create_completion(req, response=_make_mock_response(), manager=real_manager))

    assert isinstance(result, CompletionResponse)
    assert result.model == MODEL_NAME
    assert result.choices[0].text.strip()
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


def test_completion_fim(real_manager):
    """Completions: FIM (Fill-In-The-Middle) 代码补全。"""
    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="def greet(name):\n    ",
        suffix="\n    return greeting",
        max_tokens=256, temperature=0.0, stream=False,
    )
    result = _run_async(create_completion(req, response=_make_mock_response(), manager=real_manager))

    assert result.choices[0].text.strip()
    assert result.usage.completion_tokens > 0


def test_completion_streaming(real_manager):
    """Completions: 流式 SSE 补全。"""
    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="Python 之禅的第一行是:",
        max_tokens=256, temperature=0.0, stream=True,
    )
    streaming_resp = _run_async(create_completion(req, response=_make_mock_response(), manager=real_manager))
    full_text, _ = _collect_stream(streaming_resp)

    assert full_text.strip()


# ===========================================================================
# 5. Code 路由层 — 编程助手全链路
# ===========================================================================


def test_code_explain(real_manager):
    """Code: 代码解释 — 输出包含关键逻辑说明。"""
    req = CodeExplainRequest(
        model=MODEL_NAME,
        code="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        language="python", temperature=0.0, max_tokens=256,
    )
    result = _run_async(explain_code(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content).lower()

    assert content.strip()
    assert any(kw in content for kw in ("递归", "阶乘", "factorial", "recursive", "基线", "终止"))


def test_code_refactor(real_manager):
    """Code: 代码重构 — 输出包含重构后的代码。"""
    req = CodeRefactorRequest(
        model=MODEL_NAME,
        code="x = [1,2,3,4,5]\nresult = []\nfor i in x:\n    result.append(i*2)",
        language="python", instruction="使用列表推导式简化",
        temperature=0.0, max_tokens=256,
    )
    result = _run_async(refactor_code(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content for kw in ("for", "map", "推导", "lambda", "[", "*2", "* 2"))


def test_code_generate_tests(real_manager):
    """Code: 单测生成 — 输出包含可识别的测试框架代码。"""
    req = CodeTestGenerateRequest(
        model=MODEL_NAME,
        code="def add(a, b):\n    return a + b",
        language="python", framework="pytest",
        temperature=0.0, max_tokens=512,
    )
    result = _run_async(generate_tests(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content.lower() for kw in ("def test_", "assert", "import pytest", "add("))


def test_code_fix_bugs(real_manager):
    """Code: Bug 修复 — 识别除零问题并给出修复。"""
    req = CodeFixBugsRequest(
        model=MODEL_NAME,
        code="def divide(a, b):\n    return a / b",
        language="python", error_message="ZeroDivisionError: division by zero",
        temperature=0.0, max_tokens=256,
    )
    result = _run_async(fix_bugs(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content for kw in ("b == 0", "b != 0", "ZeroDivision", "除", "zero", "if b", "异常", "except", "检查"))


def test_code_edit(real_manager):
    """Code: 行内编辑 — 按指令修改代码。"""
    req = CodeEditRequest(
        model=MODEL_NAME,
        code="def get_name():\n    name = 'alice'\n    return name",
        language="python", instruction="将函数改为接受一个参数 name 并返回其大写形式",
        temperature=0.0, max_tokens=256,
    )
    result = _run_async(edit_code(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content for kw in ("upper", "Upper", "UPPER", "def get_name", "def ", "大写"))


def test_code_review(real_manager):
    """Code: 代码审查 — 给出具体问题与改进建议。"""
    req = CodeReviewRequest(
        model=MODEL_NAME,
        code="def process(data):\n    result = []\n    for item in data:\n        result.append(item)\n    return result",
        language="python", temperature=0.0, max_tokens=256,
    )
    result = _run_async(review_code(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content for kw in ("建议", "可以", "优化", "简化", "列表推导", "直接", "append", "问题", "改进"))


def test_code_docstring(real_manager):
    """Code: 文档字符串生成 — 包含函数说明。"""
    req = CodeDocstringRequest(
        model=MODEL_NAME,
        code="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        language="python", temperature=0.0, max_tokens=256,
    )
    result = _run_async(generate_docstring(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    assert content.strip()
    assert any(kw in content for kw in ("二分", "查找", "binary", "search", "参数", "Args", "返回", "Returns", '"""', "'''"))


# ===========================================================================
# 6. 高级特性 — 长上下文 / thinking / 代码正确性
# ===========================================================================


def test_long_context_handling(real_engine):
    """高级: 长上下文 — 模型能处理 4K+ token 的输入并正确回答。"""
    # 构建长文本: 重复段落 + 尾部隐藏问题
    filler = "Python 是一种广泛使用的高级编程语言，强调代码的可读性和简洁的语法。\n"
    long_text = filler * 80  # ~4K tokens
    question = f"\n以上文本重复描述了多少次？请回答数字。\n"
    prompt = f"<|im_start|>user\n{long_text}{question}<|im_end|>\n<|im_start|>assistant\n"

    prompt_tokens = real_engine.count_tokens(prompt)
    assert prompt_tokens > 1000, f"测试前提: prompt 应超过 1000 tokens，实际 {prompt_tokens}"

    result = real_engine.generate(prompt, max_tokens=256, temperature=0.0, top_p=0.9)
    answer = _strip_thinking(result)

    assert answer.strip(), "长上下文输入后模型应产生非空回答"
    assert "80" in answer, f"模型应正确回答重复次数 80，实际回答: {answer[:200]}"


def test_code_correctness(real_engine):
    """高级: 代码正确性 — 模型生成的代码应能实际执行。"""
    prompt = (
        "<|im_start|>user\n写一个 Python 函数 is_palindrome(s)，判断字符串是否为回文。"
        "只输出代码，不要解释。<|im_end|>\n<|im_start|>assistant\n"
    )
    result = real_engine.generate(prompt, max_tokens=512, temperature=0.0, top_p=0.9)
    answer = _strip_thinking(result)

    # 提取代码块
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', answer, re.DOTALL)
    code = code_match.group(1).strip() if code_match else answer.strip()

    # 代码应能实际执行
    exec_globals = {}
    exec(code, exec_globals)
    assert "is_palindrome" in exec_globals, "生成的代码应定义 is_palindrome 函数"

    # 验证函数正确性
    fn = exec_globals["is_palindrome"]
    assert fn("racecar") is True
    assert fn("hello") is False
    assert fn("abba") is True
    assert fn("") is True


def test_code_generation_sort(real_manager):
    """高级: 生成测试代码 — 测试代码应包含测试关键字并引用目标函数。"""
    req = CodeTestGenerateRequest(
        model=MODEL_NAME,
        code="def multiply(a, b):\n    return a * b",
        language="python", framework="pytest",
        temperature=0.0, max_tokens=512,
    )
    result = _run_async(generate_tests(req, response=_make_mock_response(), manager=real_manager))
    content = _strip_thinking(result.choices[0].message.content)

    # 在完整内容中搜索测试关键字 (thinking 模型可能将代码散布在解释中)
    assert any(kw in content for kw in ("def test_", "assert", "import pytest", "pytest")), \
        f"生成内容应包含测试框架关键字，实际内容前 300 字: {content[:300]}"
    # 应引用目标函数
    assert "multiply" in content, "生成内容应引用 multiply 函数"


# ===========================================================================
# 7. Engine 统计与可用性
# ===========================================================================


def test_engine_token_count(real_engine):
    """统计: Token 计数在合理范围内。"""
    assert real_engine.count_tokens("Hello, world!") > 0
    assert real_engine.count_tokens("") == 0
    # 中文 token 计数
    zh_count = real_engine.count_tokens("这是一段中文测试文本")
    assert zh_count > 0


def test_engine_prompt_length_check(real_manager):
    """统计: 正常 prompt 不报错，超长 prompt 抛 HTTPException。"""
    from fastapi import HTTPException

    # 正常长度不应抛异常
    real_manager.check_prompt_length(MODEL_NAME, 1000)

    # 超长应抛 400
    with pytest.raises(HTTPException) as exc_info:
        real_manager.check_prompt_length(MODEL_NAME, 999999)
    assert exc_info.value.status_code == 400
    assert "Prompt too long" in str(exc_info.value.detail)


def test_engine_stats(real_engine):
    """统计: 推理后统计信息正确更新。"""
    stats = real_engine.get_stats()
    assert stats["loaded"] is True
    assert stats["total_requests"] >= 1
    assert stats["model_name"] == MODEL_NAME
    assert stats["engine_type"] in ("qwen4_exp", "mlx_lm", "mlx_vlm")


def test_engine_health(real_engine):
    """统计: 健康检查返回 True。"""
    assert real_engine.health_check() is True


def test_engine_memory_stats(real_engine):
    """统计: Metal 显存统计返回有效结构。"""
    mem = real_engine.get_memory_stats()
    assert isinstance(mem, dict)
    # Apple Silicon 上应包含 active_memory_mb
    if mem:
        assert "active_memory_mb" in mem
        assert mem["active_memory_mb"] > 0
