"""
本地真实模型集成 UT。

运行这些测试会加载完整模型并执行实际推理，默认跳过：
    RUN_REAL_MODEL_TESTS=1 pytest -m real_model -q

测试覆盖两个维度：
  1. Engine 层：直接调用 MLXModelEngine.generate / stream_generate
  2. 本地函数调用层：通过 ModelManager / 路由处理函数直接调用（不经过 HTTP），
     验证 prompt 构建、Chat/Completions/Code 全链路业务逻辑的正确性。

模型名称与路径均从 config.yaml 动态读取，添加新模型时无需修改本文件。
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import load_config
from engine.mlx_engine import MLXModelEngine
from engine import ModelManager
from schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
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
MODEL_PATH = PROJECT_ROOT / _real_config.models[MODEL_NAME].path.removeprefix("./")

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_engine():
    """在主线程中创建并实际加载模型，避免 MLX 与测试线程池冲突。"""
    config = load_config(str(CONFIG_PATH))
    assert config.default_model == MODEL_NAME
    assert config.use_mock is False

    spec = config.models[MODEL_NAME]
    engine = MLXModelEngine(
        model_name=MODEL_NAME,
        model_path=spec.path,
        engine_type=spec.engine_type,
        metal_cache_limit_mb=config.performance.metal_cache_limit_mb,
        clear_cache_after_generation=config.performance.clear_cache_after_generation,
        kv_bits=config.performance.kv_bits,
        kv_group_size=config.performance.kv_group_size,
        prefill_step_size=config.performance.prefill_step_size,
        enable_prompt_cache=config.performance.enable_prompt_cache,
    )
    engine.load_model()
    assert engine._loaded is True
    # mlx_vlm 有 processor，mlx_lm 有 tokenizer
    assert engine.processor is not None or engine.tokenizer is not None
    yield engine


@pytest.fixture(scope="module")
def real_manager():
    """基于真实配置创建 ModelManager，通过本地函数调用走完整业务链路。"""
    config = load_config(str(CONFIG_PATH))
    manager = ModelManager(config)
    # 预热：获取引擎并加载模型
    engine = manager.get_engine(MODEL_NAME)
    if hasattr(engine, "load_model"):
        engine.load_model()
    return manager


def _run_async(coro):
    """在同步测试中运行异步协程。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_response():
    """创建一个轻量 Mock Response 对象，供路由函数设置 header。"""
    resp = MagicMock()
    resp.headers = {}
    return resp


# ===========================================================================
# 1. Engine 层 — 直接调用 MLXModelEngine
# ===========================================================================


def test_real_model_chat(real_engine):
    """验证真实模型能够完成普通对话请求。"""
    prompt = (
        "<|im_start|>user\n请用一句话说明 Python 是什么。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    result = real_engine.generate(prompt, max_tokens=32, temperature=0.0, top_p=0.9)

    assert result.strip()
    assert real_engine.count_tokens(result) > 0


def test_real_model_programming(real_engine):
    """验证真实模型能够处理编程解释场景。"""
    prompt = (
        "<|im_start|>user\n请详细解释以下 Python 代码的作用：\n\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    result = real_engine.generate(prompt, max_tokens=64, temperature=0.0, top_p=0.9)

    assert result.strip()
    assert any(keyword in result.lower() for keyword in ("add", "return", "函数"))


def test_real_model_chat_stream(real_engine):
    """验证真实模型能够持续产生 Chat 流式文本。"""
    prompt = (
        "<|im_start|>user\n回答：1 加 1 等于多少？<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    chunks = list(
        real_engine.stream_generate(
            prompt, max_tokens=16, temperature=0.0, top_p=0.9
        )
    )
    result = "".join(chunks)

    assert chunks
    assert result.strip()


# ===========================================================================
# 2. ModelManager 层 — Prompt 构建与引擎获取
# ===========================================================================


def test_manager_build_chat_prompt(real_manager):
    """验证 ModelManager 构建的 Chat Prompt 符合 ChatML 格式。"""
    messages = [
        {"role": "user", "content": "你好"},
    ]
    prompt = real_manager.build_chat_prompt(messages)

    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "你好" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_manager_build_chat_prompt_with_system(real_manager):
    """验证带有 system 消息的 Prompt 构建包含防伪造约束。"""
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "写一段代码"},
    ]
    prompt = real_manager.build_chat_prompt(messages)

    assert "<|im_start|>system" in prompt
    assert "你是一个助手" in prompt
    assert "【核心约束】" in prompt
    assert "<|im_start|>user" in prompt


def test_manager_build_fim_prompt(real_manager):
    """验证 FIM (Fill-In-The-Middle) Prompt 构建正确。"""
    prefix = "def add(a, b):\n"
    suffix = "\n    return result"
    prompt = real_manager.build_fim_prompt(prefix, suffix)

    assert "<|fim_prefix|>" in prompt
    assert "<|fim_suffix|>" in prompt
    assert "<|fim_middle|>" in prompt
    assert prefix in prompt
    assert suffix in prompt


def test_manager_build_fim_prompt_no_suffix(real_manager):
    """验证无 suffix 时 FIM Prompt 直接返回 prefix。"""
    prefix = "print('hello')"
    prompt = real_manager.build_fim_prompt(prefix, None)
    assert prompt == prefix


def test_manager_get_engine_returns_real_engine(real_manager):
    """验证 ModelManager 返回的是真实 MLX 引擎而非 Mock。"""
    engine = real_manager.get_engine(MODEL_NAME)
    assert isinstance(engine, MLXModelEngine)
    assert engine._loaded is True


def test_manager_model_names(real_manager):
    """验证 ModelManager 能列出所有配置的模型名称。"""
    names = real_manager.get_model_names()
    assert MODEL_NAME in names
    assert len(names) >= 1


# ===========================================================================
# 3. Chat 路由函数 — 本地直接调用（不经过 HTTP）
# ===========================================================================


def test_local_chat_completion_basic(real_manager):
    """验证通过路由函数直接调用 Chat Completions 能返回正确结构。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="1+1等于多少？回答数字即可。")],
        temperature=0.0,
        max_tokens=16,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_chat_completion(req, response=resp, manager=real_manager)
    )

    assert result.model == MODEL_NAME
    assert len(result.choices) == 1
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].message.content.strip()
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


def test_local_chat_completion_programming(real_manager):
    """验证 Chat Completions 能正确处理编程问题并返回有意义内容。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="Python 中 list 和 tuple 有什么区别？简要回答。")],
        temperature=0.0,
        max_tokens=128,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_chat_completion(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content.lower()
    assert result.choices[0].message.content.strip()
    # 模型应该提到可变/不可变 或 列表/元组 相关关键词
    assert any(kw in content for kw in ("可变", "不可变", "list", "tuple", "列表", "元组", "修改"))


def test_local_chat_completion_with_system_prompt(real_manager):
    """验证带有自定义 system prompt 的 Chat 调用。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            ChatMessage(role="system", content="你是一个只输出 JSON 的助手。"),
            ChatMessage(role="user", content="说 hello"),
        ],
        temperature=0.0,
        max_tokens=32,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_chat_completion(req, response=resp, manager=real_manager)
    )

    assert result.choices[0].message.content.strip()


def test_local_chat_completion_multi_turn(real_manager):
    """验证多轮对话上下文的正确性。"""
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            ChatMessage(role="user", content="请记住这个数字：42"),
            ChatMessage(role="assistant", content="好的，我记住了：42。"),
            ChatMessage(role="user", content="我刚才让你记住的数字是多少？"),
        ],
        temperature=0.0,
        max_tokens=16,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_chat_completion(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert "42" in content


# ===========================================================================
# 4. Completions 路由函数 — 本地直接调用
# ===========================================================================


def test_local_completion_basic(real_manager):
    """验证基础文本补全接口。"""
    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        max_tokens=32,
        temperature=0.0,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_completion(req, response=resp, manager=real_manager)
    )

    assert result.model == MODEL_NAME
    assert len(result.choices) == 1
    assert result.choices[0].text.strip()
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


def test_local_completion_fim(real_manager):
    """验证 FIM (Fill-In-The-Middle) 代码补全接口。"""
    req = CompletionRequest(
        model=MODEL_NAME,
        prompt="def greet(name):\n    ",
        suffix="\n    return greeting",
        max_tokens=32,
        temperature=0.0,
        stream=False,
    )
    resp = _make_mock_response()
    result = _run_async(
        create_completion(req, response=resp, manager=real_manager)
    )

    assert result.choices[0].text.strip()
    assert result.usage.completion_tokens > 0


# ===========================================================================
# 5. Code 路由函数 — 本地直接调用
# ===========================================================================


def test_local_code_explain(real_manager):
    """验证代码解释接口能返回有意义的解释。"""
    req = CodeExplainRequest(
        model=MODEL_NAME,
        code="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        language="python",
        temperature=0.0,
        max_tokens=128,
    )
    resp = _make_mock_response()
    result = _run_async(
        explain_code(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content.lower()
    assert content.strip()
    assert any(kw in content for kw in ("递归", "阶乘", "factorial", "recursive", "recursion", "基线"))


def test_local_code_refactor(real_manager):
    """验证代码重构接口能返回重构后的代码。"""
    req = CodeRefactorRequest(
        model=MODEL_NAME,
        code="x = [1,2,3,4,5]\nresult = []\nfor i in x:\n    result.append(i*2)",
        language="python",
        instruction="使用列表推导式简化",
        temperature=0.0,
        max_tokens=128,
    )
    resp = _make_mock_response()
    result = _run_async(
        refactor_code(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    # 重构后应该包含列表推导式或 map 等简洁写法
    assert any(kw in content for kw in ("for", "map", "推导", "lambda", "[", "*2", "* 2"))


def test_local_code_generate_tests(real_manager):
    """验证单测生成接口能产生测试代码。"""
    req = CodeTestGenerateRequest(
        model=MODEL_NAME,
        code="def add(a, b):\n    return a + b",
        language="python",
        framework="pytest",
        temperature=0.0,
        max_tokens=256,
    )
    resp = _make_mock_response()
    result = _run_async(
        generate_tests(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    # 生成的测试代码应包含 pytest 相关关键字或 assert
    assert any(kw in content.lower() for kw in ("def test_", "assert", "import pytest", "add("))


def test_local_code_fix_bugs(real_manager):
    """验证 Bug 修复接口能识别并修复代码问题。"""
    req = CodeFixBugsRequest(
        model=MODEL_NAME,
        code="def divide(a, b):\n    return a / b",
        language="python",
        error_message="ZeroDivisionError: division by zero",
        temperature=0.0,
        max_tokens=128,
    )
    resp = _make_mock_response()
    result = _run_async(
        fix_bugs(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    # 修复建议应包含除零保护相关内容
    assert any(kw in content for kw in ("b == 0", "b != 0", "ZeroDivision", "除", "zero", "if b", "异常", "except", "检查"))


def test_local_code_edit(real_manager):
    """验证行内代码编辑接口。"""
    req = CodeEditRequest(
        model=MODEL_NAME,
        code="def get_name():\n    name = 'alice'\n    return name",
        language="python",
        instruction="将函数改为接受一个参数 name 并返回其大写形式",
        temperature=0.0,
        max_tokens=128,
    )
    resp = _make_mock_response()
    result = _run_async(
        edit_code(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    assert any(kw in content for kw in ("upper", "Upper", "UPPER", "def get_name", "def ", "大写"))


def test_local_code_review(real_manager):
    """验证代码审查接口能给出审查意见。"""
    req = CodeReviewRequest(
        model=MODEL_NAME,
        code="def process(data):\n    result = []\n    for item in data:\n        result.append(item)\n    return result",
        language="python",
        temperature=0.0,
        max_tokens=128,
    )
    resp = _make_mock_response()
    result = _run_async(
        review_code(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    # 审查意见应提到代码相关问题或建议
    assert any(kw in content for kw in ("建议", "可以", "优化", "简化", "列表推导", "直接", "append", "问题", "改进"))


def test_local_code_docstring(real_manager):
    """验证文档字符串生成接口。"""
    req = CodeDocstringRequest(
        model=MODEL_NAME,
        code="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        language="python",
        temperature=0.0,
        max_tokens=256,
    )
    resp = _make_mock_response()
    result = _run_async(
        generate_docstring(req, response=resp, manager=real_manager)
    )

    content = result.choices[0].message.content
    assert content.strip()
    # 文档字符串应包含函数说明相关关键词
    assert any(kw in content for kw in ("二分", "查找", "binary", "search", "参数", "Args", "返回", "Returns", '"""', "'''"))


# ===========================================================================
# 6. Token 计数与上下文校验
# ===========================================================================


def test_real_engine_token_count(real_engine):
    """验证真实 Tokenizer 的 token 计数准确性。"""
    text = "Hello, world!"
    count = real_engine.count_tokens(text)
    assert count > 0
    # 对已知文本，token 数应在合理范围内
    assert count <= len(text)  # token 数不应超过字符数（对英文而言）

    empty_count = real_engine.count_tokens("")
    assert empty_count == 0


def test_manager_check_prompt_length_ok(real_manager):
    """验证正常长度的 prompt 不会触发异常。"""
    # 不应该抛出异常
    real_manager.check_prompt_length(MODEL_NAME, 1000)


def test_manager_check_prompt_length_too_long(real_manager):
    """验证超长 prompt 会触发 HTTPException。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        real_manager.check_prompt_length(MODEL_NAME, 999999)
    assert exc_info.value.status_code == 400
    assert "Prompt too long" in str(exc_info.value.detail)


# ===========================================================================
# 7. Engine 统计与可用性
# ===========================================================================


def test_real_engine_stats(real_engine):
    """验证引擎统计信息在推理后正确更新。"""
    prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    real_engine.generate(prompt, max_tokens=8, temperature=0.0, top_p=0.9)

    stats = real_engine.get_stats()
    assert stats["loaded"] is True
    assert stats["total_requests"] >= 1
    assert stats["total_generation_tokens"] >= 0
    assert stats["model_name"] == MODEL_NAME


def test_real_engine_health(real_engine):
    """验证引擎健康检查。"""
    assert real_engine.health_check() is True


def test_real_engine_memory_stats(real_engine):
    """验证 Metal 显存统计（在 Apple Silicon 上应返回有效数据）。"""
    mem = real_engine.get_memory_stats()
    # 在 Apple Silicon 上应返回 active_memory_mb 等字段
    # 非 Apple Silicon 可能返回空字典，这里只验证不报错
    assert isinstance(mem, dict)

