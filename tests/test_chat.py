"""
Chat Completions API 测试套件
"""

import pytest
from fastapi.testclient import TestClient
from app import create_app
from config import AppConfig, ServerConfig, ModelSpec
from tests.conftest import TEST_MODEL_NAME


def test_chat_completions_non_stream(client):
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [
            {"role": "user", "content": "请写一个 Python 快速排序函数"}
        ],
        "temperature": 0.5,
        "max_tokens": 100,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["model"] == TEST_MODEL_NAME
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completions_stream(client):
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Hello AI Code Service"}
        ],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    assert len(lines) > 0
    
    # 验证第一流数据格式与结束标记
    has_data_chunk = False
    has_done = False
    for chunk in lines:
        if chunk.startswith("data: "):
            content = chunk[6:]
            if content == "[DONE]":
                has_done = True
            else:
                has_data_chunk = True
                assert "chat.completion.chunk" in content

    assert has_data_chunk
    assert has_done


def test_chat_completions_invalid_model(client):
    payload = {
        "model": "non-existent-model-xxx",
        "messages": [{"role": "user", "content": "test"}],
    }
    # 在 mock_config 下如果 use_mock 为 True 会 fallback 到 mock 或 404
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code in (200, 404)


@pytest.fixture
def small_context_client(monkeypatch):
    """上下文长度很小的配置，用于测试 prompt 超限"""
    cfg = AppConfig(
        server=ServerConfig(host="127.0.0.1", port=1235),
        default_model=TEST_MODEL_NAME,
        use_mock=True,
        models={
            TEST_MODEL_NAME: ModelSpec(
                path="Qwen/Qwen3.8-27B",
                description="Mock small ctx",
                context_length=10,
            ),
        },
    )
    monkeypatch.setattr("app.load_config", lambda path="config.yaml": cfg)
    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_chat_completions_with_seed(client):
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "seed": 42,
        "max_tokens": 20,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_chat_prompt_too_long(small_context_client):
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "a" * 100}],
        "max_tokens": 10,
    }
    response = small_context_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "too long" in detail or "exceeds" in detail or "context" in detail


def test_responses_endpoint_compatibility(client):
    """测试 OpenAI Responses API (/v1/responses) 兼容端点"""
    payload = {
        "model": TEST_MODEL_NAME,
        "input": "请解释 Python 列表推导式",
        "instructions": "你是一个精通 Python 的助手",
        "max_output_tokens": 50,
    }
    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"]) == 1
    assert len(data["choices"][0]["message"]["content"]) > 0
