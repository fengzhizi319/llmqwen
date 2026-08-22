"""
Pytest 共享 Context 与 Fixtures 配置
"""

import pytest
from fastapi.testclient import TestClient
from config import AppConfig, ServerConfig, ModelSpec
from app import create_app
from engine import ModelManager


@pytest.fixture
def mock_config():
    """提供全 Mock 模式的配置 (默认 256K 上下文)"""
    return AppConfig(
        server=ServerConfig(host="127.0.0.1", port=8000),
        default_model="qwen3.8-27b",
        use_mock=True,
        models={
            "qwen3.8-27b": ModelSpec(path="Qwen/Qwen3.8-27B", description="Mock Qwen 27B (256K)", context_length=262144),
            "qwen3.8-27b-8bit": ModelSpec(path="Qwen/Qwen3.8-27B-8bit", description="Mock 8bit (256K)", context_length=262144),
        },
    )


@pytest.fixture
def client(mock_config, monkeypatch):
    """FastAPI TestClient Fixture"""
    # 模拟 load_config 返回 mock_config
    monkeypatch.setattr("app.load_config", lambda path="config.yaml": mock_config)
    app = create_app()
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def auth_client(mock_config, monkeypatch):
    """带 API Key 鉴权的 TestClient Fixture"""
    mock_config.server.api_key = "test-secret-key"
    monkeypatch.setattr("app.load_config", lambda path="config.yaml": mock_config)
    app = create_app()
    with TestClient(app) as tc:
        yield tc
