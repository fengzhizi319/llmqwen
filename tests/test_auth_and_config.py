"""
配置与 API Key 鉴权测试套件
"""

from tests.conftest import TEST_MODEL_NAME


def test_unauthorized_access(auth_client):
    # 未带 Bearer Token 访问受保护的 chat completion
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
    }
    res = auth_client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 401
    assert "Authorization header" in res.json()["detail"]


def test_invalid_bearer_token(auth_client):
    headers = {"Authorization": "Bearer wrong-key"}
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
    }
    res = auth_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 401
    assert "Invalid API Key" in res.json()["detail"]


def test_valid_bearer_token(auth_client):
    headers = {"Authorization": "Bearer test-secret-key"}
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
    }
    res = auth_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 200
    assert res.json()["object"] == "chat.completion"


def test_public_path_without_auth(auth_client):
    # 公开路径在配置了 API Key 时依然可以直接访问
    res = auth_client.get("/health")
    assert res.status_code == 200


def test_environment_overrides_yaml(tmp_path, monkeypatch):
    from config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "server:\n  api_key: yaml-key\ndefault_model: yaml-model\nuse_mock: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_KEY", "env-key")
    monkeypatch.setenv("DEFAULT_MODEL", "env-model")
    monkeypatch.setenv("USE_MOCK", "true")
    config = load_config(str(config_path))
    assert config.server.api_key == "env-key"
    assert config.default_model == "env-model"
    assert config.use_mock is True


def test_default_config_loads_outside_project(tmp_path, monkeypatch):
    from pathlib import Path
    import config

    monkeypatch.chdir(tmp_path)
    assert config.load_config().models == config.load_config(
        str(Path(config.__file__).parent / "config.yaml")
    ).models


def test_zero_concurrency_is_rejected():
    import pytest
    from pydantic import ValidationError
    from config import PerformanceConfig

    with pytest.raises(ValidationError):
        PerformanceConfig(max_concurrency=0)
