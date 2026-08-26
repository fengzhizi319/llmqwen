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
