"""
模型管理、健康检查与元数据 API 测试套件
"""

from tests.conftest import TEST_MODEL_NAME


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AI Code Service"
    assert "endpoints" in data


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "available_models" in data

    # 验证 /v1/health
    resp2 = client.get("/v1/health")
    assert resp2.status_code == 200


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "configured_models_count" in data


def test_models_list_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["object"] == "model"


def test_get_single_model_endpoint(client):
    response = client.get(f"/v1/models/{TEST_MODEL_NAME}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == TEST_MODEL_NAME
    assert data["object"] == "model"
