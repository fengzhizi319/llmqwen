"""
Code Completions & FIM API 测试套件
"""

def test_code_completions_basic(client):
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def add(a, b):\n   ",
        "max_tokens": 50,
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "usage" in data


def test_code_completions_fim_fill_middle(client):
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def calculate_total(a, b):\n   ",
        "suffix": "\n    return result",
        "max_tokens": 50,
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    text = data["choices"][0]["text"]
    assert len(text) > 0


def test_code_completions_stream(client):
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "class DatabaseConnection:",
        "stream": True,
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text


def test_code_completions_with_seed(client):
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def hello():",
        "seed": 123,
        "max_tokens": 30,
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"][0]["text"]) > 0
