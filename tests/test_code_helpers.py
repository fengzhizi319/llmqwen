"""
编程助手专有快捷 API (/v1/code/*) 测试套件
"""

def test_code_explain_endpoint(client):
    payload = {
        "code": "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)",
        "language": "python",
    }
    response = client.post("/v1/code/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_code_refactor_endpoint(client):
    payload = {
        "code": "l = []\nfor i in range(10):\n    if i % 2 == 0:\n        l.append(i)",
        "instruction": "使用列表推导式重构",
        "language": "python",
    }
    response = client.post("/v1/code/refactor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_code_generate_tests_endpoint(client):
    payload = {
        "code": "def multiply(a, b): return a * b",
        "framework": "pytest",
        "language": "python",
    }
    response = client.post("/v1/code/generate-tests", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_code_fix_bugs_endpoint(client):
    payload = {
        "code": "print('hello' + 123)",
        "error_message": "TypeError: can only concatenate str (not \"int\") to str",
        "language": "python",
    }
    response = client.post("/v1/code/fix-bugs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_code_edit_endpoint(client):
    payload = {
        "code": "def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total",
        "instruction": "使用内置函数简化",
        "language": "python",
    }
    response = client.post("/v1/code/edit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_code_review_endpoint(client):
    payload = {
        "code": "def divide(a, b):\n    return a / b",
        "language": "python",
    }
    response = client.post("/v1/code/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_code_docstring_endpoint(client):
    payload = {
        "code": "def add(a, b):\n    return a + b",
        "language": "python",
    }
    response = client.post("/v1/code/docstring", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"][0]["message"]["content"]) > 0
