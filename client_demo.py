"""
AI Code Service - API 客户端测试脚本
在本地启动 `start.sh` 或 `python app.py` 后运行此脚本验证全套接口
"""

import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n--- [1] 检查服务健康状态 ---")
    resp = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}, Response: {resp.json()}")


def test_models():
    print("\n--- [2] 获取模型列表 ---")
    resp = httpx.get(f"{BASE_URL}/v1/models")
    print(f"Models: {resp.json()}")


def test_chat_completion():
    print("\n--- [3] 对话补全 (/v1/chat/completions) ---")
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "user", "content": "请写一个 Python 单例模式 (Singleton) 示例"}
        ],
        "temperature": 0.5,
    }
    resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30.0)
    data = resp.json()
    print("Response Content:")
    print(data["choices"][0]["message"]["content"])
    print("Token Usage:", data.get("usage"))


def test_chat_stream():
    print("\n--- [4] 流式对话补全 (Stream SSE) ---")
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "user", "content": "用 30 个字解释 Python 中的 async/await"}
        ],
        "stream": True,
    }
    with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30.0) as resp:
        for line in resp.iter_lines():
            if line:
                print(line)


def test_code_fim_autocomplete():
    print("\n--- [5] FIM 代码补全 (/v1/completions) ---")
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def fetch_user_data(user_id: int):\n   ",
        "suffix": "\n    return user_data",
        "max_tokens": 100,
    }
    resp = httpx.post(f"{BASE_URL}/v1/completions", json=payload, timeout=30.0)
    print("Completion Result:", resp.json()["choices"][0]["text"])


def test_specialized_code_tools():
    print("\n--- [6] 专有代码重构工具 (/v1/code/refactor) ---")
    payload = {
        "code": "nums = [1, 2, 3, 4, 5]\nevens = []\nfor x in nums:\n    if x % 2 == 0:\n        evens.append(x)",
        "instruction": "简化为列表推导式",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/refactor", json=payload, timeout=30.0)
    print("Refactor Result:")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_inline_edit():
    print("\n--- [7] 行内代码编辑 (/v1/code/edit) ---")
    payload = {
        "code": "def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total",
        "instruction": "使用内置函数简化实现",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/edit", json=payload, timeout=30.0)
    print("Edit Result:")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_review():
    print("\n--- [8] 代码审查 (/v1/code/review) ---")
    payload = {
        "code": "def divide(a, b):\n    return a / b",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/review", json=payload, timeout=30.0)
    print("Review Result:")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_docstring():
    print("\n--- [9] 文档字符串生成 (/v1/code/docstring) ---")
    payload = {
        "code": "def add(a, b):\n    return a + b",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/docstring", json=payload, timeout=30.0)
    print("Docstring Result:")
    print(resp.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    try:
        test_health()
        test_models()
        test_chat_completion()
        test_chat_stream()
        test_code_fim_autocomplete()
        test_specialized_code_tools()
        test_code_inline_edit()
        test_code_review()
        test_code_docstring()
        print("\n✅ 所有 API 客户端测试完成！")
    except httpx.ConnectError:
        print("\n❌ 连接失败：请确保 AI Code Service 服务已启动 (运行 ./start.sh 或 python app.py)")
