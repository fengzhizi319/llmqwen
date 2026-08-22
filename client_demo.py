"""
AI Code Service - API 客户端测试脚本
在本地启动 `start.sh` 或 `python app.py` 后运行此脚本验证全套接口
"""

import time
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0  # 大模型生成超时时间


def test_health():
    print("\n--- [1] 检查服务健康状态 ---")
    resp = httpx.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    print(f"Status: {resp.status_code}, Response: {resp.json()}")


def test_models():
    print("\n--- [2] 获取模型列表 ---")
    resp = httpx.get(f"{BASE_URL}/v1/models", timeout=TIMEOUT)
    print(f"Models: {resp.json()}")


def test_chat_completion():
    print("\n--- [3] 对话补全 (/v1/chat/completions) ---")
    print("⏳ 正在请求 27B 大模型生成，请稍候...")
    t0 = time.time()
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "user", "content": "请用一句话解释 Python 单例模式"}
        ],
        "temperature": 0.5,
        "max_tokens": 100,
    }
    resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    data = resp.json()
    print(f"✅ 生成完成 (耗时 {round(time.time() - t0, 2)}s):")
    print(data["choices"][0]["message"]["content"])
    print("Token Usage:", data.get("usage"))


def test_chat_stream():
    print("\n--- [4] 流式对话补全 (Stream SSE) ---")
    print("⏳ 实时接收流式 Token:")
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "user", "content": "输出数字 1 到 5"}
        ],
        "stream": True,
        "max_tokens": 50,
    }
    with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as resp:
        for line in resp.iter_lines():
            if line:
                print(line)


def test_code_fim_autocomplete():
    print("\n--- [5] FIM 代码补全 (/v1/completions) ---")
    print("⏳ 正在请求 FIM 补全...")
    t0 = time.time()
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def add(a: int, b: int) -> int:\n   ",
        "suffix": "\n    return result",
        "max_tokens": 50,
    }
    resp = httpx.post(f"{BASE_URL}/v1/completions", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ 补全结果 (耗时 {round(time.time() - t0, 2)}s):", resp.json()["choices"][0]["text"])


def test_specialized_code_tools():
    print("\n--- [6] 专有代码重构工具 (/v1/code/refactor) ---")
    print("⏳ 正在请求代码重构...")
    t0 = time.time()
    payload = {
        "code": "nums = [1, 2, 3]\nev = []\nfor x in nums:\n    ev.append(x * 2)",
        "instruction": "简化为列表推导式",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/refactor", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ 重构结果 (耗时 {round(time.time() - t0, 2)}s):")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_inline_edit():
    print("\n--- [7] 行内代码编辑 (/v1/code/edit) ---")
    print("⏳ 正在请求代码编辑...")
    t0 = time.time()
    payload = {
        "code": "def sum_list(nums):\n    return sum(nums)",
        "instruction": "添加类型注解与文档",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/edit", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ 编辑结果 (耗时 {round(time.time() - t0, 2)}s):")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_review():
    print("\n--- [8] 代码审查 (/v1/code/review) ---")
    print("⏳ 正在请求代码审查...")
    t0 = time.time()
    payload = {
        "code": "def divide(a, b):\n    return a / b",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/review", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ 审查结果 (耗时 {round(time.time() - t0, 2)}s):")
    print(resp.json()["choices"][0]["message"]["content"])


def test_code_docstring():
    print("\n--- [9] 文档字符串生成 (/v1/code/docstring) ---")
    print("⏳ 正在请求文档字符串生成...")
    t0 = time.time()
    payload = {
        "code": "def multiply(a: float, b: float) -> float:\n    return a * b",
        "language": "python",
    }
    resp = httpx.post(f"{BASE_URL}/v1/code/docstring", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ Docstring 结果 (耗时 {round(time.time() - t0, 2)}s):")
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
        print("\n🎉 全部 9 项 API 客户端功能验证顺利完成！")
    except httpx.ConnectError:
        print("\n❌ 连接失败：请确保 AI Code Service 服务已启动 (运行 ./start.sh 或 python app.py)")
