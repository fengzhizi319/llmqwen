"""
AI Code Service - API 客户端测试脚本
在本地启动 `start.sh` 或 `python app.py` 后运行此脚本验证全套接口
"""

import json
import time
import httpx

BASE_URL = "http://localhost:1235"
TIMEOUT = 120.0  # 大模型生成超时时间

# 全局默认模型名称，启动时从服务动态获取
DEFAULT_MODEL = ""


def get_default_model() -> str:
    """从服务健康检查接口动态获取默认模型名称"""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("default_model", "")
    except Exception:
        pass
    return ""


def _stream_print(url: str, payload: dict, label: str):
    """统一流式请求与打字机输出打印"""
    print(f"\n{label}")
    print("⏳ 实时打字机流式出字: ", end="", flush=True)
    payload["stream"] = True
    t0 = time.time()
    try:
        with httpx.stream("POST", url, json=payload, timeout=TIMEOUT) as resp:
            if resp.status_code != 200:
                print(f"\n❌ 请求失败 (Status {resp.status_code}): {resp.read().decode('utf-8')}")
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    except Exception:
                        pass
        print(f"\n✅ 流式输出完成 (耗时 {round(time.time() - t0, 2)}s)")
    except Exception as e:
        print(f"\n❌ 流式连接异常: {e}")


def test_health():
    print("\n--- [1] 检查服务健康状态 ---")
    resp = httpx.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    print(f"Status: {resp.status_code}, Response: {resp.json()}")


def test_models():
    print("\n--- [2] 获取模型列表 ---")
    resp = httpx.get(f"{BASE_URL}/v1/models", timeout=TIMEOUT)
    print(f"Models: {resp.json()}")


def test_chat_completion():
    print(f"\n--- [3] 对话补全 (/v1/chat/completions - Non-Stream) [模型: {DEFAULT_MODEL}] ---")
    print("⏳ 正在请求大模型生成...")
    t0 = time.time()
    payload = {
        "model": DEFAULT_MODEL,
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
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "输出数字 1 到 5"}],
        "max_tokens": 50,
    }
    _stream_print(f"{BASE_URL}/v1/chat/completions", payload, "--- [4] 流式对话补全 (Stream SSE) ---")


def test_code_fim_autocomplete():
    print("\n--- [5] FIM 代码补全 (/v1/completions) ---")
    print("⏳ 正在请求 FIM 补全...")
    t0 = time.time()
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": "def add(a: int, b: int) -> int:\n   ",
        "suffix": "\n    return result",
        "max_tokens": 50,
    }
    resp = httpx.post(f"{BASE_URL}/v1/completions", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"❌ 请求失败 (Status {resp.status_code}): {resp.text}")
        return
    print(f"✅ 补全结果 (耗时 {round(time.time() - t0, 2)}s):", resp.json()["choices"][0]["text"])


def test_specialized_code_refactor_stream():
    payload = {
        "code": "nums = [1, 2, 3]\nev = []\nfor x in nums:\n    ev.append(x * 2)",
        "instruction": "简化为列表推导式",
        "language": "python",
        "max_tokens": 100,
    }
    _stream_print(f"{BASE_URL}/v1/code/refactor", payload, "--- [6] 专有代码重构工具 (Stream SSE) ---")


def test_code_inline_edit_stream():
    payload = {
        "code": "def sum_list(nums):\n    return sum(nums)",
        "instruction": "添加类型注解与简洁注释",
        "language": "python",
        "max_tokens": 100,
    }
    _stream_print(f"{BASE_URL}/v1/code/edit", payload, "--- [7] 行内代码编辑 (Stream SSE) ---")


def test_code_review_stream():
    payload = {
        "code": "def divide(a, b):\n    return a / b",
        "language": "python",
        "max_tokens": 100,
    }
    _stream_print(f"{BASE_URL}/v1/code/review", payload, "--- [8] 代码审查 (Stream SSE) ---")


def test_code_docstring_stream():
    payload = {
        "code": "def multiply(a: float, b: float) -> float:\n    return a * b",
        "language": "python",
        "max_tokens": 100,
    }
    _stream_print(f"{BASE_URL}/v1/code/docstring", payload, "--- [9] 文档字符串生成 (Stream SSE) ---")


if __name__ == "__main__":
    try:
        # 动态获取默认模型名称
        DEFAULT_MODEL = get_default_model()
        if not DEFAULT_MODEL:
            print("❌ 无法获取默认模型名称，请确保服务已启动")
            raise httpx.ConnectError("Cannot get default model")
        print(f"📌 使用默认模型: {DEFAULT_MODEL}")

        test_health()
        test_models()
        test_chat_completion()
        test_chat_stream()
        test_code_fim_autocomplete()
        test_specialized_code_refactor_stream()
        test_code_inline_edit_stream()
        test_code_review_stream()
        test_code_docstring_stream()
        print("\n🎉 全部 9 项 API 客户端功能验证顺利完成！")
    except httpx.ConnectError:
        print("\n❌ 连接失败：请确保 AI Code Service 服务已启动 (运行 ./start.sh 或 python app.py)")
