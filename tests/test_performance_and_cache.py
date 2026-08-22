"""
性能优化与响应缓存单元测试
测试 ResponseCache 命中机制、TTL 过期、LRU 淘汰与 API 性能响应头
"""

import asyncio
import time
import pytest
from engine.cache import ResponseCache
from engine.mock_engine import MockModelEngine
from schemas import ChatCompletionRequest, ChatMessage, CompletionRequest


def test_response_cache_basic_operations():
    cache = ResponseCache(max_size=3, ttl_seconds=2, enabled=True)
    key1 = ResponseCache.generate_key("test", "model1", "prompt1")
    key2 = ResponseCache.generate_key("test", "model1", "prompt2")
    key3 = ResponseCache.generate_key("test", "model1", "prompt3")
    key4 = ResponseCache.generate_key("test", "model1", "prompt4")

    # 1. 初始未命中
    assert cache.get(key1) is None
    
    # 2. 写入与命中
    cache.set(key1, "val1")
    assert cache.get(key1) == "val1"

    # 3. LRU 淘汰测试 (容量为 3)
    cache.set(key2, "val2")
    cache.set(key3, "val3")
    # key1, key2, key3 存在。再次读取 key1，使其变为最近使用
    cache.get(key1)
    # 插入 key4，最久未使用的 key2 应该被淘汰
    cache.set(key4, "val4")

    assert cache.get(key1) == "val1"
    assert cache.get(key2) is None  # 被淘汰
    assert cache.get(key3) == "val3"
    assert cache.get(key4) == "val4"


def test_response_cache_ttl_expiry():
    cache = ResponseCache(max_size=10, ttl_seconds=1, enabled=True)
    key = ResponseCache.generate_key("ttl", "model", "prompt")
    cache.set(key, "temp_value")
    assert cache.get(key) == "temp_value"

    # 等待过期
    time.sleep(1.1)
    assert cache.get(key) is None


def test_api_cache_headers_and_subsequent_hits(client):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "Explain binary search"}],
        "temperature": 0.0,
    }

    # 首次调用：缓存 MISS
    resp1 = client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"
    assert "X-Prompt-Tokens" in resp1.headers
    assert "X-Completion-Tokens" in resp1.headers

    # 第二次相同调用：缓存 HIT (< 1ms)
    resp2 = client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"
    assert resp1.json()["choices"][0]["message"]["content"] == resp2.json()["choices"][0]["message"]["content"]


def test_completions_cache_headers(client):
    payload = {
        "model": "qwen3.8-27b",
        "prompt": "def add(a, b):\n   ",
        "suffix": "\n    return res",
    }

    resp1 = client.post("/v1/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"

    resp2 = client.post("/v1/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"


def test_metrics_performance_block(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "performance" in data
    perf = data["performance"]
    assert "cache" in perf
    assert perf["cache"]["enabled"] is True
    assert "hits" in perf["cache"]
    assert "misses" in perf["cache"]


def test_stream_chunk_aggregation():
    """验证 async_stream_generate 的 chunk_size 聚合逻辑"""
    engine = MockModelEngine()
    prompt = "explain this code"

    async def collect_chunks(chunk_size: int):
        chunks = []
        async for chunk in engine.async_stream_generate(
            prompt, max_tokens=100, chunk_size=chunk_size
        ):
            chunks.append(chunk)
        return chunks

    full_text = engine.generate(prompt, max_tokens=100)

    single_chunks = asyncio.run(collect_chunks(1))
    assert "".join(single_chunks) == full_text

    triple_chunks = asyncio.run(collect_chunks(3))
    assert "".join(triple_chunks) == full_text
    assert len(triple_chunks) <= len(single_chunks)
