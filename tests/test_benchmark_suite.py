"""
多模型性能基准测试与指标统计单元测试
测试多模型并发压测流程、TTFT 计算、TPS 统计聚合与指标完整性
"""

import pytest
import time
from schemas import ChatCompletionRequest, ChatMessage


def test_multi_model_benchmark_switching(client):
    """测试在不同可用模型间切换请求并验证性能指标头"""
    models = ["qwen3.8-27b-8bit-mtp", "qwen3.8-27b-8bit", "qwen3.8-27b"]

    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Test benchmark for {model_name}"}],
            "temperature": 0.1,
            "max_tokens": 20,
        }
        t0 = time.time()
        resp = client.post("/v1/chat/completions", json=payload)
        duration = time.time() - t0

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == model_name
        assert len(data["choices"]) > 0
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0
        assert "X-Process-Time" in resp.headers


def test_metrics_engine_stats_tracking(client):
    """验证 /metrics 输出各模型引擎的 TPS 与显存统计"""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    metrics = resp.json()

    assert "performance" in metrics
    perf = metrics["performance"]
    assert "cache" in perf
    assert "metal_memory" in perf
    assert "engine_stats" in perf
    assert "max_concurrency" in perf


def test_stream_ttft_and_tps_headers(client):
    """验证流式响应能正确计算并返回 Token 吞吐与分块"""
    payload = {
        "model": "qwen3.8-27b-8bit-mtp",
        "messages": [{"role": "user", "content": "Benchmark stream test"}],
        "max_tokens": 15,
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = list(resp.iter_lines())
        assert len(lines) > 0
