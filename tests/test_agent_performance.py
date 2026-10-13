"""Regression tests for agent cancellation, backpressure and early stopping."""
import asyncio
import threading

import pytest

from engine.mock_engine import MockModelEngine
from engine.mlx_engine import MLXModelEngine
from engine.qwen4_exp_engine import Qwen4ExpEngine


@pytest.mark.asyncio
async def test_cancelled_stream_releases_worker():
    closed = threading.Event()

    class Engine(MockModelEngine):
        def stream_generate(self, *args, **kwargs):
            try:
                while True:
                    yield "token"
            finally:
                closed.set()

    stream = Engine().async_stream_generate("prompt")
    assert await anext(stream) == "token"
    await stream.aclose()
    assert await asyncio.to_thread(closed.wait, 2)


@pytest.mark.asyncio
async def test_slow_consumer_bounds_generation():
    count = [0]
    closed = threading.Event()

    class Engine(MockModelEngine):
        def stream_generate(self, *args, **kwargs):
            try:
                while True:
                    count[0] += 1
                    yield "token"
            finally:
                closed.set()

    stream = Engine().async_stream_generate("prompt")
    await anext(stream)
    await asyncio.sleep(0.05)
    assert count[0] <= 34  # one consumed, 32 queued, one awaiting publication
    await stream.aclose()
    assert await asyncio.to_thread(closed.wait, 2)


@pytest.mark.parametrize("engine_type", [MLXModelEngine, Qwen4ExpEngine])
def test_nonstream_agent_stops_without_extra_inference(engine_type):
    engine = engine_type(model_name="test", model_path="unused", enable_mtp=False)
    engine.load_model = lambda: None
    engine._build_sampler_kwargs = lambda *args, **kwargs: {}
    engine.model = object()
    engine.tokenizer = object()
    engine.generate_fn = lambda *args, **kwargs: pytest.fail("slow generate path used")
    if engine_type is Qwen4ExpEngine:
        engine.strip_thinking = False
        engine._prepare_native_stream = lambda prompt: (engine.tokenizer, prompt, [])
    consumed = []

    def native_stream(*args, **kwargs):
        for text in ["tool payload", "</tool_call>", "unnecessary output"]:
            consumed.append(text)
            yield text

    engine.stream_generate_fn = native_stream
    assert engine.generate("prompt", stop=["</tool_call>"]) == "tool payload"
    assert consumed == ["tool payload", "</tool_call>"]
