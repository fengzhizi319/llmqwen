"""
模型引擎与 Prompt 构建单元测试
"""

from config import AppConfig, ModelSpec
from engine import ModelManager, MockModelEngine


def test_mock_engine_token_counter():
    engine = MockModelEngine(model_name="test-mock")
    assert engine.count_tokens("") == 0
    assert engine.count_tokens("hello world") >= 1
    assert engine.health_check() is True


def test_prompt_builder():
    cfg = AppConfig(
        default_model="m1",
        models={"m1": ModelSpec(path="path1")},
        system_prompt="Base System Prompt",
    )
    manager = ModelManager(cfg)

    # 包含 system 消息
    messages = [
        {"role": "system", "content": "Custom System"},
        {"role": "user", "content": "Write a binary search function"},
    ]
    prompt = manager.build_chat_prompt(messages)
    assert "<|im_start|>system\nCustom System<|im_end|>" in prompt
    assert "<|im_start|>user\nWrite a binary search function<|im_end|>" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_fim_prompt_builder():
    cfg = AppConfig()
    manager = ModelManager(cfg)

    prefix = "def calculate():\n   "
    suffix = "\n    return res"
    fim_prompt = manager.build_fim_prompt(prefix, suffix)

    assert "<|fim_prefix|>" in fim_prompt
    assert "<|fim_suffix|>" in fim_prompt
    assert "<|fim_middle|>" in fim_prompt
