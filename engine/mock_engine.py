"""
AI Code Service - Mock 模型推理引擎
用于单元测试、CI/CD 及无 GPU 环境下的快速响应与仿真
"""

import time
import math
import random
from typing import Generator, Optional, List, Union, Dict, Any
from .base import BaseModelEngine


class MockModelEngine(BaseModelEngine):
    """Mock 推理引擎，仿真真实的 AI 编程助手响应"""

    def __init__(self, model_name: str = "mock-qwen"):
        self.model_name = model_name
        self._total_requests = 0
        self._total_prompt_tokens = 0
        self._total_generation_tokens = 0

    def count_tokens(self, text: str) -> int:
        """简单的 token 估算: 约 4 个字符 = 1 个 token"""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        self._total_requests += 1
        p_tokens = self.count_tokens(prompt)
        self._total_prompt_tokens += p_tokens

        # 可选：使用 seed 保证 Mock 输出可复现
        seed = kwargs.get("seed")
        if seed is not None:
            random.seed(seed)

        prompt_lower = prompt.lower()

        # FIM Fill-in-the-middle 判定
        if "<fim_suffix>" in prompt or "<|fim_suffix|>" in prompt or "[SUFFIX]" in prompt:
            res = " return a + b\n"
        elif "解释" in prompt or "explain" in prompt_lower:
            res = "```python\n# 这是一段示例代码解释\n# 核心逻辑: 接收输入并计算结果\n```\n该代码实现了一个高效的功能模块。"
        elif "重构" in prompt or "refactor" in prompt_lower:
            res = "```python\ndef optimized_func(data):\n    \"\"\"重构后的高效实现\"\"\"\n    return [item for item in data if item]\n```"
        elif "单测" in prompt or "test" in prompt_lower:
            res = "```python\nimport pytest\n\ndef test_feature():\n    assert True\n```"
        elif "修改" in prompt or "edit" in prompt_lower:
            res = "```python\n# 行内编辑后的代码示例\ndef optimized_func(data):\n    \"\"\"按需求修改后的实现\"\"\"\n    return [item for item in data if item]\n```"
        elif "审查" in prompt or "review" in prompt_lower:
            res = (
                "## 代码审查意见\n\n"
                "- 建议为函数添加类型注解，提升可读性。\n"
                "- 可将循环改写为列表推导式，提高运行效率。\n"
                "- 缺少边界值处理，建议补充异常处理逻辑。\n\n"
                "```python\ndef optimized_func(data):\n    \"\"\"重构后的实现\"\"\"\n    return [item for item in data if item]\n```"
            )
        elif "文档" in prompt or "docstring" in prompt_lower:
            res = "```python\ndef calculate(x, y):\n    \"\"\"\n    计算两个数值的和。\n\n    Args:\n        x (int/float): 第一个数值。\n        y (int/float): 第二个数值。\n\n    Returns:\n        int/float: 两个数值的和。\n    \"\"\"\n    return x + y\n```"
        else:
            res = f"```python\n# [AI Code Service Response - {self.model_name}]\ndef solution():\n    print('Hello from AI Code Service!')\n    return True\n```"

        self._total_generation_tokens += self.count_tokens(res)
        return res

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        full_text = self.generate(prompt, max_tokens, temperature, top_p, stop, **kwargs)
        
        # 将生成的文本按词拆分，逐块 yield 模拟流式生成
        tokens = full_text.split(" ")
        for i, token in enumerate(tokens):
            suffix = " " if i < len(tokens) - 1 else ""
            yield token + suffix
            time.sleep(0.005)

    def health_check(self) -> bool:
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "engine_type": "mock",
            "loaded": True,
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_generation_tokens": self._total_generation_tokens,
            "last_generation_tps": 120.0,
        }
