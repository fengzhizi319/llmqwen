"""
AI Code Service - 模型引擎抽象基类
提供同步与异步非阻塞推理接口定义
"""

import asyncio
import concurrent.futures
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, AsyncGenerator, Optional, List, Union


class BaseModelEngine(ABC):
    """模型推理引擎抽象基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """非流式文本生成（同步阻塞）"""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式文本生成（同步生成器）"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """估算或计算 token 数量"""
        pass

    async def async_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """异步非阻塞文本生成（脱离事件循环到线程池执行）"""
        return await asyncio.to_thread(
            self.generate,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

    async def async_stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None,
        chunk_size: int = 1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """异步非阻塞流式生成器，通过后台队列将 token 管道式传输至 asyncio 事件循环。

        支持 chunk_size 聚合：将多个 token 合并为一个 SSE 事件返回，减少网络开销。
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()
        done_sentinel = object()
        cancelled = threading.Event()
        chunk_size = max(1, chunk_size)

        def publish(item):
            if cancelled.is_set():
                return False
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not cancelled.is_set():
                try:
                    future.result(timeout=0.1)
                    return True
                except concurrent.futures.TimeoutError:
                    continue
            future.cancel()
            return False

        def _worker():
            tokens = None
            try:
                if cancelled.is_set():
                    return
                tokens = self.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs,
                )
                buffer = []
                for token in tokens:
                    if cancelled.is_set():
                        break
                    buffer.append(token)
                    if len(buffer) >= chunk_size:
                        if not publish("".join(buffer)):
                            break
                        buffer.clear()
                if buffer and not cancelled.is_set():
                    publish("".join(buffer))
            except Exception as exc:
                publish(exc)
            finally:
                # Close on the producing thread to release engine locks safely.
                try:
                    if tokens is not None:
                        tokens.close()
                finally:
                    publish(done_sentinel)

        worker_task = asyncio.create_task(asyncio.to_thread(_worker))
        try:
            while True:
                item = await queue.get()
                if item is done_sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancelled.set()
            # Running inference cannot be interrupted mid-step; the worker exits
            # at the next yielded token or queue timeout.
            worker_task.cancel()

    def health_check(self) -> bool:
        """检查引擎可用性"""
        return True

    def unload_model(self) -> None:
        """释放模型权重与显存，子类可覆写实现真正的资源回收"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎运行指标"""
        return {}
