"""
AI Code Service - 高性能响应与 Prompt 缓存模块
基于线程安全的 LRU 缓存与 TTL 过期机制，极大降低高频/重复请求延迟与计算开销
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class ResponseCache:
    """线程安全的 LRU + TTL 响应缓存器"""

    def __init__(self, max_size: int = 1024, ttl_seconds: int = 3600, enabled: bool = True):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        
        # 性能统计指标
        self._hits = 0
        self._misses = 0

    @staticmethod
    def generate_key(
        prefix: str,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Any = None,
        **extra
    ) -> str:
        """根据请求参数生成唯一的 SHA256 缓存键"""
        raw_key = {
            "prefix": prefix,
            "model": model,
            "prompt": prompt,
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
            "stop": stop,
            "extra": extra,
        }
        encoded = json.dumps(raw_key, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存内容，自动处理 TTL 过期与 LRU 顺序更新"""
        if not self.enabled:
            return None

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]
            now = time.time()

            # 检查是否过期
            if self.ttl_seconds > 0 and (now - timestamp) > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # 移动到最近使用 (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """写入缓存，超出容量时淘汰最久未使用的项 (LRU)"""
        if not self.enabled:
            return

        with self._lock:
            now = time.time()
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, now)

            # 超出最大容量时移除最旧元素
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存与计数器"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存命中率与统计数据"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "enabled": self.enabled,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }
