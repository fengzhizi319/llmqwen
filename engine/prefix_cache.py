"""Bounded, isolated prefill snapshots for hybrid recurrent/attention models.

The caller holds the engine lock. Never trim a recurrent snapshot: restore an
exact token prefix and replay the suffix instead. Generation never owns a stored
snapshot directly.
"""

import copy
import threading
from collections import OrderedDict


class PrefixSnapshotCache:
    def __init__(self, max_entries=4, max_tokens=8192):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    def restore(self, tokens):
        with self._lock:
            return self._restore(tokens)

    def _restore(self, tokens):
        best = None
        for prefix in self._entries:
            # Always leave at least one token for computing next-token logits.
            if len(prefix) < len(tokens) and tuple(tokens[:len(prefix)]) == prefix:
                if best is None or len(prefix) > len(best):
                    best = prefix
        if best is None:
            self.misses += 1
            return None, 0
        snapshot = copy.deepcopy(self._entries[best])
        self._entries.move_to_end(best)
        self.hits += 1
        self.reused_tokens += len(best)
        return snapshot, len(best)

    def store(self, tokens, state):
        if not tokens or len(tokens) > self.max_tokens or self.max_entries <= 0:
            return
        prefix = tuple(tokens)
        snapshot = copy.deepcopy(state)
        with self._lock:
            self._entries[prefix] = snapshot
            self._entries.move_to_end(prefix)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = self.reused_tokens = 0

    def stats(self):
        with self._lock:
            return self._stats()

    def _stats(self):
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "max_tokens_per_entry": self.max_tokens,
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "retained_tokens": sum(map(len, self._entries)),
            "state_bytes": sum(
                getattr(layer, "nbytes", 0)
                for entry in self._entries.values() for layer in entry
            ),
        }
