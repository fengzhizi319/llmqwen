"""Prefix matching must preserve complete hybrid state and request isolation."""

from engine.prefix_cache import PrefixSnapshotCache


def test_restore_uses_longest_exact_prefix_and_leaves_a_token():
    cache = PrefixSnapshotCache()
    cache.store([1, 2], [{"recurrent": [20]}])
    cache.store([1, 2, 3], [{"recurrent": [30]}])
    state, reused = cache.restore([1, 2, 3, 4])
    assert reused == 3
    assert state[0]["recurrent"] == [30]
    state, reused = cache.restore([1, 2, 3])
    assert reused == 2
    assert state[0]["recurrent"] == [20]


def test_fork_and_shorter_requests_never_trim_recurrent_state():
    cache = PrefixSnapshotCache()
    cache.store([1, 2, 3], [{"recurrent": [30]}])
    assert cache.restore([1, 2, 9, 4]) == (None, 0)
    assert cache.restore([1, 2]) == (None, 0)
    assert cache.restore([9, 2, 3, 4]) == (None, 0)


def test_both_storage_and_restore_are_isolated():
    cache = PrefixSnapshotCache()
    state = [{"linear": [1], "ple_conv": [2], "ple_history": [3], "qsa_index": [4]}]
    cache.store([1, 2], state)
    state[0]["linear"][0] = 99
    first, _ = cache.restore([1, 2, 3])
    first[0]["ple_history"][0] = 99
    second, _ = cache.restore([1, 2, 3])
    assert second == [{"linear": [1], "ple_conv": [2], "ple_history": [3], "qsa_index": [4]}]


def test_lru_limits_and_clear():
    cache = PrefixSnapshotCache(max_entries=2, max_tokens=3)
    cache.store([1], [])
    cache.store([2], [])
    cache.restore([1, 4])
    cache.store([3], [])
    assert cache.restore([2, 4]) == (None, 0)
    cache.store([1, 2, 3, 4], [])
    assert cache.stats()["entries"] == 2
    assert cache.stats()["reused_tokens"] == 1
    cache.clear()
    assert cache.stats()["entries"] == cache.stats()["hits"] == 0


def test_zero_limits_disable_storage():
    for cache in [PrefixSnapshotCache(max_entries=0), PrefixSnapshotCache(max_tokens=0)]:
        cache.store([1], [])
        assert cache.restore([1, 2]) == (None, 0)
