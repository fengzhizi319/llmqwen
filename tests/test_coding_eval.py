import json
from types import SimpleNamespace

from agent import eval_humaneval as evaluation


def arguments(tmp_path, **overrides):
    values = dict(output_dir=tmp_path, resume=False, request_retries=0,
                  base_url="http://unused/v1", model="test", timeout=1,
                  temperature=0, max_tokens=128, test_timeout=2,
                  dump_code=False, limit=0)
    return SimpleNamespace(**(values | overrides))


def tasks():
    return [{"uid": str(i), "tier": "test", "entry_point": "answer",
             "prompt": "def answer():\n", "test": "def check(candidate):\n    assert candidate() == 42"}
            for i in range(2)]


def test_resume_preserves_code_failure_but_retries_request_error(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"uid": "failed_code", "pass": False, "status": "test_failed", "err": "AssertionError"},
        {"uid": "retry", "pass": False, "status": "request_error", "err": "request: connection refused"},
        {"uid": "legacy_retry", "pass": False, "err": "request: timeout"},
        {"uid": "recovered", "status": "request_error"},
        {"uid": "recovered", "pass": True, "status": "passed"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records))
    assert evaluation.done_uids(path) == {"failed_code", "recovered"}


def test_request_failure_cannot_produce_complete_score(tmp_path, monkeypatch):
    def fail(*args):
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr(evaluation, "call_chat", fail)
    summary = evaluation.evaluate("humaneval", tasks(), arguments(tmp_path))
    assert summary["complete"] is False
    assert summary["completed_generation_and_test"] == 0
    assert summary["infrastructure_errors"] == 1
    assert summary["missing"] == 1


def test_completed_failures_are_not_resampled(tmp_path, monkeypatch):
    calls = []

    def generate(*args):
        calls.append(args)
        return {"text": "def answer():\n    return 0", "elapsed": 0.1,
                "completion_tokens": 8, "finish_reason": "stop"}

    monkeypatch.setattr(evaluation, "call_chat", generate)
    first = evaluation.evaluate("humaneval", tasks(), arguments(tmp_path))
    assert first["complete"] is True
    assert first["passed"] == 0
    assert first["completed_generation_and_test"] == 2
    second = evaluation.evaluate("humaneval", tasks(), arguments(tmp_path, resume=True))
    assert second["complete"] is True
    assert len(calls) == 2
    records = [json.loads(line) for line in (tmp_path / "humaneval.jsonl").read_text().splitlines()]
    assert all(row["code"] and row["model_output"] for row in records)
