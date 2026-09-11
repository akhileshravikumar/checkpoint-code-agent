"""Retry exhaustion must end in an error, not a silent END that the dashboard
reports as `finished: approved` for a PR that is still red."""
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import graph as graph_mod
from app.nodes import execute as ex
from tests.test_execute_node import DIFF, PLAN, TOKEN, FakeClient


@pytest.fixture
def world(monkeypatch, tmp_path):
    calls = {"ci": []}
    monkeypatch.setattr(ex, "GitHubClient", lambda: FakeClient(calls))
    monkeypatch.setattr(ex, "apply_patch",
                        lambda repo, diff: calls.setdefault("applied", []).append(diff))

    class S:
        ollama_model = "qwen2.5-coder:3b"
        github_token = TOKEN
    monkeypatch.setattr(ex, "get_settings", lambda: S())
    monkeypatch.setattr(graph_mod, "plan_node",
                        lambda state: {"plan": PLAN.model_dump(), "error": ""})
    monkeypatch.setattr(graph_mod, "propose_diff_node", lambda state: {
        "diff": DIFF.replace("pass", f"pass  # attempt {state.get('retry_count', 0) + 1}"),
        "commit_message": "fix(search): validate query",
        "approval_status": "pending", "error": ""})

    def watch_ci(state, config=None):
        verdict = calls["ci"].pop(0)
        return {"ci_status": verdict, "ci_run_url": "https://ci/run",
                "ci_failure_log": "E   TypeError" if verdict == "failed" else ""}
    monkeypatch.setattr(graph_mod, "watch_ci_node", watch_ci)

    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
    saver.setup()
    return graph_mod.build_graph(saver), calls, str(tmp_path)


def _start(graph, repo, tid):
    cfg = {"configurable": {"thread_id": tid}}
    graph.invoke({"task": "fix sandbox/search.py", "repo_path": repo, "retry_count": 0},
                 config=cfg)
    return cfg


def test_ci_exhaustion_is_an_error_not_a_success(world):
    graph, calls, repo = world
    calls["ci"] = ["failed", "failed", "failed"]
    cfg = _start(graph, repo, "g1")
    gates = 1
    while True:
        out = graph.invoke(Command(resume={"decision": "approved"}), config=cfg)
        if not out.get("__interrupt__"):
            break
        gates += 1
    assert gates == 3, "MAX_RETRIES=2 means the original gate plus two retries"
    assert "CI still failed after 3 attempts" in out["error"]
    assert "https://github.com/o/r/pull/1" in out["error"], "name the PR left open"
    assert len(calls["pushed"]) == 3


def test_edit_limit_is_an_error(world):
    graph, calls, repo = world
    cfg = _start(graph, repo, "g2")
    for _ in range(2):
        out = graph.invoke(Command(resume={"decision": "edit_requested", "note": "x"}),
                           config=cfg)
        assert out["__interrupt__"]
    out = graph.invoke(Command(resume={"decision": "edit_requested", "note": "x"}),
                       config=cfg)
    assert "Edit limit reached after 3 attempts" in out["error"]
    assert "pushed" not in calls


def test_rejection_is_a_clean_stop_not_an_error(world):
    graph, calls, repo = world
    cfg = _start(graph, repo, "g3")
    out = graph.invoke(Command(resume={"decision": "rejected"}), config=cfg)
    assert not out.get("error")
    assert "pushed" not in calls
