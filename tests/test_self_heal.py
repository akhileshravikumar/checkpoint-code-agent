"""W2D10 end to end through the real graph: CI fails, the fix comes back to the
same gate, and approving it pushes a second commit to the same PR.

LLM nodes and CI are stubbed; execute runs for real against a fake GitHub that
counts side effects.
"""
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import graph as graph_mod
from app.nodes import execute as ex
from tests.test_execute_node import DIFF, PLAN, TOKEN, FakeClient


@pytest.fixture
def world(monkeypatch, tmp_path):
    calls = {"ci": ["failed", "passed"]}
    monkeypatch.setattr(ex, "GitHubClient", lambda: FakeClient(calls))
    monkeypatch.setattr(ex, "apply_patch",
                        lambda repo, diff: calls.setdefault("applied", []).append(diff))

    class S:
        ollama_model = "qwen2.5-coder:3b"
        github_token = TOKEN
    monkeypatch.setattr(ex, "get_settings", lambda: S())

    monkeypatch.setattr(graph_mod, "plan_node",
                        lambda state: {"plan": PLAN.model_dump(), "error": ""})

    def propose(state):
        n = state.get("retry_count", 0)
        return {"diff": DIFF.replace("pass", f"pass  # attempt {n + 1}"),
                "commit_message": "fix(search): validate query",
                "approval_status": "pending", "error": ""}
    monkeypatch.setattr(graph_mod, "propose_diff_node", propose)

    def watch_ci(state, config=None):
        calls.setdefault("polled", []).append(state["head_sha"])
        verdict = calls["ci"].pop(0)
        return {"ci_status": verdict, "ci_run_url": "https://ci/run",
                "ci_failure_log": "E   TypeError" if verdict == "failed" else ""}
    monkeypatch.setattr(graph_mod, "watch_ci_node", watch_ci)

    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
    saver.setup()
    return graph_mod.build_graph(saver), calls, str(tmp_path)


def test_ci_failure_regates_and_the_fix_lands_on_the_same_pr(world):
    graph, calls, repo = world
    cfg = {"configurable": {"thread_id": "heal01"}}

    gate1 = graph.invoke({"task": "fix sandbox/search.py", "repo_path": repo,
                          "retry_count": 0}, config=cfg)
    assert gate1["__interrupt__"][0].value["retry_count"] == 0

    gate2 = graph.invoke(Command(resume={"decision": "approved"}), config=cfg)
    assert gate2["__interrupt__"], "a CI failure must come back to the gate"
    assert gate2["__interrupt__"][0].value["retry_count"] == 1
    assert len(calls["pushed"]) == 1, "nothing may be pushed before gate 2 is approved"

    done = graph.invoke(Command(resume={"decision": "approved"}), config=cfg)
    assert done["ci_status"] == "passed"
    assert not done.get("error")
    assert len(calls["pushed"]) == 2
    assert len(calls["opened"]) == 1, "one thread, one PR"
    assert len(set(calls["polled"])) == 2, "attempt 2 must be polled at its own sha"
    assert calls["pushed"][1][1].endswith("Checkpoint-Attempt: 2")


def test_rejecting_the_retry_pushes_nothing_more(world):
    graph, calls, repo = world
    cfg = {"configurable": {"thread_id": "heal02"}}
    graph.invoke({"task": "fix sandbox/search.py", "repo_path": repo,
                  "retry_count": 0}, config=cfg)
    graph.invoke(Command(resume={"decision": "approved"}), config=cfg)
    graph.invoke(Command(resume={"decision": "rejected"}), config=cfg)
    assert len(calls["pushed"]) == 1
