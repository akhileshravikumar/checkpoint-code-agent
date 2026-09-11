"""ADR-004: with CHECKPOINT_OFFLINE=1 an approval applies locally and nothing
reaches GitHub. LLM nodes are stubbed; the patch is applied for real."""
import sqlite3
import subprocess

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import github_client as gc
from app import graph as graph_mod
from app.diffing import build_unified_diff
from app.nodes import execute as ex
from app.schemas import ChangePlan

BEFORE = "def parse_query(q):\n    return q.strip().split()\n"
AFTER = "def parse_query(q):\n    if not q.strip():\n        raise ValueError('empty')\n    return q.strip().split()\n"
PLAN = ChangePlan(target_file="search.py", summary="Reject empty queries",
                  steps=["Raise ValueError when blank"], rationale="returns [] today")


class Offline:
    checkpoint_offline = True
    max_retries = 2


@pytest.fixture
def offline_graph(monkeypatch, tmp_path):
    repo = tmp_path / "ws"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "search.py").write_text(BEFORE)

    monkeypatch.setattr(graph_mod, "get_settings", lambda: Offline())
    monkeypatch.setattr(graph_mod, "plan_node",
                        lambda state: {"plan": PLAN.model_dump(), "error": ""})
    monkeypatch.setattr(graph_mod, "propose_diff_node", lambda state: {
        "diff": build_unified_diff("search.py", BEFORE, AFTER),
        "commit_message": "fix(search): reject empty queries",
        "approval_status": "pending", "error": ""})

    def no_network(*_a, **_k):
        raise AssertionError("offline mode reached a GitHub node")
    monkeypatch.setattr(ex, "GitHubClient", no_network)
    monkeypatch.setattr(graph_mod, "watch_ci_node", no_network)

    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
    saver.setup()
    return graph_mod.build_graph(saver), repo


def test_offline_approval_applies_locally_and_never_builds_a_client(offline_graph):
    graph, repo = offline_graph
    cfg = {"configurable": {"thread_id": "off1"}}
    gate = graph.invoke({"task": "fix search.py", "repo_path": str(repo),
                         "retry_count": 0}, config=cfg)
    assert gate["__interrupt__"]
    assert (repo / "search.py").read_text() == BEFORE, "nothing applied before approval"

    out = graph.invoke(Command(resume={"decision": "approved"}), config=cfg)
    assert not out.get("error")
    assert out.get("pr_url") is None
    assert (repo / "search.py").read_text() == AFTER
    assert graph.get_state(cfg).next == (), "offline must end after the local apply"


def test_offline_rejection_still_applies_nothing(offline_graph):
    graph, repo = offline_graph
    cfg = {"configurable": {"thread_id": "off2"}}
    graph.invoke({"task": "fix search.py", "repo_path": str(repo), "retry_count": 0}, config=cfg)
    graph.invoke(Command(resume={"decision": "rejected"}), config=cfg)
    assert (repo / "search.py").read_text() == BEFORE


def test_github_client_refuses_to_start_offline(monkeypatch):
    class S(Offline):
        github_token = "github_pat_x"
    monkeypatch.setattr(gc, "get_settings", lambda: S())
    with pytest.raises(RuntimeError, match="CHECKPOINT_OFFLINE"):
        gc.GitHubClient()
