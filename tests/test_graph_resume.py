"""End-to-end: plan -> propose_diff -> pause -> (process dies) -> resume -> execute.

No Ollama. The model is stubbed so the test exercises the graph, the
checkpointer and the diff engine, which is where the Week-1 bugs lived.
"""
import sqlite3
import subprocess
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import graph as graph_mod
from app.nodes import plan as plan_mod
from app.nodes import propose_diff as diff_mod
from app.schemas import ChangePlan, FileRewrite

BEFORE = "def parse_query(q):\n    return q.strip().split()\n"
AFTER = (
    "def parse_query(q):\n"
    "    if not q.strip():\n"
    "        raise ValueError('empty query')\n"
    "    return q.strip().split()\n"
)

PLAN = ChangePlan(
    target_file="search.py",
    summary="Reject empty queries",
    steps=["Raise ValueError when the query is blank"],
    rationale="parse_query silently returns [] today",
)


class _StubPlan:
    def invoke(self, _messages):
        return PLAN.model_copy()


class _StubRewrite:
    """Mimics with_structured_output(..., include_raw=True)."""

    def __init__(self, content=AFTER, done_reason="stop", parsed=True):
        self.content, self.done_reason, self.parsed = content, done_reason, parsed

    def invoke(self, _messages):
        raw = type("Msg", (), {"response_metadata": {"done_reason": self.done_reason}})()
        if not self.parsed:
            return {"raw": raw, "parsed": None,
                    "parsing_error": ValueError("unterminated string")}
        return {
            "raw": raw,
            "parsed": FileRewrite(
                path="search.py",
                commit_message="fix(search): reject empty queries",
                new_content=self.content,
            ),
            "parsing_error": None,
        }


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "search.py").write_text(BEFORE)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True,
    )
    return repo


@pytest.fixture
def stub(monkeypatch):
    """Patch both nodes' LLM factories. Returns a knob for the rewrite stub."""
    box = {"rewrite": _StubRewrite()}

    class _Chain:
        def __init__(self, which):
            self.which = which

        def with_structured_output(self, *_a, **_k):
            return _StubPlan() if self.which == "plan" else box["rewrite"]

    monkeypatch.setattr(plan_mod, "get_llm", lambda **_k: _Chain("plan"))
    monkeypatch.setattr(diff_mod, "get_llm", lambda **_k: _Chain("diff"))
    return box


def _build(db: Path):
    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return graph_mod.build_graph(saver)


def test_pause_survives_a_process_restart_and_resumes(fixture_repo, stub, tmp_path):
    """The Week-1 acceptance criterion, as a test."""
    db = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "t1"}}

    result = _build(db).invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config=cfg,
    )
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "diff_proposed"
    assert payload["plan"]["summary"] == "Reject empty queries"
    assert payload["stats"] == {"additions": 2, "deletions": 0}
    assert (fixture_repo / "search.py").read_text() == BEFORE, "nothing applied yet"

    # --- pretend the process died here; nothing but the sqlite file survives ---
    graph = _build(db)
    snap = graph.get_state(cfg)
    assert snap.next == ("await_approval",)
    assert snap.values["task"] == "make search.py reject empty queries"
    assert snap.values["repo_path"] == str(fixture_repo)

    out = graph.invoke(Command(resume={"decision": "approved", "note": ""}), config=cfg)
    assert not out.get("__interrupt__")
    assert not out.get("error")
    assert (fixture_repo / "search.py").read_text() == AFTER


def test_restarting_with_an_empty_task_does_not_wipe_the_thread(fixture_repo, stub, tmp_path):
    """`run '' --thread X` used to replan against task='' and repo='.'.

    The plan node now refuses instead of producing a mystifying empty diff.
    """
    db = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "t2"}}
    graph = _build(db)
    graph.invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config=cfg,
    )
    out = graph.invoke({"task": "", "repo_path": ".", "retry_count": 0}, config=cfg)
    assert "Empty task" in out["error"]
    assert "resume" in out["error"]


def test_rejection_applies_nothing(fixture_repo, stub, tmp_path):
    db = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "t3"}}
    graph = _build(db)
    graph.invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config=cfg,
    )
    graph.invoke(Command(resume={"decision": "rejected", "note": ""}), config=cfg)
    assert (fixture_repo / "search.py").read_text() == BEFORE


def test_unchanged_rewrite_reports_empty_diff_not_a_crash(fixture_repo, stub, tmp_path):
    stub["rewrite"] = _StubRewrite(content=BEFORE)
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config={"configurable": {"thread_id": "t4"}},
    )
    assert "Empty diff" in out["error"]


def test_truncated_generation_is_named_as_such(fixture_repo, stub, tmp_path):
    """A hit on num_predict must not masquerade as a schema violation."""
    stub["rewrite"] = _StubRewrite(done_reason="length", parsed=False)
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config={"configurable": {"thread_id": "t5"}},
    )
    assert "num_predict" in out["error"]
    assert "OLLAMA_NUM_PREDICT_REWRITE" in out["error"]


def test_missing_repo_is_a_clear_error(stub, tmp_path):
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "fix search.py", "repo_path": "/nope/not/here", "retry_count": 0},
        config={"configurable": {"thread_id": "t6"}},
    )
    assert "does not exist" in out["error"]


def test_venv_is_not_a_candidate(fixture_repo, stub, tmp_path):
    """A checked-out venv must not make target resolution ambiguous."""
    junk = fixture_repo / ".venv" / "lib"
    junk.mkdir(parents=True)
    (junk / "something.py").write_text("x = 1\n")
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "reject empty queries", "repo_path": str(fixture_repo),
         "retry_count": 0},
        config={"configurable": {"thread_id": "t7"}},
    )
    # search.py is the only real candidate, so it resolves without being named.
    assert not out.get("error")
    assert out["__interrupt__"][0].value["plan"]["target_file"] == "search.py"
