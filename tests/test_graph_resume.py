"""End-to-end: plan -> propose_diff -> pause -> (process dies) -> resume -> execute."""
import sqlite3
import subprocess
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import graph as graph_mod
from app.nodes import execute as exec_mod
from app.nodes import plan as plan_mod
from app.nodes import propose_diff as diff_mod
from app.schemas import ChangePlan

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
    """Mimics a plain completion — ADR-007, the body is not JSON."""

    def __init__(self, content=AFTER, done_reason="stop"):
        self.content, self.done_reason = content, done_reason

    def invoke(self, _messages):
        return type("AIMessage", (), {
            "content": self.content,
            "response_metadata": {"done_reason": self.done_reason},
        })()


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
            return _StubPlan()

        def invoke(self, messages):
            return box["rewrite"].invoke(messages)

    monkeypatch.setattr(plan_mod, "get_llm", lambda **_k: _Chain("plan"))
    monkeypatch.setattr(diff_mod, "get_llm", lambda **_k: _Chain("diff"))
    return box


@pytest.fixture(autouse=True)
def local_github(monkeypatch, tmp_path):
    """Week 2 swapped execute_local_node for a node that pushes to GitHub.

    These tests are about the graph's semantics — that approval applies the
    patch and rejection does not — so GitHub is stubbed down to a local apply.
    Weakening the assertions instead would delete the project's central claim
    from the test suite.
    """
    box = {"repo": None}

    class _Fake:
        def ensure_workspace(self):
            return box["repo"]

        @staticmethod
        def branch_name(task, seed=""):
            return f"checkpoint/stub-{seed or 'x'}"

        def branch_exists(self, name):
            return False

        def create_branch(self, name):
            pass

        def commit_and_push(self, branch, message, paths):
            return "deadbeef"

        def find_pr_for_branch(self, branch):
            return None

        def open_pr(self, branch, title, body):
            return type("PR", (), {"url": "https://github.com/o/r/pull/1",
                                   "head_sha": "deadbeef", "number": 1,
                                   "branch": branch})()

        def close(self):
            pass

    monkeypatch.setattr(exec_mod, "GitHubClient", _Fake)
    # Since W2D9 an approval runs on into watch_ci, which would build a real
    # GitHubClient: "GITHUB_TOKEN is empty" in CI, or ten minutes polling
    # GitHub for sha "deadbeef" with a real .env. CI is green here.
    monkeypatch.setattr(graph_mod, "watch_ci_node", lambda state, config=None: {
        "ci_status": "passed", "ci_run_url": "https://ci/1", "ci_failure_log": "",
    })
    return box


def _build(db: Path):
    saver = SqliteSaver(sqlite3.connect(str(db), check_same_thread=False))
    saver.setup()
    return graph_mod.build_graph(saver)


def test_pause_survives_a_process_restart_and_resumes(fixture_repo, stub, tmp_path, local_github):
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
    assert payload["stats"] == {"additions": 2, "deletions": 0}
    assert (fixture_repo / "search.py").read_text() == BEFORE, "nothing applied yet"

    # --- pretend the process died here; nothing but the sqlite file survives ---
    graph = _build(db)
    snap = graph.get_state(cfg)
    assert snap.next == ("await_approval",)
    assert snap.values["task"] == "make search.py reject empty queries"
    assert snap.values["repo_path"] == str(fixture_repo)

    local_github["repo"] = str(fixture_repo)
    out = graph.invoke(Command(resume={"decision": "approved", "note": ""}), config=cfg)
    assert not out.get("error")
    assert (fixture_repo / "search.py").read_text() == AFTER
    assert out["pr_url"].endswith("/pull/1")


def test_restarting_with_an_empty_task_does_not_wipe_the_thread(fixture_repo, stub, tmp_path):
    """ADR-005: `run '' --thread X` used to replan against task='' and repo='.'."""
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


def test_rejection_applies_nothing(fixture_repo, stub, tmp_path):
    cfg = {"configurable": {"thread_id": "t3"}}
    graph = _build(tmp_path / "cp.sqlite")
    graph.invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config=cfg,
    )
    graph.invoke(Command(resume={"decision": "rejected", "note": ""}), config=cfg)
    assert (fixture_repo / "search.py").read_text() == BEFORE


def test_unchanged_rewrite_is_a_no_change_outcome_not_a_crash(fixture_repo, stub, tmp_path):
    """A task that is already done ends quietly: no error, no gate, nothing applied."""
    stub["rewrite"] = _StubRewrite(content=BEFORE)
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config={"configurable": {"thread_id": "t4"}},
    )
    assert not out.get("error")
    assert "No change proposed for search.py" in out["no_change_reason"]
    assert "__interrupt__" not in out
    assert (fixture_repo / "search.py").read_text() == BEFORE


def test_truncated_generation_is_named_as_such(fixture_repo, stub, tmp_path):
    """A hit on num_predict must not masquerade as a schema violation."""
    stub["rewrite"] = _StubRewrite(content="def parse_query(q):", done_reason="length")
    out = _build(tmp_path / "cp.sqlite").invoke(
        {"task": "make search.py reject empty queries",
         "repo_path": str(fixture_repo), "retry_count": 0},
        config={"configurable": {"thread_id": "t5"}},
    )
    assert "num_predict" in out["error"]


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
    assert not out.get("error")
    assert out["__interrupt__"][0].value["plan"]["target_file"] == "search.py"


# --- one thread, two tasks: the second must not inherit the first's target ---

class _RecordingPlan:
    """Captures the prompt and targets whichever file the task names."""

    def __init__(self, seen):
        self.seen = seen

    def invoke(self, messages):
        user = messages[1][1]
        self.seen.append(user)
        target = "ranker.py" if "ranker.py" in user else "search.py"
        return ChangePlan(target_file=target, summary="s",
                          steps=["do it"], rationale="r")


@pytest.fixture
def two_file_repo(fixture_repo):
    (fixture_repo / "ranker.py").write_text("def rank(xs):\n    return sorted(xs)\n")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(fixture_repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "add ranker"], check=True,
    )
    return fixture_repo


@pytest.fixture
def recording(monkeypatch, two_file_repo):
    seen = []

    class _Append:
        def invoke(self, _messages):
            target = "ranker.py" if "ranker.py" in seen[-1] else "search.py"
            cur = (two_file_repo / target).read_text()
            return type("AIMessage", (), {
                "content": cur.rstrip("\n") + "\n# edit\n",
                "response_metadata": {"done_reason": "stop"},
            })()

    class _Chain:
        def __init__(self, which):
            self.which = which

        def with_structured_output(self, *_a, **_k):
            return _RecordingPlan(seen)

        def invoke(self, messages):
            return _Append().invoke(messages)

    monkeypatch.setattr(plan_mod, "get_llm", lambda **_k: _Chain("plan"))
    monkeypatch.setattr(diff_mod, "get_llm", lambda **_k: _Chain("diff"))
    return seen


def _target_of(prompt: str) -> str:
    return prompt.split("File: ")[1].split("\n")[0]


def test_new_task_on_a_finished_thread_retargets(two_file_repo, recording, tmp_path, local_github):
    """The bug: task 2 named ranker.py and was planned against search.py."""
    cfg = {"configurable": {"thread_id": "reuse"}}
    graph = _build(tmp_path / "cp.sqlite")

    graph.invoke({"task": "fix search.py", "repo_path": str(two_file_repo),
                  "retry_count": 0}, config=cfg)
    local_github["repo"] = str(two_file_repo)
    graph.invoke(Command(resume={"decision": "approved", "note": ""}), config=cfg)
    assert _target_of(recording[-1]) == "search.py"

    graph.invoke({"task": "sort the output of ranker.py",
                  "repo_path": str(two_file_repo), "retry_count": 0}, config=cfg)
    assert _target_of(recording[-1]) == "ranker.py"


def test_a_replan_stays_on_the_reviewed_file(two_file_repo, recording, tmp_path):
    """The other half: edit_requested must NOT re-resolve the target."""
    cfg = {"configurable": {"thread_id": "replan"}}
    graph = _build(tmp_path / "cp.sqlite")

    graph.invoke({"task": "fix search.py", "repo_path": str(two_file_repo),
                  "retry_count": 0}, config=cfg)
    assert _target_of(recording[-1]) == "search.py"

    # A note that names the other file must not drag the replan onto it.
    graph.invoke(Command(resume={"decision": "edit_requested",
                                 "note": "same idea as in ranker.py please"}),
                 config=cfg)
    assert _target_of(recording[-1]) == "search.py"


# --- the agent must not be able to edit the specification it is graded against ---

def test_the_agent_cannot_target_the_test_suite(fixture_repo, stub, tmp_path):
    """W2D10 grades the agent on CI. If it can edit tests, deleting the failing
    test is the cheapest way to go green — and the loop would find that."""
    from app.nodes.plan import PlanError, _resolve_target

    for rel in ["tests/test_search.py", "tests/conftest.py", "demo/seed_failing.py"]:
        f = fixture_repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x = 1\n")

    # Named explicitly in the task, it still must not be reachable.
    for task in ["make CI pass by editing tests/conftest.py",
                 "delete the failing case in test_search.py",
                 "change demo/seed_failing.py"]:
        target = _resolve_target({"task": task}, fixture_repo)
        assert target.relative_to(fixture_repo).as_posix() == "search.py", task

    # And with no source file to fall back on, it refuses rather than picking one.
    (fixture_repo / "search.py").unlink()
    with pytest.raises(PlanError, match="No Python files"):
        _resolve_target({"task": "edit tests/conftest.py"}, fixture_repo)