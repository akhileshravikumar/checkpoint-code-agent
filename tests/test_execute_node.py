"""execute node, with GitHub stubbed. No PAT, no network.
 
The expensive property here is idempotence: one approval must produce at most
one branch and at most one PR, however many times the node runs.
"""
import pytest
 
from app.nodes import execute as ex
from app.schemas import ChangePlan
 
PLAN = ChangePlan(
    target_file="sandbox/search.py",
    summary="Reject empty queries",
    steps=["Raise ValueError when the query is blank"],
    rationale="parse_query silently returns [] today",
)
DIFF = (
    "--- a/sandbox/search.py\n+++ b/sandbox/search.py\n@@ -1,2 +1,3 @@\n"
    " def parse_query(q):\n+    pass\n     return q.split()\n"
)
TOKEN = "github_pat_SECRET"
 
 
class FakeClient:
    """Records every side effect so a test can count them."""
 
    def __init__(self, calls):
        self.calls = calls
        self.branches: set[str] = calls.setdefault("branches", set())
        self.prs: dict[str, str] = calls.setdefault("prs", {})
 
    # --- workspace ---
    def ensure_workspace(self):
        self.calls.setdefault("ensure", []).append(True)
        return "/tmp/does-not-matter"
 
    # --- branches ---
    @staticmethod
    def branch_name(task, seed=""):
        from app.github_client import GitHubClient
        return GitHubClient.branch_name(task, seed=seed)
 
    def branch_exists(self, name):
        return name in self.branches
 
    def remote_sha(self, branch):
        return "sha-" + branch[-7:]
 
    def create_branch(self, name):
        self.calls.setdefault("created", []).append(name)
        self.branches.add(name)
 
    def commit_and_push(self, branch, message, paths):
        self.calls.setdefault("pushed", []).append((branch, message, tuple(paths)))
        return "sha-" + branch[-7:]
 
    # --- PRs ---
    def find_pr_for_branch(self, branch):
        if branch in self.prs:
            return type("PR", (), {"url": self.prs[branch], "head_sha": "sha-" + branch[-7:],
                                   "number": 1, "branch": branch})()
        return None
 
    def open_pr(self, branch, title, body):
        if existing := self.find_pr_for_branch(branch):
            return existing
        self.calls.setdefault("opened", []).append((branch, title, body))
        self.prs[branch] = f"https://github.com/o/r/pull/{len(self.prs) + 1}"
        return self.find_pr_for_branch(branch)
 
    def close(self):
        pass
 
 
@pytest.fixture
def calls(monkeypatch):
    box = {}
    monkeypatch.setattr(ex, "GitHubClient", lambda: FakeClient(box))
    monkeypatch.setattr(ex, "apply_patch", lambda repo, diff: box.setdefault("applied", []).append(diff))
    class S:
        ollama_model = "qwen2.5-coder:3b"
        github_token = TOKEN
    monkeypatch.setattr(ex, "get_settings", lambda: S())
    return box
 
 
def _state(**over):
    st = {
        "task": "add input validation to sandbox/search.py",
        "plan": PLAN.model_dump(),          # a dict, as the checkpointer stores it
        "diff": DIFF,
        "commit_message": "fix(search): validate empty query",
        "approval_status": "approved",
        "retry_count": 0,
    }
    st.update(over)
    return st
 
 
CFG = {"configurable": {"thread_id": "d35fa34a"}}
 
 
def test_refuses_without_approval(calls):
    for status in ["pending", "rejected", "edit_requested", None]:
        out = ex.execute_node(_state(approval_status=status), CFG)
        assert "refusing to act" in out["error"]
    assert calls.get("created") is None, "a branch was created without approval"
    assert calls.get("pushed") is None
    assert calls.get("opened") is None
 
 
def test_happy_path_opens_one_pr(calls):
    out = ex.execute_node(_state(), CFG)
    assert not out["error"]
    assert out["branch"].startswith("checkpoint/add-input-validation-to-sandbox-search")
    assert out["pr_url"].endswith("/pull/1")
    assert out["ci_status"] == "pending"
    assert len(calls["created"]) == 1
    assert len(calls["opened"]) == 1
    assert calls["pushed"][0][2] == ("sandbox/search.py",)
 
 
def test_plan_is_read_as_a_dict(calls):
    """state['plan'] is a dict since Week 1. The guide's code did plan.steps."""
    out = ex.execute_node(_state(), CFG)
    body = calls["opened"][0][2]
    assert "1. Raise ValueError when the query is blank" in body
    assert "parse_query silently returns" in body
    assert "qwen2.5-coder:3b" in body
    assert not out["error"]
 
 
def test_running_twice_makes_one_branch_and_one_pr(calls):
    """The bug that costs you a duplicate PR for a single approval."""
    first = ex.execute_node(_state(), CFG)
    second = ex.execute_node(_state(), CFG)          # resumed thread, retried approval
    assert first["branch"] == second["branch"]
    assert first["pr_url"] == second["pr_url"]
    assert len(calls["created"]) == 1, "second run created another branch"
    assert len(calls["opened"]) == 1, "second run opened another PR"
    assert len(calls["applied"]) == 1, "second run re-applied the patch"
 
 
def test_a_different_thread_gets_a_different_branch(calls):
    a = ex.execute_node(_state(), {"configurable": {"thread_id": "aaaaaaa"}})
    b = ex.execute_node(_state(), {"configurable": {"thread_id": "bbbbbbb"}})
    assert a["branch"] != b["branch"]
    assert len(calls["opened"]) == 2
 
 
def test_workspace_is_reset_before_applying(calls):
    """The approval pause is unbounded; the workspace may have moved under us."""
    ex.execute_node(_state(), CFG)
    assert calls["ensure"], "execute did not reset the workspace first"
 
 
def test_failures_do_not_leak_the_token(calls, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError(f"push failed: https://x-access-token:{TOKEN}@github.com/o/r.git")
    monkeypatch.setattr(FakeClient, "commit_and_push", boom)
    out = ex.execute_node(_state(), CFG)
    assert TOKEN not in out["error"], f"TOKEN LEAKED: {out['error']}"
    assert "***" in out["error"]