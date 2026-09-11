"""The WebSocket and REST layer, driven end to end with the graph stubbed.

The graph tests never go through main.py, so a broken `start` branch (it once
refreshed the workspace and then never ran the graph) passed the whole suite.
These use TestClient without entering it, so the lifespan — Ollama check,
GitHub clone — does not run; the graph is installed by hand.
"""
import queue
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app import graph as graph_mod
from app import main
from app.events import EventBus
from tests.test_execute_node import DIFF, PLAN

TOKEN = "github_pat_SECRET_IN_AN_ERROR"


@pytest.fixture
def client(monkeypatch, tmp_path):
    knobs = {"plan_fails": 0}

    def plan(state):
        if knobs["plan_fails"] > 0:
            knobs["plan_fails"] -= 1
            raise ConnectionError(f"ollama down (token {TOKEN})")
        return {"plan": PLAN.model_dump(), "error": ""}

    monkeypatch.setattr(graph_mod, "plan_node", plan)
    monkeypatch.setattr(graph_mod, "propose_diff_node", lambda state: {
        "diff": DIFF, "commit_message": "fix(search): x",
        "approval_status": "pending", "error": ""})
    monkeypatch.setattr(main, "_refresh_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(main, "bus", EventBus())

    class S:
        github_token = TOKEN
        ollama_model = "qwen2.5-coder:3b"
    monkeypatch.setattr(main, "get_settings", lambda: S())

    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
    saver.setup()
    main.app.state.graph = graph_mod.build_graph(saver)
    main.app.state.workspace = str(tmp_path)
    return TestClient(main.app), knobs


def _until(ws, want: str, limit: int = 20, timeout: float = 10) -> list[dict]:
    """Read messages until one of type `want`.

    Fails after `timeout` seconds instead of hanging: a daemon thread does the
    blocking read, so a server that never answers cannot wedge the suite.
    """
    seen = []
    for _ in range(limit):
        box: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: box.put(ws.receive_json()), daemon=True).start()
        try:
            m = box.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"timed out waiting for {want!r}; got {seen}") from None
        seen.append(m)
        if m["type"] == want:
            return seen
    raise AssertionError(f"no {want!r} in {seen}")


def test_start_runs_the_graph_to_the_gate(client):
    c, _ = client
    with c.websocket_connect("/ws?thread_id=m1") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        seen = _until(ws, "diff_proposed")
    assert {"type": "status", "message": "planning..."} in seen


def test_a_crash_mid_node_is_reported_and_retry_resumes_it(client):
    c, knobs = client
    knobs["plan_fails"] = 1
    with c.websocket_connect("/ws?thread_id=m2") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        err = _until(ws, "error")[-1]
        assert err["resumable"] is True
        assert "ConnectionError" in err["message"]
        assert TOKEN not in err["message"], "the PAT reached the browser"

        ws.send_json({"type": "start", "task": "another task"})      # must not restart
        refused = _until(ws, "error")[-1]
        assert "stopped inside plan" in refused["message"]

        ws.send_json({"type": "resume"})
        _until(ws, "diff_proposed")


def test_reconnecting_after_a_restart_offers_the_retry(client, monkeypatch):
    c, knobs = client
    knobs["plan_fails"] = 1
    with c.websocket_connect("/ws?thread_id=m3") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "error")
    monkeypatch.setattr(main, "bus", EventBus())       # a restart empties the bus
    with c.websocket_connect("/ws?thread_id=m3") as ws:
        seen = _until(ws, "error")
        assert seen[-1]["resumable"] is True


def test_reconnecting_at_the_gate_shows_the_diff_again(client, monkeypatch):
    c, _ = client
    with c.websocket_connect("/ws?thread_id=m4") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
    monkeypatch.setattr(main, "bus", EventBus())
    with c.websocket_connect("/ws?thread_id=m4") as ws:
        _until(ws, "diff_proposed", limit=3)


def test_guards_refuse_actions_that_do_not_fit_the_thread(client):
    c, _ = client
    with c.websocket_connect("/ws?thread_id=m5") as ws:
        ws.send_json({"type": "approval", "decision": "approved"})
        assert "Nothing is awaiting approval" in _until(ws, "error")[-1]["message"]
        ws.send_json({"type": "resume"})
        assert "Nothing to retry" in _until(ws, "error")[-1]["message"]
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
        ws.send_json({"type": "start", "task": "again"})
        assert "awaiting approval" in _until(ws, "error")[-1]["message"]


def test_thread_endpoints_read_the_checkpointer(client):
    c, _ = client
    assert c.get("/threads/nope").json() == {"found": False}
    with c.websocket_connect("/ws?thread_id=m6") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
    s = c.get("/threads/m6").json()
    assert s["found"] and s["awaiting_approval"]
    assert s["next"] == ["await_approval"]
    assert s["values"]["task"] == "fix sandbox/search.py"
    assert any(t["thread_id"] == "m6" for t in c.get("/threads").json())


def test_startup_fails_fast_when_ollama_is_down(monkeypatch):
    class S:
        ollama_base_url = "http://127.0.0.1:9"      # nothing listens on the discard port
    monkeypatch.setattr(main, "get_settings", lambda: S())
    with pytest.raises(RuntimeError, match="Ollama unreachable"):
        main._check_ollama()


def test_no_change_is_published_as_its_own_message(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(graph_mod, "propose_diff_node", lambda state: {
        "diff": "", "no_change_reason": "No change proposed for sandbox/search.py.",
        "error": ""})
    main.app.state.graph = graph_mod.build_graph(main.app.state.graph.checkpointer)
    with c.websocket_connect("/ws?thread_id=m7") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        m = _until(ws, "no_change")[-1]
    assert m["message"].startswith("No change proposed")
    assert m["plan"]["target_file"] == "sandbox/search.py"
    state = c.get("/threads/m7").json()
    assert state["next"] == [], "the thread is finished, not parked"
    assert state["values"]["no_change_reason"].startswith("No change proposed")


def test_dashboard_is_never_served_from_cache(client):
    """A cached dashboard ignores message types added since it was cached."""
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    assert "case 'no_change'" in r.text


def test_files_lists_what_the_agent_may_edit(client, tmp_path):
    """/files once raised NameError (list_candidates was never imported)."""
    c, _ = client
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox" / "search.py").write_text("x = 1\n")
    (tmp_path / "sandbox" / "twosum.py").write_text("y = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text("")
    r = c.get("/files")
    assert r.status_code == 200
    assert r.json()["editable"] == ["sandbox/search.py", "sandbox/twosum.py"]
    assert "tests/" in r.json()["protected"]


def _with_github_stubbed(monkeypatch):
    monkeypatch.setattr(graph_mod, "execute_node", lambda state, config=None: {
        "branch": "checkpoint/x-1", "pr_url": "https://github.com/o/r/pull/9",
        "head_sha": "abc", "ci_status": "pending", "error": ""})
    monkeypatch.setattr(graph_mod, "watch_ci_node", lambda state, config=None: {
        "ci_status": "passed", "ci_run_url": "https://github.com/o/r/actions/runs/1",
        "ci_failure_log": ""})
    main.app.state.graph = graph_mod.build_graph(main.app.state.graph.checkpointer)


def test_the_pr_link_arrives_when_execute_finishes(client, monkeypatch):
    c, _ = client
    _with_github_stubbed(monkeypatch)
    with c.websocket_connect("/ws?thread_id=p1") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
        ws.send_json({"type": "approval", "decision": "approved"})
        seen = _until(ws, "execution_result")
    done = next(m for m in seen if m["type"] == "node_exit" and m["node"] == "execute")
    assert done["pr_url"] == "https://github.com/o/r/pull/9"
    assert done["branch"] == "checkpoint/x-1"
    assert seen.index(done) < [m["type"] for m in seen].index("node_enter", seen.index(done)), \
        "before watch_ci starts"
    assert seen[-1]["ci_status"] == "passed"


def test_a_busy_thread_refuses_instead_of_claiming_a_crash(client):
    c, _ = client
    main._active.add("b1")
    try:
        with c.websocket_connect("/ws?thread_id=b1") as ws:
            for msg in ({"type": "start", "task": "x"}, {"type": "resume"}):
                ws.send_json(msg)
                err = _until(ws, "error")[-1]
                assert err.get("refused") is True
                assert "resumable" not in err
                assert "in progress" in err["message"] or "Nothing to retry" in err["message"]
    finally:
        main._active.discard("b1")


def test_guard_errors_are_marked_refused(client):
    c, _ = client
    with c.websocket_connect("/ws?thread_id=b2") as ws:
        ws.send_json({"type": "approval", "decision": "approved"})
        assert _until(ws, "error")[-1]["refused"] is True


def test_a_finished_run_is_no_longer_active(client):
    c, _ = client
    with c.websocket_connect("/ws?thread_id=b3") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
    assert "b3" not in main._active, "paused at the gate is not running"


def test_pull_main_is_refused_while_a_run_is_active(client):
    c, _ = client
    main._active.add("r1")
    try:
        r = c.post("/workspace/refresh")
        assert r.status_code == 409
    finally:
        main._active.discard("r1")


def test_pull_main_returns_the_file_list(client, tmp_path):
    c, _ = client
    (tmp_path / "app.py").write_text("x = 1\n")
    r = c.post("/workspace/refresh")
    assert r.status_code == 200
    assert "app.py" in r.json()["editable"]
    assert "branch" in r.json()


def test_pull_main_failures_are_redacted(client, monkeypatch):
    c, _ = client
    def boom():
        raise RuntimeError(f"fetch failed for https://x-access-token:{TOKEN}@github.com/o/r")
    monkeypatch.setattr(main, "_refresh_workspace", boom)
    r = c.post("/workspace/refresh")
    assert r.status_code == 502
    assert TOKEN not in r.text


def test_files_names_the_workspace_branch(client, tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    assert client[0].get("/files").json()["branch"] == "main", "even before the first commit"
