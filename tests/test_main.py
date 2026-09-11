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
