"""The human pause is the project's headline metric; it has to be measured,
survive a restart, and reach LangSmith as metadata on the resumed trace."""
import time

from app import main
from tests.test_main import _until, client  # noqa: F401  (fixture)


def test_the_pause_is_measured_from_the_checkpoint(client, monkeypatch):  # noqa: F811
    c, _ = client
    spawned = []
    real_spawn = main._spawn
    monkeypatch.setattr(main, "_spawn", lambda fn, *a: (spawned.append(a), real_spawn(fn, *a)))
    with c.websocket_connect("/ws?thread_id=hp1") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        gate = _until(ws, "diff_proposed")[-1]
        assert gate["opened_at"] <= time.time()
        time.sleep(1.1)                                   # the human thinking
        ws.send_json({"type": "approval", "decision": "rejected"})
        _until(ws, "execution_result")
    graph, cmd, cfg, _tid = spawned[-1]
    assert cmd.resume["human_pause_s"] >= 1.0
    assert cfg["metadata"]["human_pause_s"] >= 1.0, "on the resumed trace, for LangSmith"
    assert cfg["configurable"]["thread_id"] == "hp1", "still the same thread"
    state = c.get("/threads/hp1").json()["values"]
    assert state["human_pause_s"] >= 1.0


def test_the_pause_survives_a_restart(client, monkeypatch):  # noqa: F811
    """opened_at is read from the checkpointed payload, not from memory."""
    c, _ = client
    with c.websocket_connect("/ws?thread_id=hp2") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        _until(ws, "diff_proposed")
    from app.events import EventBus
    monkeypatch.setattr(main, "bus", EventBus())          # restart: memory is gone
    snap = main.app.state.graph.get_state({"configurable": {"thread_id": "hp2"}})
    time.sleep(0.3)
    assert main.human_pause(snap) >= 0.3
