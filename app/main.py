"""FastAPI app: WebSocket endpoint plus the static dashboard."""
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langgraph.types import Command

from app.config import get_settings
from app.events import bus
from app.github_client import GitHubClient, _redact
from app.graph import build_graph, make_checkpointer
from app.state import new_task
from app.tracing import configure_tracing

DASHBOARD = Path(__file__).parent.parent / "dashboard" / "index.html"


def _check_ollama() -> None:
    """Fail at startup with a useful message, not 60s into the first inference.

    Runs in offline mode too: CHECKPOINT_OFFLINE removes GitHub and LangSmith,
    not the model.
    """
    url = get_settings().ollama_base_url
    try:
        httpx.get(f"{url}/api/version", timeout=3.0).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama unreachable at {url}. "
            f"Start it with `sudo systemctl start ollama`. ({exc})"
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_ollama()
    configure_tracing()
    app.state.graph = build_graph(make_checkpointer())
    s = get_settings()
    app.state.workspace = str(s.workspace_dir.expanduser().resolve())
    if not s.checkpoint_offline:
        try:
            gh = GitHubClient()
            try:
                app.state.workspace = str(await asyncio.to_thread(gh.ensure_workspace))
            finally:
                gh.close()
        except Exception as exc:
            print(f"[startup] workspace not refreshed: {_redact(str(exc), s.github_token)}")
    yield


app = FastAPI(title="Checkpoint", lifespan=lifespan)

# asyncio keeps only a weak reference to tasks; hold them until they finish.
_background: set[asyncio.Task] = set()


def _spawn(fn, *args) -> None:
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    _background.add(task)
    task.add_done_callback(_background.discard)


def _error(message: str, *, resumable: bool = False) -> dict:
    """Every error string reaches the browser: redact the PAT first."""
    msg = {"type": "error", "message": _redact(message, get_settings().github_token)}
    if resumable:
        msg["resumable"] = True
    return msg


@app.get("/")
async def index():
    # no-store: without it the browser may keep serving a cached dashboard
    # after the code changes, so new message types (e.g. no_change) arrive over
    # the socket and are silently ignored until a hard reload.
    return FileResponse(DASHBOARD, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return {"ok": True, "model": get_settings().ollama_model}


@app.get("/threads/{thread_id}")
def thread_state(thread_id: str):
    """Rehydrate a session from the checkpointer, not from memory.

    Plain `def`: FastAPI runs it in its threadpool, so the SQLite read does
    not block the event loop.
    """
    snap = app.state.graph.get_state({"configurable": {"thread_id": thread_id}})
    if snap.created_at is None:
        return {"found": False}
    return {
        "found": True,
        "next": list(snap.next),
        "awaiting_approval": bool(snap.interrupts),
        "interrupt": snap.interrupts[0].value if snap.interrupts else None,
        "values": {
            k: v for k, v in snap.values.items()
            if k in {"task", "approval_status", "branch", "pr_url",
                     "ci_status", "retry_count", "error", "no_change_reason"}
        },
    }


@app.get("/threads")
def list_threads(limit: int = 20):
    seen, out = set(), []
    for st in app.state.graph.checkpointer.list(None, limit=limit * 5):
        tid = st.config["configurable"]["thread_id"]
        if tid in seen:
            continue
        seen.add(tid)
        out.append({"thread_id": tid, "task": st.checkpoint["channel_values"].get("task")})
        if len(out) >= limit:
            break
    return out


def _refresh_workspace() -> str:
    """Pull origin/main into the workspace before planning against it."""
    s = get_settings()
    if s.checkpoint_offline:
        return str(s.workspace_dir.expanduser().resolve())
    gh = GitHubClient()
    try:
        return str(gh.ensure_workspace())
    finally:
        gh.close()


def _run_until_pause(graph, payload, config, thread_id: str) -> None:
    """Run the graph in a worker thread; publish whatever it stops on.

    graph.invoke is blocking and CPU inference holds it for a while, so it must
    not run on the event loop or the WebSocket would stop responding.

    An exception here would otherwise vanish inside the background task and
    leave the dashboard on "planning..." forever. The thread is left parked
    mid-node with its input checkpointed, so the error is resumable: `resume`
    re-runs that node (ADR-005: invoke(None)).
    """
    try:
        result = graph.invoke(payload, config=config)
    except Exception as exc:
        bus.publish(thread_id, _error(f"{type(exc).__name__}: {exc}", resumable=True))
        return
    if interrupts := result.get("__interrupt__"):
        bus.publish(thread_id, interrupts[0].value)
    elif err := result.get("error"):
        bus.publish(thread_id, _error(err))
    elif reason := result.get("no_change_reason"):
        # Not an error: the model looked and found nothing to do. Show the plan
        # so the user can see what it thought the task meant.
        bus.publish(thread_id, {
            "type": "no_change", "message": reason, "plan": result.get("plan"),
        })
    else:
        bus.publish(thread_id, {
            "type": "execution_result",
            "approval_status": result.get("approval_status"),
            "pr_url": result.get("pr_url"),
            "ci_status": result.get("ci_status"),
            "ci_run_url": result.get("ci_run_url"),
        })


def _stopped_mid_node(snap) -> str:
    return (f"This thread stopped inside {', '.join(snap.next)}. "
            "Retry that step, or open a new session.")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    thread_id = websocket.query_params.get("thread_id") or uuid.uuid4().hex[:8]
    queue = bus.subscribe(thread_id)
    graph = websocket.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    await websocket.send_json({"type": "session", "thread_id": thread_id})
    for past in bus.history(thread_id):      # replay so a reconnect isn't blank
        await websocket.send_json(past)

    # After a server restart the in-process bus history is empty, so rebuild
    # the view from the checkpointer — the durable source of truth.
    snap = graph.get_state(config)
    if not bus.history(thread_id):
        if snap.interrupts:                  # paused at the gate: show it again
            await websocket.send_json(snap.interrupts[0].value)
        elif snap.next:                      # crashed mid-node: offer a retry
            await websocket.send_json(_error(_stopped_mid_node(snap), resumable=True))

    async def pump():
        while True:
            await websocket.send_json(await queue.get())

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = await websocket.receive_json()
            snap = graph.get_state(config)

            if msg["type"] == "start":
                # ADR-005: an input dict restarts a thread; never do that to a
                # thread that is paused at the gate or parked mid-node.
                if snap.interrupts:
                    bus.publish(thread_id, _error(
                        "This thread is awaiting approval. Decide on the current "
                        "diff, or open a new session."))
                    continue
                if snap.next:
                    bus.publish(thread_id, _error(_stopped_mid_node(snap), resumable=True))
                    continue
                bus.publish(thread_id, {"type": "status", "message": "refreshing workspace..."})
                try:
                    workspace = await asyncio.to_thread(_refresh_workspace)
                except Exception as exc:        # an expired PAT lands here
                    bus.publish(thread_id, _error(f"could not refresh the workspace: {exc}"))
                    continue
                websocket.app.state.workspace = workspace
                bus.publish(thread_id, {"type": "status", "message": "planning..."})
                payload = new_task(msg["task"], msg.get("repo_path") or workspace)
                _spawn(_run_until_pause, graph, payload, config, thread_id)

            elif msg["type"] == "resume":
                # Only for a thread parked mid-node. At the gate, None would just
                # re-raise the interrupt; on a finished thread it does nothing.
                if not snap.next or snap.interrupts:
                    bus.publish(thread_id, _error("Nothing to retry on this thread."))
                    continue
                bus.publish(thread_id, {"type": "status",
                                        "message": f"retrying {', '.join(snap.next)}..."})
                _spawn(_run_until_pause, graph, None, config, thread_id)

            elif msg["type"] == "approval":
                # A second click, or a stale tab, must not resume a thread that is
                # no longer at the gate.
                if not snap.interrupts:
                    bus.publish(thread_id, _error("Nothing is awaiting approval on this thread."))
                    continue
                bus.publish(thread_id, {"type": "status", "message": f"{msg['decision']}..."})
                cmd = Command(resume={
                    "decision": msg["decision"], "note": msg.get("note", "")
                })
                _spawn(_run_until_pause, graph, cmd, config, thread_id)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        bus.unsubscribe(thread_id, queue)
