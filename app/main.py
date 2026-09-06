"""FastAPI app: WebSocket endpoint plus the static dashboard."""
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langgraph.types import Command

from app.config import get_settings
from app.events import bus
from app.github_client import GitHubClient, _redact
from app.graph import build_graph, make_checkpointer
from app.tracing import configure_tracing
from app.state import new_task

DASHBOARD = Path(__file__).parent.parent / "dashboard" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/")
async def index():
    return FileResponse(DASHBOARD)


@app.get("/health")
async def health():
    return {"ok": True, "model": get_settings().ollama_model}

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
    """
    result = graph.invoke(payload, config=config)
    if interrupts := result.get("__interrupt__"):
        bus.publish(thread_id, interrupts[0].value)
    elif err := result.get("error"):
        bus.publish(thread_id, {"type": "error", "message": err})
    else:
        bus.publish(thread_id, {
            "type": "execution_result",
            "approval_status": result.get("approval_status"),
            "pr_url": result.get("pr_url"),
        })


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

    # Reconnecting to a paused thread: re-show the gate. The EventBus is
    # in-process, so its history is empty after a server restart — the
    # checkpointer is the durable source of truth, not the bus.
    snap = graph.get_state(config)
    if snap.interrupts and not bus.history(thread_id):
        await websocket.send_json(snap.interrupts[0].value)

    async def pump():
        while True:
            await websocket.send_json(await queue.get())

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = await websocket.receive_json()
            if msg["type"] == "start":
                # ADR-005: an input dict restarts a paused thread. Refuse.
                if graph.get_state(config).next:
                    bus.publish(thread_id, {
                        "type": "error",
                        "message": "This thread is already awaiting approval. "
                                   "Decide on the current diff, or open a new session.",
                    })
                    continue
                bus.publish(thread_id, {"type": "status", "message": "refreshing workspace..."})
                workspace = await asyncio.to_thread(_refresh_workspace)
                bus.publish(thread_id, {"type": "status", "message": "planning..."})
                payload = new_task(msg["task"], msg.get("repo_path") or workspace)
                
                asyncio.create_task(asyncio.to_thread(
                    _run_until_pause, graph, payload, config, thread_id
                ))
            elif msg["type"] == "approval":
                bus.publish(thread_id, {"type": "status", "message": f"{msg['decision']}..."})
                cmd = Command(resume={
                    "decision": msg["decision"], "note": msg.get("note", "")
                })
                asyncio.create_task(asyncio.to_thread(
                    _run_until_pause, graph, cmd, config, thread_id
                ))
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        bus.unsubscribe(thread_id, queue)