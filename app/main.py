"""FastAPI app: WebSocket endpoint plus the static dashboard."""
import asyncio
import subprocess
import sys
import time
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langgraph.types import Command

from app.config import get_settings
from app.events import bus
from app.github_client import GitHubClient, _redact
from app.graph import build_graph, make_checkpointer
from app.nodes.plan import list_candidates
from app.state import new_task
from app.tracing import configure_tracing

DASHBOARD = Path(__file__).parent.parent / "dashboard" / "index.html"


def reload_watch_warning(argv: list[str], cwd: Path, workspace: Path) -> str | None:
    """`--reload` plus a workspace inside the watched tree restarts mid-run.

    The agent's whole job is writing .py files into .workspace/. uvicorn's
    reloader includes `*.py` and excludes only names starting with a dot, which
    a dotted *directory* does not satisfy (`Path(".workspace/x.py").match(".*")`
    is False). So `execute` applying its own patch restarts the server, the
    dashboard socket closes mid-run and the thread is left parked in a node —
    durable, resumable, and baffling if you don't know why it happened.
    """
    if not any(a == "--reload" or a.startswith("--reload=") for a in argv):
        return None
    try:
        rel = workspace.expanduser().resolve().relative_to(cwd.resolve())
    except ValueError:
        return None                      # workspace lives outside the watched tree
    return (
        f"--reload is watching {rel}/, which is where the agent writes the code it "
        f"patches. Approving a diff will restart the server and drop the dashboard "
        f"socket mid-run. Start it without --reload, or add: "
        f"--reload-exclude '{rel}/*'"
    )


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
    if warning := reload_watch_warning(sys.argv, Path.cwd(), s.workspace_dir):
        print(f"[startup] WARNING: {warning}")
    if not s.checkpoint_offline:
        try:
            gh = GitHubClient()
            try:
                # Say who GitHub will see: a PAT would carry your admin bypass.
                print(f"[startup] GitHub identity: {gh.identity}")
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


def _error(message: str, *, resumable: bool = False, refused: bool = False) -> dict:
    """Every error string reaches the browser: redact the PAT first.

    resumable: a node crashed and can be retried.
    refused:   a request was turned down (wrong moment); nothing ran or changed,
               so the dashboard must not treat it as a failed step.
    """
    msg = {"type": "error", "message": _redact(message, get_settings().github_token)}
    if resumable:
        msg["resumable"] = True
    if refused:
        msg["refused"] = True
    return msg


# Threads with a graph run in flight. While a node is running, get_state()
# shows it in `next` exactly as it would after a crash, so "busy" and "stopped"
# can only be told apart by tracking runs here.
_active: set[str] = set()
_active_lock = threading.Lock()


def _begin(thread_id: str) -> bool:
    with _active_lock:
        if thread_id in _active:
            return False
        _active.add(thread_id)
        return True


def _end(thread_id: str) -> None:
    with _active_lock:
        _active.discard(thread_id)


_BUSY = "A run is already in progress on this thread. Wait for it to finish."


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
                     "ci_status", "retry_count", "error", "no_change_reason",
                     "human_pause_s", "rewrite_attempt"}
        },
    }

def _current_branch(repo: Path) -> str | None:
    """The workspace's branch, for the file panel. None if it isn't a git repo.

    symbolic-ref names the branch even before its first commit; a detached
    HEAD falls back to the short sha.
    """
    for args in (["symbolic-ref", "--short", "HEAD"], ["rev-parse", "--short", "HEAD"]):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


@app.get("/files")
def files():
    """What the agent may edit: the same filter the planner uses."""
    repo = Path(app.state.workspace)
    return {
        "branch": _current_branch(repo),
        "editable": [str(p.relative_to(repo)) for p in list_candidates(repo)],
        "protected": ["tests/", "demo/", ".github/"],
    }


@app.post("/workspace/refresh")
def refresh_workspace():
    """Pull origin/main after a human merge (the agent never merges).

    Refused while any run is in flight: it hard-resets the workspace the graph
    is reading and writing.
    """
    if _active:
        raise HTTPException(409, "A run is in progress; pull main once it has finished.")
    try:
        app.state.workspace = _refresh_workspace()
    except Exception as exc:
        raise HTTPException(502, _redact(f"could not refresh the workspace: {exc}")) from None
    return files()


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
    """Run the graph in a worker thread; publish progress and whatever it stops on.

    graph.stream (not invoke) so the dashboard sees each node start and finish
    while the run is in progress. `tasks` mode gives a start event (has "input")
    and a finish event per node; the gate's finish event carries the interrupt.
    `custom` carries what nodes write with get_stream_writer() (token progress).

    An exception leaves the thread parked mid-node with its input checkpointed,
    so the error is resumable: `resume` re-runs that node (ADR-005: invoke(None)).
    """
    try:
        for mode, ev in graph.stream(payload, config=config, stream_mode=["tasks", "custom"]):
            if mode == "tasks":
                if "input" in ev:
                    kind = "node_enter"
                elif ev.get("interrupts"):
                    kind = "node_paused"          # await_approval: the gate is open
                else:
                    kind = "node_exit"
                msg = {"type": kind, "node": ev["name"]}
                # So the PR button appears as soon as execute finishes, not only
                # after CI. Only these two fields: node results can hold errors.
                result = ev.get("result")
                if kind == "node_exit" and isinstance(result, dict):
                    msg.update({k: result[k] for k in ("pr_url", "branch") if result.get(k)})
                bus.publish(thread_id, msg)
            else:
                bus.publish(thread_id, ev)        # e.g. token_progress
    except Exception as exc:
        bus.publish(thread_id, _error(f"{type(exc).__name__}: {exc}", resumable=True))
        return
    finally:
        _end(thread_id)

    snap = graph.get_state(config)
    values = snap.values
    if snap.interrupts:
        bus.publish(thread_id, snap.interrupts[0].value)
    elif err := values.get("error"):
        bus.publish(thread_id, _error(err))
    elif reason := values.get("no_change_reason"):
        bus.publish(thread_id, {"type": "no_change", "message": reason, "plan": values.get("plan")})
    else:
        bus.publish(thread_id, {
            "type": "execution_result",
            "approval_status": values.get("approval_status"),
            "pr_url": values.get("pr_url"),
            "ci_status": values.get("ci_status"),
            "ci_run_url": values.get("ci_run_url"),
        })


def human_pause(snap) -> float | None:
    """Seconds since the pending gate opened, from its checkpointed payload.

    Survives a server restart: opened_at lives in the checkpoint, not in memory.
    """
    opened = snap.interrupts[0].value.get("opened_at") if snap.interrupts else None
    return round(time.time() - opened, 1) if opened else None


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
                # thread that is running, paused at the gate or parked mid-node.
                if thread_id in _active:
                    bus.publish(thread_id, _error(_BUSY, refused=True))
                    continue
                if snap.interrupts:
                    bus.publish(thread_id, _error(
                        "This thread is awaiting approval. Decide on the current "
                        "diff, or open a new session.", refused=True))
                    continue
                if snap.next:
                    bus.publish(thread_id, _error(_stopped_mid_node(snap), resumable=True))
                    continue
                if not _begin(thread_id):
                    bus.publish(thread_id, _error(_BUSY, refused=True))
                    continue
                bus.publish(thread_id, {"type": "status", "message": "refreshing workspace..."})
                try:
                    workspace = await asyncio.to_thread(_refresh_workspace)
                except Exception as exc:        # an expired PAT lands here
                    _end(thread_id)
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
                    bus.publish(thread_id, _error("Nothing to retry on this thread.",
                                                  refused=True))
                    continue
                if not _begin(thread_id):
                    bus.publish(thread_id, _error(_BUSY, refused=True))
                    continue
                bus.publish(thread_id, {"type": "status",
                                        "message": f"retrying {', '.join(snap.next)}..."})
                _spawn(_run_until_pause, graph, None, config, thread_id)

            elif msg["type"] == "approval":
                # A second click, or a stale tab, must not resume a thread that is
                # no longer at the gate.
                if not snap.interrupts:
                    bus.publish(thread_id, _error("Nothing is awaiting approval on this thread.",
                                                  refused=True))
                    continue
                if not _begin(thread_id):
                    bus.publish(thread_id, _error(_BUSY, refused=True))
                    continue
                bus.publish(thread_id, {"type": "status", "message": f"{msg['decision']}..."})
                pause = human_pause(snap)
                cmd = Command(resume={
                    "decision": msg["decision"], "note": msg.get("note", ""),
                    "human_pause_s": pause,
                })
                # Each resume is its own LangSmith trace (grouped with the rest of
                # the thread by its thread_id metadata). The pause sits between
                # traces, not inside a span, so it is attached here explicitly.
                traced = {**config, "metadata": {"human_pause_s": pause,
                                                 "decision": msg["decision"]}}
                _spawn(_run_until_pause, graph, cmd, traced, thread_id)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        bus.unsubscribe(thread_id, queue)
