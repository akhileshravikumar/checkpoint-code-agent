"""watch_ci node: poll the Actions run triggered by our own PR.

Blocking by design — it runs in a worker thread (app/main.py) and publishes
progress to the dashboard as it goes.
"""
import time

from langchain_core.runnables import RunnableConfig

from app.config import get_settings
from app.events import bus
from app.github_client import GitHubClient, _redact
from app.state import AgentState

TERMINAL = {"success", "failure", "cancelled", "timed_out", "action_required", "stale"}


def watch_ci_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    s = get_settings()
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    sha = state.get("head_sha")
    if not sha:
        # new_task() clears head_sha, so an empty one here means execute did not
        # run — never poll a stale sha from a previous task.
        return {"ci_status": None, "error": "watch_ci reached with no head_sha."}

    client = GitHubClient()
    deadline = time.time() + s.ci_poll_timeout
    try:
        while time.time() < deadline:
            run = client.latest_run_for_sha(sha)
            if run is None:
                bus.publish(thread_id, {"type": "ci_status", "status": "queued",
                                        "message": "waiting for Actions to pick up the push"})
                time.sleep(s.ci_poll_interval)
                continue

            status, conclusion = run["status"], run.get("conclusion")
            bus.publish(thread_id, {
                "type": "ci_status",
                "status": conclusion or status,
                "run_url": run["html_url"],
            })

            if status == "completed" and conclusion in TERMINAL:
                if conclusion == "success":
                    return {"ci_status": "passed", "ci_run_url": run["html_url"],
                            "ci_failure_log": ""}
                log = client.failure_log(run["id"])
                return {"ci_status": "failed", "ci_run_url": run["html_url"],
                        "ci_failure_log": log}

            time.sleep(s.ci_poll_interval)

        return {"ci_status": "timeout",
                "error": f"CI did not finish within {s.ci_poll_timeout}s"}
    except Exception as exc:
        return {"ci_status": None,
                "error": _redact(f"watch_ci failed: {exc}", s.github_token)}
    finally:
        client.close()