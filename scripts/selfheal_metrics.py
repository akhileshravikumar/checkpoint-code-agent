"""Self-healing metrics, read from the checkpointer. No LangSmith or GitHub calls.

Every node boundary is a checkpoint with a timestamp, so the SQLite file already
holds the whole run: which attempt each node belonged to, how long it took, what
CI said, and how long the human sat at each gate.

    # one finished thread -> prints a per-attempt table, appends a row to the CSV
    python scripts/selfheal_metrics.py thread <thread_id>

    # every row collected so far -> the numbers for METRICS.md
    python scripts/selfheal_metrics.py summary

Timing notes:
  * A node's duration is the gap between the checkpoint where it was next and the
    one written after it. For await_approval that gap is proposal -> decision.
  * "agent time" excludes every gate, so it is the time the machine was working.
  * A node re-run after a resumable error (Retry) is summed into its attempt.
"""
import argparse
import csv
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CSV_DEFAULT = Path("metrics/selfheal.csv")
FIELDS = [
    "thread_id", "task", "outcome", "attempts", "healed", "attempts_to_green",
    "ci_failures", "failure_to_green_s", "agent_time_s", "wall_time_s",
    "gates", "human_pause_total_s", "pr_url",
]


def _ts(snap) -> float:
    return datetime.fromisoformat(snap.created_at).timestamp()


def load_graph(db: str | None):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from app.graph import build_graph, make_checkpointer

    if db:
        saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
    else:
        saver = make_checkpointer()
    return build_graph(saver)


def spans(graph, thread_id: str) -> list[dict]:
    """One entry per completed node execution, oldest first."""
    config = {"configurable": {"thread_id": thread_id}}
    hist = list(graph.get_state_history(config))[::-1]
    if not hist:
        raise SystemExit(f"no checkpoints for thread {thread_id}")
    out = []
    for before, after in zip(hist, hist[1:]):
        for node in before.next:
            if node.startswith("__"):
                continue
            out.append({
                "node": node,
                "attempt": (before.values.get("retry_count") or 0) + 1,
                "start": _ts(before),
                "end": _ts(after),
                "s": round(_ts(after) - _ts(before), 1),
                "after": after.values,
            })
    return out


def analyse(graph, thread_id: str) -> tuple[dict, list[dict]]:
    sp = spans(graph, thread_id)
    final = graph.get_state({"configurable": {"thread_id": thread_id}})
    fv = final.values

    per = defaultdict(lambda: defaultdict(float))
    info = defaultdict(dict)
    for x in sp:
        a, n, v = x["attempt"], x["node"], x["after"]
        per[a][n] += x["s"]
        if n == "propose_diff":
            info[a]["rewrite_attempt"] = v.get("rewrite_attempt")
        elif n == "await_approval":
            info[a]["decision"] = v.get("approval_status")
            info[a]["human_pause_s"] = v.get("human_pause_s")
        elif n == "watch_ci":
            info[a]["ci"] = v.get("ci_status")
            info[a]["ci_end"] = x["end"]
            info[a]["ci_run_url"] = v.get("ci_run_url")

    attempts = []
    for a in sorted(per):
        attempts.append({
            "attempt": a,
            "plan_s": per[a].get("plan", 0),
            "propose_diff_s": per[a].get("propose_diff", 0),
            "rewrite_attempt": info[a].get("rewrite_attempt"),
            "gate_s": per[a].get("await_approval", 0),
            "human_pause_s": info[a].get("human_pause_s"),
            "decision": info[a].get("decision"),
            "approve_to_pr_s": per[a].get("execute", 0),
            "ci_round_trip_s": per[a].get("watch_ci", 0),
            "ci": info[a].get("ci"),
            "ci_run_url": info[a].get("ci_run_url"),
        })

    ci = [(a["attempt"], info[a["attempt"]].get("ci"), info[a["attempt"]].get("ci_end"))
          for a in attempts if info[a["attempt"]].get("ci")]
    failures = [c for c in ci if c[1] in {"failed", "timeout"}]
    green = next((c for c in ci if c[1] == "passed"), None)

    if final.next:
        outcome = "in_progress"
    elif fv.get("no_change_reason"):
        outcome = "no_change"
    elif green:
        outcome = "passed"
    elif "give_up" in {x["node"] for x in sp}:
        outcome = "gave_up"
    elif fv.get("approval_status") == "rejected":
        outcome = "rejected"
    elif fv.get("error"):
        outcome = "error"
    else:
        outcome = "ended"

    gate_total = sum(x["s"] for x in sp if x["node"] == "await_approval")
    wall = round(sp[-1]["end"] - sp[0]["start"], 1)
    row = {
        "thread_id": thread_id,
        "task": fv.get("task", ""),
        "outcome": outcome,
        "attempts": len(attempts),
        "healed": bool(failures and green),
        "attempts_to_green": green[0] if green else "",
        "ci_failures": len(failures),
        "failure_to_green_s": round(green[2] - failures[0][2], 1) if failures and green else "",
        "agent_time_s": round(wall - gate_total, 1),
        "wall_time_s": wall,
        "gates": sum(1 for x in sp if x["node"] == "await_approval"),
        "human_pause_total_s": round(sum(a["human_pause_s"] or 0 for a in attempts), 1),
        "pr_url": fv.get("pr_url") or "",
    }
    return row, attempts


def cmd_thread(args) -> None:
    graph = load_graph(args.db)
    row, attempts = analyse(graph, args.thread_id)

    cols = ["attempt", "plan_s", "propose_diff_s", "rewrite_attempt", "gate_s",
            "human_pause_s", "decision", "approve_to_pr_s", "ci_round_trip_s", "ci"]
    print(" | ".join(cols))
    for a in attempts:
        print(" | ".join("" if a[c] is None else str(a[c]) for c in cols))
    print()
    for k in FIELDS:
        print(f"{k:>20}: {row[k]}")
    for a in attempts:
        if a["ci_run_url"]:
            print(f"{'ci run ' + str(a['attempt']):>20}: {a['ci_run_url']}")

    if args.no_save or row["outcome"] == "in_progress":
        if row["outcome"] == "in_progress":
            print("\nthread still in progress; not saved")
        return
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open() as f:
            existing = [r for r in csv.DictReader(f) if r["thread_id"] != args.thread_id]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing + [row])
    print(f"\nsaved to {path} ({len(existing) + 1} threads)")


def _num(rows, key):
    return [float(r[key]) for r in rows if r[key] not in ("", None)]


def _med(xs):
    return f"{statistics.median(xs):.0f}s" if xs else "n/a"


def cmd_summary(args) -> None:
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"{path} not found; run `thread <id>` first")
    rows = list(csv.DictReader(path.open()))
    failed = [r for r in rows if int(r["ci_failures"]) > 0]
    healed = [r for r in failed if r["healed"] == "True"]
    gave_up = [r for r in failed if r["outcome"] == "gave_up"]
    first_try = [r for r in rows if r["attempts_to_green"] == "1"]

    print(f"threads: {len(rows)}  (with a CI failure: {len(failed)})\n")
    print("| Metric | Value |\n|---|---|")
    if failed:
        print(f"| Self-heal success rate | {len(healed)}/{len(failed)} "
              f"({100 * len(healed) / len(failed):.0f}%) of CI failures ended green |")
        print(f"| Gave up after retry cap | {len(gave_up)}/{len(failed)} |")
        att = _num(healed, "attempts_to_green")
        if att:
            print(f"| Attempts to green (healed) | median {statistics.median(att):.0f}, "
                  f"max {max(att):.0f} |")
        print(f"| CI failure → green, wall clock | median {_med(_num(healed, 'failure_to_green_s'))} |")
    print(f"| Green on first attempt | {len(first_try)}/{len(rows)} |")
    print(f"| Agent time per thread (excl. human) | median {_med(_num(rows, 'agent_time_s'))} |")
    print(f"| Human pause per thread | median {_med(_num(rows, 'human_pause_total_s'))} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", help="checkpoint SQLite file (default: settings.checkpoint_db)")
    p.add_argument("--csv", default=str(CSV_DEFAULT))
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thread")
    t.add_argument("thread_id")
    t.add_argument("--no-save", action="store_true")
    t.set_defaults(fn=cmd_thread)
    s = sub.add_parser("summary")
    s.set_defaults(fn=cmd_summary)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()