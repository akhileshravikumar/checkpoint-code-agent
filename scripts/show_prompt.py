"""Print the exact prompts a thread's next plan/rewrite would send. No LLM call.

When a retry keeps proposing the wrong fix there are two possible culprits, and
guessing between them costs a full CI round trip each time:

  1. the prompt never carried the information (a plumbing bug), or
  2. it did, and the model ignored it (a model limit).

This rebuilds both prompts from the thread's checkpointed state and its
workspace, so the answer takes two seconds instead of a run.

    python scripts/show_prompt.py <thread_id> [--repo .workspace]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ci_log                                          # noqa: E402
from app.config import get_settings                             # noqa: E402
from app.graph import build_graph, make_checkpointer            # noqa: E402
from app.llm import retry_model                                 # noqa: E402
from app.nodes.propose_diff import DIFF_SYSTEM, DIFF_USER, _retry_context  # noqa: E402
from app.prompts import PLAN_SYSTEM, PLAN_USER                  # noqa: E402
from app.schemas import ChangePlan                              # noqa: E402


def _rule(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("thread_id")
    ap.add_argument("--repo", default="", help="override the repo in state")
    args = ap.parse_args()

    s = get_settings()
    graph = build_graph(make_checkpointer())
    snap = graph.get_state({"configurable": {"thread_id": args.thread_id}})
    if not snap.created_at:
        raise SystemExit(f"no checkpoint for thread {args.thread_id}")
    st = dict(snap.values)
    if args.repo:
        st["repo_path"] = args.repo
    repo = Path(st.get("repo_path", ".workspace"))
    log = st.get("ci_failure_log", "")

    print(f"thread      : {args.thread_id}")
    print(f"retry_count : {st.get('retry_count', 0)}")
    print(f"ci_status   : {st.get('ci_status')}")
    print(f"model       : {s.ollama_model}"
          f"{'  (retries: ' + retry_model() + ')' if retry_model() != s.ollama_model else ''}")
    print(f"repo        : {repo}")

    _rule("ci_failure_log, as stored")
    print(log or "(empty — nothing for a retry to work from)")

    _rule("digest")
    print(ci_log.digest(log) if log else "(none)")

    _rule("failing tests, read from the workspace")
    print(ci_log.failing_tests(repo, log) if log else "(none)")

    _rule("required behaviour")
    print(ci_log.expectations(repo, log) if log
          else "(none — the prompts cannot state what the test wants)")

    _rule("PLAN prompt (system)")
    print(PLAN_SYSTEM)

    extra = ""
    if note := st.get("edit_note"):
        extra = f"\nThe reviewer rejected the previous attempt with this note:\n{note}\n"
    if log:
        extra += f"\nThe previous change failed CI. Failing output:\n```\n{log[-2000:]}\n```\n"
        if musts := ci_log.expectations(repo, log):
            extra += (f"\nRequired behaviour — the plan must say exactly this, "
                      f"naming the same exception type:\n{musts}\n")

    plan = st.get("plan") or {}
    target = repo / plan.get("target_file", "")
    source = target.read_text(encoding="utf-8") if target.is_file() else "(file not found)"

    _rule("PLAN prompt (user)")
    print(PLAN_USER.format(task=st.get("task", ""),
                           path=plan.get("target_file", "?"),
                           source=source, extra_context=extra))

    _rule("REWRITE prompt (system)")
    print(DIFF_SYSTEM)

    _rule("REWRITE prompt (user)")
    if not plan:
        print("(no plan in state yet)")
        return
    p = ChangePlan.model_validate(plan)
    print(DIFF_USER.format(
        summary=p.summary,
        steps="\n".join(f"{i}. {st_}" for i, st_ in enumerate(p.steps, 1)),
        context=_retry_context(st),
        path=p.target_file,
        source=source,
        line_count=source.strip().count("\n") + 1,
    ))


if __name__ == "__main__":
    main()
