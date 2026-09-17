"""plan node: task + source -> ChangePlan."""
import re
from pathlib import Path

from app import ci_log
from app.config import get_settings
from app.llm import get_llm, retry_model
from app.prompts import PLAN_SYSTEM, PLAN_USER
from app.schemas import ChangePlan
from app.state import AgentState

# Directories that are never the agent's business.
#
# Build and vendor dirs are here so a checked-out virtualenv does not flood the
# candidate list. `tests` and `demo` are here for a different and more important
# reason: the tests are the specification the CI loop grades the agent against,
# so the agent must not be able to edit them. Otherwise the cheapest way to make
# a red build green is to delete the failing test, and W2D10's self-healing loop
# would eventually find it. Filtering `test_*.py` by name is not enough — a
# conftest.py or a helper module inside tests/ is just as load-bearing.
_SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules",
              ".tox", ".mypy_cache", ".pytest_cache", "build", "dist", ".workspace",
              "tests", "test", "demo", ".github"}


# Anything that looks like an exception class, for spotting a step that names
# the wrong one ("raise a ValueError if it is None" when the test wants TypeError).
_EXC = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b")


class PlanError(RuntimeError):
    """The plan node cannot proceed — reported to the user, not retried."""


def _repo_root(state: AgentState) -> Path:
    raw = state.get("repo_path") or ""
    if not raw:
        raise PlanError(
            "No repo_path in state. Pass --repo when starting a thread."
        )
    repo = Path(raw).expanduser().resolve()
    if not repo.is_dir():
        raise PlanError(f"repo_path {repo} does not exist or is not a directory.")
    return repo


def _is_replan(state: AgentState) -> bool:
    """True when we re-entered plan from the approval gate or a CI failure.

    A replan must stay on the file the reviewer was looking at. A NEW task on
    the same thread must not: reusing the old target there means the second
    request is silently planned against the first request's file, however
    plainly the task names a different one.
    """
    return bool(state.get("edit_note") or state.get("ci_failure_log"))

def list_candidates(repo: Path) -> list[Path]:
    return sorted(
        f for f in repo.rglob("*.py")
        if not (_SKIP_DIRS & set(f.relative_to(repo).parts))
        and not f.name.startswith("test_") and not f.name.endswith("_test.py")
    )


def _resolve_target(state: AgentState, repo: Path) -> Path:
    """Locate the target file. On a replan, stay on the file already chosen."""
    if _is_replan(state) and (p := state.get("plan")) and p.get("target_file"):
        prior = repo / p["target_file"]
        if not prior.is_file():
            raise PlanError(
                f"The previous plan targeted {p['target_file']!r}, which does not "
                f"exist under {repo}. The thread may have been started against a "
                f"different repo."
            )
        return prior

    candidates = list_candidates(repo)
    if not candidates:
        raise PlanError(f"No Python files found under {repo}.")

    task = state.get("task", "").lower()
    for f in candidates:
        if f.name.lower() in task:
            return f
    if len(candidates) == 1:
        return candidates[0]
    raise PlanError(
        "Could not identify a target file. Name one explicitly in the task. "
        f"Candidates: {sorted(str(c.relative_to(repo)) for c in candidates)}"
    )


def _pin_required_exceptions(plan: ChangePlan, repo: Path, log: str) -> None:
    """Add the exception type the tests demand as a step, in Python.

    Live evidence (three attempts, one thread): told the failing test, shown its
    source, and told outright that TypeError was required, the 3B model still
    planned "raise a ValueError if it is None" every time — and the rewrite
    follows the plan. The requirement is mechanical, so it is written here
    instead of asked for, the same bargain as ADR-001.
    """
    if not log:
        return
    required = ci_log.required_exceptions(repo, log)
    if not required:
        return

    # Once Python knows every exception the tests demand, the model's own words
    # about exception types are only a chance to contradict them — including
    # "change the raise to ValueError", which is how the loop started flipping
    # one guard back and forth. Drop them all and state the contract instead.
    kept = [st for st in plan.steps if not _EXC.search(st)]
    pinned = [
        f"Raise {exc} — not any other exception type — when called as "
        f"`{call}` ({test})." if call else
        f"Raise {exc} — not any other exception type — as {test} requires."
        for test, exc, call in required
    ]
    if len({exc for _t, exc, _c in required}) > 1:
        pinned.append("Use a separate check per required exception: one combined "
                      "condition cannot raise two different types.")
    plan.steps = (pinned + kept)[:5]


def plan_node(state: AgentState) -> dict:
    s = get_settings()

    if not (state.get("task") or "").strip():
        # Nearly always means a thread was restarted with an empty task instead
        # of resumed. Planning against "" yields a no-op rewrite and an
        # "Empty diff" three nodes later, which is a miserable thing to debug.
        return {"error": "Empty task. To continue an existing thread use "
                         "`python -m app.cli resume <id>`."}

    try:
        repo = _repo_root(state)
        target = _resolve_target(state, repo)
        source = target.read_text(encoding="utf-8")
    except (PlanError, OSError) as exc:
        return {"error": str(exc)}

    line_count = source.count("\n") + 1
    if line_count > s.max_file_lines:
        return {"error": f"{target.name} is {line_count} lines; limit is {s.max_file_lines} (ADR-003)."}

    # On a re-plan after CI failure or an edit request, feed the reason back in.
    extra = ""
    if note := state.get("edit_note"):
        extra = f"\nThe reviewer rejected the previous attempt with this note:\n{note}\n"
    if log := state.get("ci_failure_log"):
        extra += f"\nThe previous change failed CI. Failing output:\n```\n{log[-2000:]}\n```\n"
        # Without this the plan says "validate the input" and the rewrite obeys
        # the plan, not the test: three attempts in a row raised ValueError for
        # a test that requires TypeError.
        if musts := ci_log.expectations(repo, log):
            extra += (f"\nRequired behaviour — the plan must say exactly this, "
                      f"naming the same exception type:\n{musts}\n")

    model = retry_model() if _is_replan(state) else None
    llm = get_llm(num_predict=s.ollama_num_predict, model=model).with_structured_output(
        ChangePlan, method="json_schema"
    )
    plan: ChangePlan = llm.invoke([
        ("system", PLAN_SYSTEM),
        ("user", PLAN_USER.format(
            task=state["task"],
            path=str(target.relative_to(repo)),
            source=source,
            extra_context=extra,
        )),
    ])
    plan.target_file = str(target.relative_to(repo))  # trust our resolution, not the model's
    _pin_required_exceptions(plan, repo, state.get("ci_failure_log", ""))
    # Stored as a dict, not a Pydantic object: see app/state.py.
    return {"plan": plan.model_dump(), "repo_path": str(repo), "error": ""}