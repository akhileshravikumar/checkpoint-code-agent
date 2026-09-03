"""plan node: task + source -> ChangePlan."""
from pathlib import Path

from app.config import get_settings
from app.llm import get_llm
from app.prompts import PLAN_SYSTEM, PLAN_USER
from app.schemas import ChangePlan
from app.state import AgentState

# Directories that are never the agent's business, and would otherwise flood
# the candidate list in any repo with a virtualenv checked out beside the code.
_SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules",
              ".tox", ".mypy_cache", ".pytest_cache", "build", "dist", ".workspace"}


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


def _resolve_target(state: AgentState, repo: Path) -> Path:
    """Locate the target file. If a previous plan chose one, reuse it."""
    if (p := state.get("plan")) and p.get("target_file"):
        prior = repo / p["target_file"]
        if not prior.is_file():
            raise PlanError(
                f"The previous plan targeted {p['target_file']!r}, which does not "
                f"exist under {repo}. The thread may have been started against a "
                f"different repo."
            )
        return prior

    candidates = [
        f for f in repo.rglob("*.py")
        if not (_SKIP_DIRS & set(f.relative_to(repo).parts))
        and not f.name.startswith("test_")
        and not f.name.endswith("_test.py")
    ]
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


def plan_node(state: AgentState) -> dict:
    s = get_settings()

    if not (state.get("task") or "").strip():
        # Nearly always means a thread was restarted with an empty task instead
        # of resumed. Planning against "" yields a no-op rewrite and an
        # "Empty diff" three nodes later, which is a miserable thing to debug.
        return {"error": "Empty task. To continue an existing thread use "
                         "`python -m app.cli resume --thread <id>`."}

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

    llm = get_llm(num_predict=s.ollama_num_predict).with_structured_output(
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
    # Stored as a dict, not a Pydantic object: see app/state.py.
    return {"plan": plan.model_dump(), "repo_path": str(repo), "error": ""}
