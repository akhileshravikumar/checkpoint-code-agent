"""plan node: task + source -> ChangePlan."""
from pathlib import Path

from app.config import get_settings
from app.llm import get_llm
from app.prompts import PLAN_SYSTEM, PLAN_USER
from app.schemas import ChangePlan
from app.state import AgentState


def _resolve_target(state: AgentState) -> Path:
    """Locate the target file. If a previous plan chose one, reuse it."""
    repo = Path(state["repo_path"])
    if (p := state.get("plan")) and p.target_file:
        return repo / p.target_file

    # First pass: pick the candidate whose name appears in the task text,
    # else the only Python file if there is exactly one.
    candidates = [
        f for f in repo.rglob("*.py")
        if ".git" not in f.parts and "test" not in f.name
    ]
    task = state["task"].lower()
    for f in candidates:
        if f.name.lower() in task:
            return f
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not identify a target file. Name one explicitly in the task. "
        f"Candidates: {[str(c.relative_to(repo)) for c in candidates]}"
    )


def plan_node(state: AgentState) -> dict:
    s = get_settings()
    repo = Path(state["repo_path"])
    target = _resolve_target(state)
    source = target.read_text(encoding="utf-8")

    line_count = source.count("\n") + 1
    if line_count > s.max_file_lines:
        return {"error": f"{target.name} is {line_count} lines; limit is {s.max_file_lines} (ADR-003)."}

    # On a re-plan after CI failure or an edit request, feed the reason back in.
    extra = ""
    if note := state.get("edit_note"):
        extra = f"\nThe reviewer rejected the previous attempt with this note:\n{note}\n"
    if log := state.get("ci_failure_log"):
        extra += f"\nThe previous change failed CI. Failing output:\n```\n{log[-2000:]}\n```\n"

    llm = get_llm(num_predict=512).with_structured_output(ChangePlan, method="json_schema")
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
    return {"plan": plan, "error": ""}