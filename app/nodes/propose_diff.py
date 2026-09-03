"""propose_diff node: ChangePlan -> validated unified diff."""
from pathlib import Path

from app.config import get_settings
from app.diffing import PatchError, build_unified_diff, validate_patch
from app.llm import get_llm
from app.schemas import FileRewrite
from app.state import AgentState

DIFF_SYSTEM = """You rewrite a single Python file to implement an approved plan.

Return the COMPLETE new file content in `new_content`. Not a diff. Not a snippet.
Not an explanation. The entire file, top to bottom, ready to save to disk.

Rules:
- Preserve every part of the file the plan does not mention: imports, docstrings,
  unrelated functions, blank lines, comments.
- Keep the existing indentation style exactly.
- Do not wrap the output in markdown code fences.
- Do not add commentary."""

DIFF_USER = """Plan: {summary}

Steps:
{steps}

Current content of {path}:
```python
{source}
```

Return the complete rewritten file."""


def propose_diff_node(state: AgentState) -> dict:
    s = get_settings()
    repo = Path(state["repo_path"])
    plan = state["plan"]
    target = repo / plan.target_file
    before = target.read_text(encoding="utf-8")

    prompt = [
        ("system", DIFF_SYSTEM),
        ("user", DIFF_USER.format(
            summary=plan.summary,
            steps="\n".join(f"{i}. {st}" for i, st in enumerate(plan.steps, 1)),
            path=plan.target_file,
            source=before,
        )),
    ]

    llm = get_llm(num_predict=s.ollama_num_predict).with_structured_output(
        FileRewrite, method="json_schema"
    )

    last_error = ""
    for attempt in range(2):  # one bounded retry (ADR-001)
        messages = list(prompt)
        if last_error:
            messages.append((
                "user",
                f"Your previous output produced an invalid patch: {last_error}\n"
                "Return the complete file again, unabridged.",
            ))
        try:
            result: FileRewrite = llm.invoke(messages)
            diff = build_unified_diff(plan.target_file, before, result.new_content)
            validate_patch(repo, diff)
        except PatchError as exc:
            last_error = str(exc)
            continue
        return {
            "diff": diff,
            "new_content": result.new_content,
            "commit_message": result.commit_message,
            "approval_status": "pending",
            "error": "",
        }

    return {"error": f"Could not produce a valid patch after 2 attempts: {last_error}"}