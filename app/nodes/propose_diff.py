"""propose_diff node: ChangePlan -> validated unified diff."""
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from app.config import get_settings
from app.diffing import PatchError, build_unified_diff, validate_patch
from app.llm import get_llm
from app.schemas import FileRewrite
from app.state import AgentState

DIFF_SYSTEM = """You rewrite a single Python file to implement an approved plan.

Return a structured FileRewrite object with ALL required fields:

* `path`: the relative path of the file being modified.
* `new_content`: the COMPLETE new file content. Not a diff. Not a snippet.
  Not an explanation. The entire file, top to bottom, ready to save to disk.
* `commit_message`: a short imperative git commit message describing the change.

Rules:

* Preserve every part of the file the plan does not mention: imports, docstrings,
  unrelated functions, blank lines, comments.
* Keep the existing indentation style exactly.
* Make the smallest change necessary to implement the plan.
* Do not add example usage or demonstration code unless explicitly requested.
* Do not add unnecessary comments.
* Do not repeat code.
* Do not wrap content in markdown code fences.
* Do not add commentary outside the structured response."""

DIFF_USER = """Plan: {summary}

Steps:
{steps}

Target file: {path}

Current content:

```python
{source}
```

Rewrite the file according to the plan.

Requirements:

* Set `path` to exactly `{path}`.
* Return the complete replacement content in `new_content`.
* Provide a concise imperative `commit_message`.
* Preserve unrelated code.
* Make only the changes required by the plan."""

def propose_diff_node(state: AgentState) -> dict:
    s = get_settings()
    repo = Path(state["repo_path"])
    plan = state["plan"]
    target = repo / plan.target_file
    before = target.read_text(encoding="utf-8")

    prompt = [
        ("system", DIFF_SYSTEM),
        (
            "user",
            DIFF_USER.format(
                summary=plan.summary,
                steps="\n".join(
                    f"{i}. {st}" for i, st in enumerate(plan.steps, 1)
                ),
                path=plan.target_file,
                source=before,
            ),
        ),
    ]

    llm = get_llm(
        num_predict=s.ollama_num_predict
    ).with_structured_output(
        FileRewrite,
        method="json_schema",
    )

    last_error = ""

    for attempt in range(2):
        messages = list(prompt)

        if last_error:
            messages.append(
                (
                    "user",
                    "Your previous response was invalid.\n"
                    f"Error: {last_error}\n\n"
                    "Return a valid FileRewrite object with all required fields: "
                    "path, new_content, and commit_message.",
                )
            )

        try:
            result: FileRewrite = llm.invoke(messages)

            # Guard against the model changing a different file.
            if result.path != plan.target_file:
                raise PatchError(
                    f"Expected path {plan.target_file!r}, "
                    f"got {result.path!r}"
                )

            diff = build_unified_diff(
                plan.target_file,
                before,
                result.new_content,
            )

            validate_patch(repo, diff)

        except (PatchError, OutputParserException) as exc:
            last_error = str(exc)
            continue

        return {
            "diff": diff,
            "new_content": result.new_content,
            "commit_message": result.commit_message,
            "approval_status": "pending",
            "error": "",
        }

    return {
        "error": (
            "Could not produce a valid patch after "
            f"2 attempts: {last_error}"
        )
    }
