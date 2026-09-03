"""propose_diff node: ChangePlan -> validated unified diff."""
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from app.config import get_settings
from app.diffing import PatchError, build_unified_diff, validate_patch
from app.llm import get_llm
from app.schemas import ChangePlan, FileRewrite
from app.state import AgentState

DIFF_SYSTEM = """You rewrite a single Python file to implement an approved plan.

Return a structured FileRewrite object with ALL required fields:

* `path`: the relative path of the file being modified.
* `commit_message`: a short imperative git commit message describing the change.
* `new_content`: the COMPLETE new file content. Not a diff. Not a snippet.
  Not an explanation. The entire file, top to bottom, ready to save to disk.

Rules:

* The file MUST differ from the current content. Returning it unchanged is a
  failure, not a valid answer.
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
* Provide a concise imperative `commit_message`.
* Return the complete replacement content in `new_content`.
* Preserve unrelated code.
* Make only the changes required by the plan."""


class TruncatedError(RuntimeError):
    """Generation hit num_predict before the model closed the JSON object."""


# What to tell the model on the second attempt. Generic "return all required
# fields" feedback is useless when the response was structurally valid but
# semantically a no-op — the model just repeats itself.
_RETRY_HINTS = {
    "empty": (
        "Your previous `new_content` was byte-identical to the current file. "
        "You did not apply the plan. Re-read the steps and produce a file that "
        "actually differs — the specific lines the plan describes must change."
    ),
    "path": (
        "Your previous response rewrote the wrong file. Set `path` to exactly "
        "the target path given above and rewrite THAT file."
    ),
    "apply": (
        "Your previous `new_content` produced a patch git could not apply. This "
        "usually means content was dropped. Return the ENTIRE file — every line "
        "from the first to the last, including parts the plan does not touch."
    ),
    "parse": (
        "Your previous response was not valid structured output. Return a "
        "FileRewrite object with exactly the fields path, commit_message and "
        "new_content, and nothing else."
    ),
}


def _classify(exc: Exception) -> str:
    if isinstance(exc, OutputParserException):
        return "parse"
    msg = str(exc)
    if "Empty diff" in msg:
        return "empty"
    if msg.startswith("Expected path"):
        return "path"
    return "apply"


def propose_diff_node(state: AgentState) -> dict:
    s = get_settings()
    repo = Path(state["repo_path"])
    plan = ChangePlan.model_validate(state["plan"])
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

    last_error = ""
    kind = ""

    for attempt in range(2):
        # A second pass at temperature 0.1 against an unchanged prompt
        # re-samples almost the same tokens. Nudge it off the previous mode.
        llm = get_llm(
            num_predict=s.ollama_num_predict_rewrite,
            temperature=s.ollama_temperature if attempt == 0 else 0.4,
        ).with_structured_output(
            FileRewrite,
            method="json_schema",
            include_raw=True,   # need response_metadata to spot truncation
        )

        messages = list(prompt)
        if kind:
            messages.append(("user", _RETRY_HINTS[kind] + f"\n\n(Error: {last_error})"))

        try:
            raw = llm.invoke(messages)

            if err := raw.get("parsing_error"):
                # Distinguish "ran out of tokens" from "emitted nonsense".
                meta = getattr(raw.get("raw"), "response_metadata", {}) or {}
                if meta.get("done_reason") == "length":
                    raise TruncatedError(
                        f"Generation hit num_predict="
                        f"{s.ollama_num_predict_rewrite} before finishing the "
                        f"file. Raise OLLAMA_NUM_PREDICT_REWRITE or lower "
                        f"MAX_FILE_LINES (file is "
                        f"{before.count(chr(10)) + 1} lines)."
                    )
                raise err

            result: FileRewrite = raw["parsed"]

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

        except TruncatedError as exc:
            # Not the model's fault and not fixable by re-prompting it.
            return {"error": str(exc)}
        except (PatchError, OutputParserException) as exc:
            last_error = str(exc)
            kind = _classify(exc)
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
