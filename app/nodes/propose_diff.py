"""propose_diff node: ChangePlan -> validated unified diff.

ADR-007 — the file body is plain text, not a JSON string. *(amends ADR-001)*

ADR-001 moved diff arithmetic out of the model. It left one thing in that a
small model is just as bad at: emitting the file as a JSON string value. A
6-line module with two docstrings needs 12 correctly-escaped quote characters,
and qwen2.5-coder:3b drops one often enough to fail every run — closing a
triple-quoted docstring with two quotes instead of three, and producing an
unterminated string literal that has nothing to do with the code it was asked
to write. (Writing this docstring hit the same bug: a literal triple quote in
the prose closed it early.)

So the rewrite is now an ordinary completion whose entire body is the file, and
`commit_message` is computed in Python from the plan. Nothing the model emits
has to be escaped, and the one remaining field it could get wrong is gone.

The trade is that the output is unconstrained: it may arrive fenced or wrapped
in commentary. Both are cheap to strip, and `check_rewrite` rejects anything
else — which is the same bargain as ADR-001, just moved one layer out.
"""
from pathlib import Path
from langgraph.config import get_stream_writer

from app.config import get_settings
from app.diffing import (
    PatchError,
    RewriteError,
    build_unified_diff,
    check_rewrite,
    extract_file_body,
    validate_patch,
)
from app.llm import get_llm
from app.schemas import ChangePlan
from app.state import AgentState

DIFF_SYSTEM = """You rewrite a single Python file to implement an approved plan.

Your ENTIRE response is written to disk as the file, byte for byte. Output the
file content and nothing else.

Rules:

* No markdown fences. No explanation before or after. No commentary.
* Output the WHOLE file, first line to last, including every import, docstring,
  function and class that is already there.
* Preserve every part the plan does not mention.
* Copy strings and docstrings EXACTLY as they appear, including the quoting
  style. Do not rewrap or re-quote them.
* Keep the existing indentation style.
* Make the smallest change that satisfies the plan."""

DIFF_USER = """Plan: {summary}

Steps:
{steps}

Target file: {path}

Current content ({line_count} lines):
---BEGIN FILE---
{source}
---END FILE---

Rewrite the file according to the plan. Your answer must be about {line_count}
lines — the whole file, with only the planned change applied.

Output the complete file now, starting with its first line:"""


NO_CHANGE = (
    "No change proposed for {path}. The model returned the file unchanged on "
    "both attempts, so it may already do what you asked. Current plan: "
    "{summary!r}. If something should change, name the exact behaviour, e.g. "
    "which input should raise which exception."
)


MAX_ATTEMPTS = 2   # the first try plus one bounded retry (ADR-001)


class TruncatedError(RuntimeError):
    """Generation hit num_predict before the model finished the file."""


# What to tell the model on the second attempt. Generic "try again" feedback is
# useless when the response was structurally fine but semantically a no-op.
_RETRY_HINTS = {
    # Deliberately NOT "you must change something". When the task is already
    # done, pressure to differ is how a model invents a cosmetic edit that
    # reaches the gate looking like a real one. A second unchanged answer after
    # this hint is taken as a considered "no change" (see NO_CHANGE below).
    "empty": (
        "Your previous answer was identical to the current file. Re-read the "
        "plan steps and check each one against the code. If a step is not yet "
        "implemented, apply it and return the whole file. If the file already "
        "satisfies every step, return it unchanged again."
    ),
    "dropped": (
        "Your previous answer was NOT the complete file — it replaced the module "
        "instead of editing it. Copy the current content line for line, apply "
        "ONLY the planned change, and return the whole thing. Every function, "
        "class, import and docstring that exists now must still exist."
    ),
    "syntax": (
        "Your previous answer was not valid Python. Copy every string and "
        "docstring from the current content EXACTLY, character for character, "
        "including the quoting style. A docstring opened with three double "
        "quotes must be closed with three double quotes."
    ),
    "apply": (
        "Your previous answer produced a patch git could not apply, which "
        "usually means content was dropped. Return the ENTIRE file — every line "
        "from the first to the last, including parts the plan does not touch."
    ),
}


def _classify(exc: Exception) -> str:
    if isinstance(exc, RewriteError):
        # "you deleted the module" and "you broke the quoting" need different
        # advice; the same hint for both sends the model chasing the wrong fix.
        return "syntax" if "not valid Python" in str(exc) else "dropped"
    if "Empty diff" in str(exc):
        return "empty"
    return "apply"


def _commit_message(plan: ChangePlan) -> str:
    """Derive the commit message rather than asking for it.

    It was the last structured field, and the model has nothing to add that the
    plan does not already contain. One fewer thing to get wrong.
    """
    scope = Path(plan.target_file).stem
    summary = plan.summary.strip().rstrip(".")
    if summary:
        summary = summary[0].lower() + summary[1:]
    msg = f"fix({scope}): {summary or 'apply planned change'}"
    if len(msg) <= 72:
        return msg
    # Cut at the last word boundary, never mid-word: titles like
    # "...to the parse_query function to ensur" end up on public PRs.
    head = msg[:72]
    cut = head.rsplit(" ", 1)[0].rstrip(" ,;:-`(")
    return cut if len(cut) > len(f"fix({scope}): ") else head

def _writer():
    """The graph's stream writer, or a no-op when called outside a graph run."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _ev: None


def _generate(llm, messages, attempt: int):
    """Stream the completion, reporting progress; return the merged message.

    Merging the chunks keeps response_metadata, so done_reason (truncation
    detection) works exactly as it did with invoke().
    """
    write, full, n = _writer(), None, 0
    for chunk in llm.stream(messages):
        full = chunk if full is None else full + chunk
        n += 1
        if n % 20 == 0:                   # ~1 update/second on CPU
            write({"type": "token_progress", "node": "propose_diff",
                   "attempt": attempt, "max_attempts": MAX_ATTEMPTS, "tokens": n})
    return full

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
                line_count=before.strip().count(chr(10)) + 1,
            ),
        ),
    ]

    last_error = ""
    kind = ""
    kinds: list[str] = []

    for attempt in range(MAX_ATTEMPTS):  # one bounded retry (ADR-001)
        # A second pass at temperature 0.1 against an unchanged prompt
        # re-samples almost the same tokens. Nudge it off the previous mode.
        llm = get_llm(
            num_predict=s.ollama_num_predict_rewrite,
            temperature=s.ollama_temperature if attempt == 0 else 0.4,
            streaming=True,
        )

        messages = list(prompt)
        if kind:
            messages.append(("user", _RETRY_HINTS[kind] + f"\n\n(Error: {last_error})"))

        try:
            resp = _generate(llm, messages, attempt + 1)

            meta = getattr(resp, "response_metadata", {}) or {}
            if meta.get("done_reason") == "length":
                raise TruncatedError(
                    f"Generation hit num_predict={s.ollama_num_predict_rewrite} "
                    f"before finishing the file. Raise "
                    f"OLLAMA_NUM_PREDICT_REWRITE or lower MAX_FILE_LINES "
                    f"(file is {before.count(chr(10)) + 1} lines)."
                )

            new_content = extract_file_body(resp.content)

            # Semantic check first: a patch that deletes the file applies
            # perfectly cleanly, so git cannot be the one to catch this.
            check_rewrite(plan.target_file, before, new_content)

            diff = build_unified_diff(plan.target_file, before, new_content)
            validate_patch(repo, diff)

        except TruncatedError as exc:
            # Not the model's fault and not fixable by re-prompting it.
            return {"error": str(exc)}
        except PatchError as exc:
            last_error = str(exc)
            kind = _classify(exc)
            kinds.append(kind)
            continue

        return {
            "diff": diff,
            "new_content": new_content,
            "commit_message": _commit_message(plan),
            "approval_status": "pending",
            "error": "",
        }

    if kinds == ["empty", "empty"]:
        if state.get("retry_count", 0) == 0:
            # A fresh task that is already done is an answer, not a failure.
            # Nothing is applied and nothing reaches the gate either way.
            return {
                "diff": "", "new_content": "",
                "no_change_reason": NO_CHANGE.format(
                    path=plan.target_file, summary=plan.summary),
                "error": "",
            }
        # On a retry (CI failure or edit request) "no change" leaves the PR as
        # it is, i.e. still red or still not what the reviewer asked for.
        return {"error": (
            f"The model proposed no change to {plan.target_file} on retry "
            f"{state.get('retry_count', 0)}, so nothing was pushed. The PR is "
            f"unchanged. Try an edit note that names the exact fix."
        )}
    return {"error": f"Could not produce a valid patch after {MAX_ATTEMPTS} attempts: {last_error}"}