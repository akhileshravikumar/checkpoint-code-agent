"""Diff generation and validation.

The model never writes a unified diff (ADR-001). It returns the complete new
file body; we compute the patch with difflib and prove it applies with
`git apply --check` before any human ever sees it.

Newline handling: the `before` side is used EXACTLY as read from disk, because
`git apply --check` matches context lines against the working tree byte for
byte. Normalising both sides (the previous behaviour) produced context lines
that did not exist in the file whenever the file used CRLF or lacked a trailing
newline — a patch that passed difflib and then failed git. Only the model's
`after` side is coerced, into whatever shape the file on disk already has.
"""
from __future__ import annotations

import ast
import difflib
import re
import subprocess
from pathlib import Path


class PatchError(RuntimeError):
    """The generated patch does not apply cleanly."""


def _detect_shape(text: str) -> tuple[str, bool]:
    """Return (line ending, ends_with_newline) for an existing file body."""
    if "\r\n" in text:
        eol = "\r\n"
    elif "\r" in text and "\n" not in text:
        eol = "\r"
    else:
        eol = "\n"
    return eol, text.endswith(("\n", "\r"))


def _coerce_to(after: str, before: str) -> str:
    """Reshape model output to match the target file's existing conventions.

    A model that emits LF into a CRLF file, or drops the final newline, should
    not cause a spurious rejection — that is a formatting artefact, not a
    disagreement about the code.
    """
    eol, trailing = _detect_shape(before)
    body = after.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if trailing or body:
        body += "\n" if trailing else ""
    if eol != "\n":
        body = body.replace("\n", eol)
    return body


_NO_EOL = "\\ No newline at end of file\n"


def _mark_missing_eol(lines: list[str]) -> str:
    """Add git's `\\ No newline at end of file` markers.

    difflib emits the final line verbatim when the source had no trailing
    newline, which git rejects as `corrupt patch`. Any diff body line that does
    not end in a newline is exactly such a line.
    """
    out = []
    for line in lines:
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n")
            out.append(_NO_EOL)
    return "".join(out)


def build_unified_diff(rel_path: str, before: str, after: str, context: int = 3) -> str:
    """Diff `before` (verbatim, as on disk) against the model's `after`."""
    after = _coerce_to(after, before)
    if before == after:
        return ""
    return _mark_missing_eol(list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=context,
        )
    ))


_FENCE = re.compile(r"^```[A-Za-z0-9_+-]*$")


def strip_code_fences(text: str) -> str:
    """Remove markdown fences a model wrapped around the file body.

    The prompt says not to, and a bigger model wouldn't. A 3B one does it often
    enough that treating it as a formatting artefact — like line endings — beats
    spending a retry on it.
    """
    lines = text.strip("\n").split("\n")
    if lines and _FENCE.match(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
    elif lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines) + "\n"


_FENCE_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)```", re.S)


def extract_file_body(text: str) -> str:
    """Pull the file out of a plain-text completion.

    ADR-007 asks the model for the file body as prose, not as a JSON string, so
    there is no escaping to get wrong. The cost is that the model may wrap it in
    fences or bracket it with commentary; both are cheap to strip here, and
    anything else check_rewrite rejects.
    """
    blocks = _FENCE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    return strip_code_fences(text)


def _excerpt(source: str, lineno: int | None, radius: int = 3) -> str:
    """The neighbourhood of a syntax error, numbered, for the error message."""
    lines = source.split("\n")
    if lineno is None:
        lineno = len(lines)
    lo, hi = max(0, lineno - 1 - radius), min(len(lines), lineno + radius)
    out = []
    for i in range(lo, hi):
        mark = ">>" if i == lineno - 1 else "  "
        out.append(f"{mark} {i + 1:3d} | {lines[i]}")
    return "\n".join(out)


class RewriteError(PatchError):
    """The "complete file" the model returned plainly is not the complete file."""


def _toplevel_symbols(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def check_rewrite(rel_path: str, before: str, after: str, min_ratio: float = 0.5) -> None:
    """Reject a rewrite that dropped the file instead of editing it.

    ADR-001 has the model return the whole file and lets Python compute the
    diff, which makes a malformed *patch* impossible. It does nothing about a
    well-formed patch that deletes the program: replacing a 6-line module with
    `import re` applies perfectly cleanly, and neither `git apply --check` nor
    the empty-diff guard has any opinion about it.

    So this is the semantic half of validation. Deterministic, not a prompt —
    same reasoning as ADR-001. Runs before the diff is built, so the retry hint
    can tell the model exactly what it destroyed.
    """
    if not after.strip():
        raise RewriteError("new_content was empty.")

    if rel_path.endswith(".py"):
        try:
            ast.parse(after)
        except SyntaxError as exc:
            # Catching this here turns a CI round-trip into an instant retry.
            # The excerpt is what makes the failure debuggable from the
            # dashboard error card, instead of just a line number.
            # repr() of the head survives copy-paste into a chat or an issue,
            # where the excerpt's quotes and whitespace do not.
            raise RewriteError(
                f"new_content is not valid Python: line {exc.lineno}: {exc.msg}\n"
                f"{_excerpt(after, exc.lineno)}\n"
                f"raw head: {after[:160]!r}"
            ) from None

        lost = _toplevel_symbols(before) - _toplevel_symbols(after)
        if lost:
            raise RewriteError(
                "new_content is missing top-level definitions that were in the "
                f"original file: {', '.join(sorted(lost))}. The rewrite must "
                "contain the ENTIRE file."
            )

    n_before = len(before.strip().splitlines())
    n_after = len(after.strip().splitlines())
    if n_before and n_after < n_before * min_ratio:
        raise RewriteError(
            f"new_content is {n_after} lines but the original file is "
            f"{n_before}. Content was dropped rather than edited."
        )


def _git(repo: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin, text=True, capture_output=True,
    )


def validate_patch(repo: Path, diff: str) -> None:
    """Raise PatchError unless git can apply this patch to the working tree."""
    if not diff.strip():
        raise PatchError("Empty diff: the model returned the file unchanged.")
    r = _git(repo, "apply", "--check", "--verbose", "-", stdin=diff)
    if r.returncode != 0:
        raise PatchError(r.stderr.strip() or "git apply --check failed")


def apply_patch(repo: Path, diff: str) -> None:
    validate_patch(repo, diff)
    r = _git(repo, "apply", "-", stdin=diff)
    if r.returncode != 0:
        raise PatchError(r.stderr.strip())


def diff_stats(diff: str) -> dict[str, int]:
    add = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    rem = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return {"additions": add, "deletions": rem}