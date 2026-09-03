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

import difflib
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
    add = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return {"additions": add, "deletions": rem}
