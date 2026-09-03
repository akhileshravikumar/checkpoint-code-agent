"""Diff generation and validation.

The model never writes a unified diff (ADR-001). It returns the complete new
file body; we compute the patch with difflib and prove it applies with
`git apply --check` before any human ever sees it.
"""
from __future__ import annotations

import difflib
import subprocess
from pathlib import Path


class PatchError(RuntimeError):
    """The generated patch does not apply cleanly."""


def _normalise(text: str) -> str:
    """LF endings, exactly one trailing newline.

    Without this, a model that omits the final newline produces a
    '\\ No newline at end of file' marker and the patch fails to apply.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def build_unified_diff(rel_path: str, before: str, after: str, context: int = 3) -> str:
    before, after = _normalise(before), _normalise(after)
    if before == after:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=context,
        )
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
    add = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return {"additions": add, "deletions": rem}