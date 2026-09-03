import subprocess
from pathlib import Path

import pytest

from app.diffing import PatchError, apply_patch, build_unified_diff, validate_patch

BEFORE = "def parse_query(q):\n    return q.strip().split()\n"
AFTER = (
    "def parse_query(q):\n"
    "    if not q.strip():\n"
    "        raise ValueError('empty query')\n"
    "    return q.strip().split()\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "search.py").write_text(BEFORE)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True,
    )
    return tmp_path


def test_diff_applies_cleanly(repo):
    diff = build_unified_diff("search.py", BEFORE, AFTER)
    apply_patch(repo, diff)
    assert (repo / "search.py").read_text() == AFTER


def test_missing_trailing_newline_is_normalised(repo):
    diff = build_unified_diff("search.py", BEFORE, AFTER.rstrip("\n"))
    validate_patch(repo, diff)          # must not raise


def test_crlf_input_is_normalised(repo):
    diff = build_unified_diff("search.py", BEFORE, AFTER.replace("\n", "\r\n"))
    validate_patch(repo, diff)


def test_no_change_is_rejected(repo):
    with pytest.raises(PatchError, match="Empty diff"):
        validate_patch(repo, build_unified_diff("search.py", BEFORE, BEFORE))


def test_stale_context_is_rejected(repo):
    diff = build_unified_diff("search.py", "totally different\n", AFTER)
    with pytest.raises(PatchError):
        validate_patch(repo, diff)

# --- regression: the before-side must match the working tree byte for byte ---

def test_crlf_file_on_disk_still_applies(tmp_path: Path):
    """A CRLF file must not be rejected because we normalised the context lines."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    crlf_before = BEFORE.replace("\n", "\r\n")
    (tmp_path / "search.py").write_bytes(crlf_before.encode())
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True,
    )
    # Model replies in LF, as models do.
    diff = build_unified_diff("search.py", crlf_before, AFTER)
    validate_patch(tmp_path, diff)      # must not raise
    apply_patch(tmp_path, diff)
    assert (tmp_path / "search.py").read_bytes() == AFTER.replace("\n", "\r\n").encode()


def test_no_trailing_newline_on_disk_still_applies(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    before = BEFORE.rstrip("\n")
    (tmp_path / "search.py").write_text(before)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True,
    )
    diff = build_unified_diff("search.py", before, AFTER)
    validate_patch(tmp_path, diff)      # must not raise


def test_cosmetic_newline_difference_is_not_a_change(repo):
    """Model drops the trailing newline but changes nothing else -> empty diff."""
    assert build_unified_diff("search.py", BEFORE, BEFORE.rstrip("\n")) == ""
