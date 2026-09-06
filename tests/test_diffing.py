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


def _init(repo: Path, content: str | bytes) -> Path:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    p = repo / "search.py"
    p.write_bytes(content.encode() if isinstance(content, str) else content)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True,
    )
    return repo


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _init(tmp_path, BEFORE)


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


# --- the before side must match the working tree byte for byte ---

def test_crlf_file_on_disk_still_applies(tmp_path: Path):
    """A CRLF file must not be rejected because we normalised the context lines."""
    crlf_before = BEFORE.replace("\n", "\r\n")
    repo = _init(tmp_path, crlf_before)
    diff = build_unified_diff("search.py", crlf_before, AFTER)   # model replies in LF
    validate_patch(repo, diff)          # must not raise
    apply_patch(repo, diff)
    assert (repo / "search.py").read_bytes() == AFTER.replace("\n", "\r\n").encode()


def test_no_trailing_newline_on_disk_still_applies(tmp_path: Path):
    before = BEFORE.rstrip("\n")
    repo = _init(tmp_path, before)
    diff = build_unified_diff("search.py", before, AFTER)
    validate_patch(repo, diff)          # must not raise


def test_cosmetic_newline_difference_is_not_a_change(repo):
    """Model drops the trailing newline but changes nothing else -> empty diff."""
    assert build_unified_diff("search.py", BEFORE, BEFORE.rstrip("\n")) == ""


# --- the semantic guard: a valid patch that destroys the file ---

SANDBOX = (
    '"""A tiny query utility. Deliberately under-validated."""\n'
    "\n"
    "\n"
    "def parse_query(query):\n"
    '    """Split a raw query string into lowercase terms."""\n'
    "    return query.strip().lower().split()\n"
)


def test_one_line_replacement_is_rejected(repo):
    """The live-demo failure: the model returned `import re` as the whole file.

    git apply --check passes on this, and the diff is not empty, so neither
    existing guard sees anything wrong.
    """
    from app.diffing import RewriteError, check_rewrite
    diff = build_unified_diff("sandbox/search.py", SANDBOX, "import re\n")
    assert diff, "not an empty diff"
    with pytest.raises(RewriteError, match="parse_query"):
        check_rewrite("sandbox/search.py", SANDBOX, "import re\n")


def test_dropping_a_function_is_rejected():
    from app.diffing import RewriteError, check_rewrite
    two = SANDBOX + "\n\ndef rank(xs):\n    return sorted(xs)\n"
    with pytest.raises(RewriteError, match="rank"):
        check_rewrite("sandbox/search.py", two, SANDBOX)


def test_invalid_python_is_rejected_before_ci_ever_sees_it():
    from app.diffing import RewriteError, check_rewrite
    broken = SANDBOX.replace("return query", "return query(")
    with pytest.raises(RewriteError, match="not valid Python"):
        check_rewrite("sandbox/search.py", SANDBOX, broken)


def test_empty_new_content_is_rejected():
    from app.diffing import RewriteError, check_rewrite
    with pytest.raises(RewriteError, match="empty"):
        check_rewrite("sandbox/search.py", SANDBOX, "   \n")


def test_a_legitimate_edit_passes():
    from app.diffing import check_rewrite
    good = SANDBOX.replace(
        "    return query.strip().lower().split()",
        "    if not query or not query.strip():\n"
        "        raise ValueError('empty query')\n"
        "    return query.strip().lower().split()",
    )
    check_rewrite("sandbox/search.py", SANDBOX, good)      # must not raise


def test_a_deliberate_shrink_still_passes_if_symbols_survive():
    """Removing comments is legitimate; removing parse_query is not."""
    from app.diffing import check_rewrite
    verbose = SANDBOX + "\n\n# a comment\n# another\n# and another\n# and more\n"
    check_rewrite("sandbox/search.py", verbose, SANDBOX)   # must not raise


def test_markdown_fences_are_stripped_not_retried():
    """A 3B model wraps the file in ```python often enough to handle it."""
    from app.diffing import check_rewrite, strip_code_fences
    fenced = "```python\n" + SANDBOX + "```\n"
    assert strip_code_fences(fenced).strip() == SANDBOX.strip()
    check_rewrite("sandbox/search.py", SANDBOX, strip_code_fences(fenced))


def test_a_trailing_fence_alone_is_stripped():
    from app.diffing import strip_code_fences
    assert strip_code_fences(SANDBOX + "```").strip() == SANDBOX.strip()


def test_unfenced_content_is_untouched():
    from app.diffing import strip_code_fences
    assert strip_code_fences(SANDBOX) == SANDBOX


def test_syntax_error_message_shows_the_offending_lines():
    from app.diffing import RewriteError, check_rewrite
    broken = SANDBOX + "\n\ndef rank(xs)\n    return sorted(xs)\n"
    with pytest.raises(RewriteError) as e:
        check_rewrite("sandbox/search.py", SANDBOX, broken)
    msg = str(e.value)
    assert "def rank(xs)" in msg, "the error must show what actually broke"
    assert ">>" in msg


def test_reused_apostrophe_docstring_is_diagnosable():
    """The live failure: the model re-quoted a docstring containing an apostrophe."""
    from app.diffing import RewriteError, check_rewrite
    src = '"""A utility. This is the agent\'s target."""\n\n\ndef f():\n    return 1\n'
    requoted = "\n'A utility. This is the agent's target.'\n\n"
    with pytest.raises(RewriteError) as e:
        check_rewrite("sandbox/search.py", src, requoted)
    msg = str(e.value)
    assert "unterminated string literal" in msg
    assert "raw head:" in msg, "the raw repr must survive copy-paste"


# --- ADR-007: the body arrives as plain text, not a JSON string ---

def test_extract_takes_the_fenced_block_when_one_is_present():
    from app.diffing import extract_file_body
    resp = "Here is the updated file:\n\n```python\n" + SANDBOX + "```\n\nHope that helps!"
    assert extract_file_body(resp).strip() == SANDBOX.strip()


def test_extract_passes_bare_code_through():
    from app.diffing import extract_file_body
    assert extract_file_body(SANDBOX).strip() == SANDBOX.strip()


def test_extract_prefers_the_largest_block():
    from app.diffing import extract_file_body
    resp = "```python\nx = 1\n```\nand the file:\n```python\n" + SANDBOX + "```"
    assert extract_file_body(resp).strip() == SANDBOX.strip()


def test_docstring_quoting_survives_a_plain_text_round_trip():
    """The failure ADR-007 exists to remove: no escaping, so nothing to drop."""
    from app.diffing import check_rewrite, extract_file_body
    edited = SANDBOX.replace(
        "    return query.strip().lower().split()",
        "    if not query:\n        raise ValueError('empty')\n"
        "    return query.strip().lower().split()",
    )
    check_rewrite("sandbox/search.py", SANDBOX, extract_file_body(edited))