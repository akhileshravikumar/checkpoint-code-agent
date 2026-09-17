"""The self-heal retry must see which test failed and what it expects.

Found on a live run: CI failed on the seeded None test, plan and propose_diff
ran again, the model returned the file unchanged twice, and the retry ended with
"proposed no change on retry 1". The rewrite prompt had never seen the failure.
"""
from pathlib import Path

from app import ci_log

FIXTURE = Path(__file__).parent / "fixtures" / "actions_job_log_pytest_failure.txt"
RAW = FIXTURE.read_text(encoding="utf-8")

TEST_FILE = '''import pytest
from sandbox.search import parse_query


def test_parse_query_splits_and_lowercases():
    assert parse_query("  Python Guide ") == ["python", "guide"]


def test_parse_query_rejects_none():
    with pytest.raises(TypeError):
        parse_query(None)
'''


def test_the_raw_tail_is_mostly_post_job_noise():
    """Why focus exists: the old prompt input was the log's last 2000 chars."""
    old = RAW[-4000:][-2000:]
    assert "FAILED tests/test_search.py::test_parse_query_rejects_none" not in old
    assert old.count("[command]/usr/bin/git") >= 5


def test_focus_keeps_pytest_output_and_drops_cleanup():
    out = ci_log.focus(RAW)
    assert out.startswith("=" * 35), "starts at pytest's FAILURES section"
    assert "E   AttributeError: 'NoneType' object has no attribute 'strip'" in out
    assert "FAILED tests/test_search.py::test_parse_query_rejects_none" in out
    assert "Post job cleanup" not in out
    assert "git config" not in out
    assert "pip install" not in out
    assert "2026-09-17T" not in out, "timestamps stripped"
    assert "﻿" not in out


def test_focus_without_pytest_keeps_the_failing_step():
    raw = "\n".join(f"2026-09-17T17:00:0{i}.0000000Z {ln}" for i, ln in enumerate([
        "##[group]Run pip install pytest", "pip install pytest", "##[endgroup]",
        "ERROR: No matching distribution found for pytest==99",
        "##[error]Process completed with exit code 1.", "Post job cleanup.",
    ]))
    out = ci_log.focus(raw)
    assert "No matching distribution" in out
    assert "Post job cleanup" not in out


def test_focus_respects_max_chars_from_the_end():
    out = ci_log.focus(RAW, max_chars=120)
    assert len(out) <= 120
    assert out.endswith("1 failed, 4 passed in 0.03s")


def test_digest_names_the_test_and_the_error():
    d = ci_log.digest(ci_log.focus(RAW))
    assert "FAILED tests/test_search.py::test_parse_query_rejects_none" in d
    assert "E   AttributeError" in d
    assert "short test summary" not in d


def test_failing_tests_reads_the_expectation_from_the_workspace(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(TEST_FILE)
    src = ci_log.failing_tests(tmp_path, ci_log.focus(RAW))
    assert "pytest.raises(TypeError)" in src, "the log alone never says TypeError"
    assert "def test_parse_query_rejects_none" in src
    assert "splits_and_lowercases" not in src, "only the failing tests"


def test_failing_tests_is_best_effort(tmp_path):
    assert ci_log.failing_tests(tmp_path, ci_log.focus(RAW)) == ""
    escape = "FAILED ../../etc/test_x.py::test_x - boom"
    assert ci_log.failing_tests(tmp_path, escape) == ""


def test_failure_log_returns_the_focused_log(monkeypatch):
    from app import github_client as gc

    class Resp:
        def __init__(self, code, body=None, text=""):
            self.status_code, self._body, self.text = code, body, text
        def json(self):
            return self._body

    class Http:
        def get(self, url, **_k):
            if url.endswith("/jobs"):
                return Resp(200, {"jobs": [{"id": 7, "conclusion": "success"},
                                           {"id": 9, "conclusion": "failure"}]})
            assert url.endswith("/jobs/9/logs")
            return Resp(200, text=RAW)

    client = gc.GitHubClient.__new__(gc.GitHubClient)
    client._http = Http()
    client.s = type("S", (), {"repo_slug": "o/r"})()
    out = client.failure_log(123)
    assert "FAILED tests/test_search.py::test_parse_query_rejects_none" in out
    assert "git config" not in out

# --- the expected exception type (live run: 3 attempts, 3x ValueError) -------

RAISES_VARIANTS = '''import pytest
from sandbox.search import parse_query


def test_bare():
    with pytest.raises(TypeError):
        parse_query(None)


def test_imported_raises():
    with raises(ValueError, match="empty"):
        parse_query("  ")


def test_tuple():
    with pytest.raises((TypeError, ValueError)):
        parse_query(1)


def test_dotted():
    with pytest.raises(exceptions.CustomError):
        parse_query(1)


def test_no_raises():
    assert parse_query("a") == ["a"]
'''


def _log(*names):
    return "\n".join(f"FAILED tests/test_search.py::{n} - boom" for n in names)


def test_expectations_names_the_required_exception(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(TEST_FILE)
    out = ci_log.expectations(tmp_path, ci_log.focus(RAW))
    assert "requires TypeError from `parse_query(None)`" in out
    assert "test_parse_query_rejects_none" in out


def test_expectations_handles_the_shapes_pytest_allows(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(RAISES_VARIANTS)
    log = _log("test_bare", "test_imported_raises", "test_tuple", "test_dotted",
               "test_no_raises")
    out = ci_log.expectations(tmp_path, log)
    assert "test_bare requires TypeError" in out
    assert "test_imported_raises requires ValueError" in out
    assert "test_tuple requires TypeError or ValueError" in out
    assert "test_dotted requires CustomError" in out
    assert "test_no_raises" not in out, "nothing to state for a plain assertion"


def test_expectations_is_empty_without_the_workspace(tmp_path):
    assert ci_log.expectations(tmp_path, ci_log.focus(RAW)) == ""


# --- the whole contract, not just today's red test ---------------------------

TWO_CONTRACTS = '''import pytest
from sandbox.search import parse_query


def test_parse_query_rejects_none():
    with pytest.raises(TypeError):
        parse_query(None)


def test_parse_query_rejects_empty():
    with pytest.raises(ValueError):
        parse_query("   ")


def test_parse_query_splits():
    assert parse_query("A b") == ["a", "b"]
'''


def test_expectations_state_every_required_exception_in_the_file(tmp_path):
    """Live failure: stating only the red test made the model flip one raise
    between TypeError and ValueError, turning the other test red each time."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(TWO_CONTRACTS)
    out = ci_log.expectations(
        tmp_path, "FAILED tests/test_search.py::test_parse_query_rejects_none - boom")
    assert "test_parse_query_rejects_none requires TypeError" in out
    assert "test_parse_query_rejects_empty requires ValueError" in out, "the green one too"
    assert "each needs its own check" in out
    assert "test_parse_query_splits" not in out, "no exception, nothing to state"


def test_a_single_required_type_gets_no_warning(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(TEST_FILE)
    out = ci_log.expectations(
        tmp_path, "FAILED tests/test_search.py::test_parse_query_rejects_none - boom")
    assert "each needs its own check" not in out


def test_required_exceptions_carries_the_call_for_each(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_search.py").write_text(TWO_CONTRACTS)
    got = ci_log.required_exceptions(
        tmp_path, "FAILED tests/test_search.py::test_parse_query_rejects_empty - boom")
    assert [(e, c) for _t, e, c in got] == [
        ("TypeError", "parse_query(None)"), ("ValueError", "parse_query('   ')")]
