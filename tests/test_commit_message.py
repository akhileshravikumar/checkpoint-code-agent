from app.nodes.propose_diff import _commit_message
from app.schemas import ChangePlan

def _p(**kw):
    base = dict(target_file="sandbox/search.py", summary="Reject empty queries.",
                steps=["a"], rationale="r")
    base.update(kw)
    return ChangePlan(**base)

def test_commit_message_is_derived_not_asked_for():
    assert _commit_message(_p()) == "fix(search): reject empty queries"

def test_commit_message_is_bounded():
    assert len(_commit_message(_p(summary="x" * 200))) <= 72

def test_commit_message_survives_an_empty_summary():
    m = _commit_message(_p(summary="  "))
    assert m.startswith("fix(search):") and len(m) > 12
def test_commit_message_is_cut_at_a_word_boundary():
    """Seen on a real PR: '...to the `parse_query` function to ensur'."""
    summary = ("Add input validation to the parse_query function "
               "to ensure it raises for empty input")
    m = _commit_message(_p(summary=summary))
    assert len(m) <= 72
    assert m.split()[-1] in summary.lower().split(), f"cut mid-word: {m!r}"


def test_a_single_huge_word_is_still_bounded():
    m = _commit_message(_p(summary="x" * 200))
    assert m.startswith("fix(search): x") and len(m) == 72
