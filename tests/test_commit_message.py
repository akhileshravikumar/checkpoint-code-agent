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