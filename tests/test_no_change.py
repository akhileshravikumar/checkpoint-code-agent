"""A task that is already done must end as "no change", not as an error, and
nothing in the prompts may pressure the model into inventing an edit.

Real propose_diff and plan nodes; only the LLM is stubbed, with a script of
responses so each attempt can answer differently.
"""
import sqlite3
import subprocess

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app import graph as graph_mod
from app.nodes import plan as plan_mod
from app.nodes import propose_diff as diff_mod
from app.schemas import ChangePlan

BEFORE = 'def parse_query(q):\n    if not q:\n        raise ValueError("empty")\n    return q.split()\n'
CHANGED = ('def parse_query(q):\n    if not isinstance(q, str):\n        raise TypeError("str")\n'
           '    if not q:\n        raise ValueError("empty")\n    return q.split()\n')
BROKEN = 'def parse_query(q):\n    return """oops\n'
PLAN = ChangePlan(target_file="search.py", summary="Add input validation to parse_query",
                  steps=["Raise ValueError on empty input"], rationale="guard input")


class Msg:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"done_reason": "stop"}


@pytest.fixture
def llm(monkeypatch):
    """Script the rewrite responses and record every prompt sent."""
    box = {"script": [], "prompts": []}

    class Rewrite:
        def invoke(self, messages):
            box["prompts"].append(messages)
            return Msg(box["script"].pop(0))
        def stream(self, messages):
            yield self.invoke(messages)

    class Plan:
        def with_structured_output(self, *_a, **_k):
            return type("P", (), {"invoke": lambda self, m: PLAN.model_copy()})()

    monkeypatch.setattr(plan_mod, "get_llm", lambda **_k: Plan())
    monkeypatch.setattr(diff_mod, "get_llm", lambda **_k: Rewrite())
    return box


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    (r / "search.py").write_text(BEFORE)
    return r


def _run(tmp_path, repo, **state):
    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
    saver.setup()
    graph = graph_mod.build_graph(saver)
    cfg = {"configurable": {"thread_id": "nc"}}
    out = graph.invoke({"task": "add input validation to search.py",
                        "repo_path": str(repo), "retry_count": 0, **state}, config=cfg)
    return graph, cfg, out


def test_already_done_ends_as_no_change(llm, repo, tmp_path):
    llm["script"] = [BEFORE, BEFORE]
    graph, cfg, out = _run(tmp_path, repo)
    assert not out.get("error")
    assert out["no_change_reason"].startswith("No change proposed for search.py")
    assert "Add input validation" in out["no_change_reason"], "show what the plan was"
    assert graph.get_state(cfg).next == (), "the run ends; there is nothing to approve"
    assert (repo / "search.py").read_text() == BEFORE


def test_no_prompt_demands_a_change(llm, repo, tmp_path):
    llm["script"] = [BEFORE, BEFORE]
    _run(tmp_path, repo)
    system = llm["prompts"][0][0][1]
    assert "MUST differ" not in system
    retry_hint = llm["prompts"][1][-1][1]
    assert "return it unchanged again" in retry_hint
    assert "must change" not in retry_hint.lower()


def test_a_lazy_first_answer_still_gets_a_second_chance(llm, repo, tmp_path):
    llm["script"] = [BEFORE, CHANGED]
    _, _, out = _run(tmp_path, repo)
    assert out["__interrupt__"], "a real change on attempt 2 reaches the gate"
    assert not out.get("no_change_reason")
    assert out["rewrite_attempt"] == 2, "METRICS needs to know which attempt won"


def test_unchanged_then_broken_is_an_error_not_no_change(llm, repo, tmp_path):
    llm["script"] = [BEFORE, BROKEN]
    _, _, out = _run(tmp_path, repo)
    assert "not valid Python" in out["error"]
    assert not out.get("no_change_reason")


def test_no_change_on_a_retry_is_an_error(llm, repo, tmp_path):
    """After a CI failure, "no change" leaves a red PR: that must be loud."""
    llm["script"] = [BEFORE, BEFORE]
    _, _, out = _run(tmp_path, repo, retry_count=1,
                     plan=PLAN.model_dump(), ci_failure_log="E   TypeError")
    assert "proposed no change to search.py on retry 1" in out["error"]
    assert not out.get("no_change_reason")

CI_LOG = ("E   AttributeError: 'NoneType' object has no attribute 'strip'\n"
          "FAILED tests/test_search.py::test_parse_query_rejects_none - AttributeError: ...")
TESTS = ("import pytest\nfrom search import parse_query\n\n\n"
         "def test_parse_query_rejects_none():\n    with pytest.raises(TypeError):\n"
         "        parse_query(None)\n")


def test_a_ci_retry_shows_the_rewrite_the_failure_and_the_test(llm, repo, tmp_path):
    """The live bug: the rewrite never saw the CI failure, so it returned the
    file unchanged twice and the retry died as "no change"."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    llm["script"] = [BEFORE, CHANGED]
    _, _, out = _run(tmp_path, repo, retry_count=1, plan=PLAN.model_dump(),
                     ci_failure_log=CI_LOG)
    assert out["__interrupt__"], "the fix reaches the gate"
    first_user = llm["prompts"][0][1][1]
    assert "FAILED CI" in first_user
    assert "test_parse_query_rejects_none" in first_user
    assert "pytest.raises(TypeError)" in first_user, "the expectation, not just the error"
    assert "requires TypeError from `parse_query(None)`" in first_user, "stated, not inferred"
    hint = llm["prompts"][1][-1][1]
    assert "version that was rejected" in hint
    assert "return it unchanged again" not in hint


def test_an_edit_note_reaches_the_rewrite(llm, repo, tmp_path):
    llm["script"] = [CHANGED]
    _run(tmp_path, repo, retry_count=1, plan=PLAN.model_dump(),
         edit_note="raise TypeError when query is not a str")
    assert "raise TypeError when query is not a str" in llm["prompts"][0][1][1]


def test_a_fresh_task_prompt_has_no_retry_context(llm, repo, tmp_path):
    llm["script"] = [CHANGED]
    _run(tmp_path, repo)
    user = llm["prompts"][0][1][1]
    assert "FAILED CI" not in user and "reviewer" not in user


def test_the_plan_prompt_states_the_required_exception(llm, repo, tmp_path, monkeypatch):
    """The plan is what the rewrite follows. If the plan says "validate the
    input" while the test wants TypeError, the rewrite obeys the plan."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    seen = {}

    class Plan:
        def with_structured_output(self, *_a, **_k):
            return type("P", (), {"invoke": lambda self, m: (seen.update(prompt=m[1][1])
                                                             or PLAN.model_copy())})()
    monkeypatch.setattr(plan_mod, "get_llm", lambda **k: Plan())
    llm["script"] = [CHANGED]
    _run(tmp_path, repo, retry_count=1, plan=PLAN.model_dump(), ci_failure_log=CI_LOG)
    assert "Required behaviour" in seen["prompt"]
    assert "requires TypeError from `parse_query(None)`" in seen["prompt"]


def test_retries_can_run_on_a_stronger_model(llm, repo, tmp_path, monkeypatch):
    from app import llm as llm_mod

    class S:
        ollama_model = "qwen2.5-coder:3b"
        ollama_model_retry = "qwen2.5-coder:7b"
    monkeypatch.setattr(llm_mod, "get_settings", lambda: S())
    assert llm_mod.retry_model() == "qwen2.5-coder:7b"
    S.ollama_model_retry = ""
    assert llm_mod.retry_model() == "qwen2.5-coder:3b", "unset means: same model"


def test_the_model_is_chosen_per_attempt(llm, repo, tmp_path, monkeypatch):
    picked = []
    monkeypatch.setattr(diff_mod, "retry_model", lambda: "big-model")
    real = diff_mod.get_llm
    monkeypatch.setattr(diff_mod, "get_llm",
                        lambda **k: (picked.append(k.get("model")), real(**k))[1])
    llm["script"] = [CHANGED]
    _run(tmp_path, repo)                       # first attempt: default model
    assert picked == [None]
    picked.clear()
    llm["script"] = [CHANGED]
    _run(tmp_path, repo, retry_count=1, plan=PLAN.model_dump(), ci_failure_log=CI_LOG)
    assert picked == ["big-model"]


# --- the plan itself must carry the required exception (live: it never did) ---

def test_the_plan_gets_the_required_exception_pinned_in_python(repo):
    from app.nodes.plan import _pin_required_exceptions

    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    p = ChangePlan(target_file="search.py", summary="Add input validation",
                   steps=["Check if query is None and raise a ValueError if it is."],
                   rationale="guard")
    _pin_required_exceptions(p, repo, CI_LOG)
    assert "Raise TypeError" in p.steps[0], "the requirement leads the plan"
    assert "parse_query(None)" in p.steps[0], "and says what must raise it"
    assert not any("ValueError" in st for st in p.steps), "the contradiction is gone"
    assert len(p.steps) <= 5


def test_every_step_about_exceptions_is_written_by_python(repo):
    """Even a step that happens to be right is the model's phrasing; the
    contract is known exactly, so it is stated exactly."""
    from app.nodes.plan import _pin_required_exceptions

    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    p = ChangePlan(target_file="search.py", summary="s",
                   steps=["Raise TypeError when query is not a str",
                          "Keep the lowercase split"], rationale="r")
    _pin_required_exceptions(p, repo, CI_LOG)
    assert [st for st in p.steps if "TypeError" in st] == [
        "Raise TypeError — not any other exception type — when called as "
        "`parse_query(None)` (tests/test_search.py::test_parse_query_rejects_none)."]
    assert "Keep the lowercase split" in p.steps, "steps about anything else survive"


def test_nothing_is_pinned_on_a_fresh_task(repo):
    from app.nodes.plan import _pin_required_exceptions

    p = ChangePlan(target_file="search.py", summary="s", steps=["a"], rationale="r")
    _pin_required_exceptions(p, repo, "")
    assert p.steps == ["a"]


def test_requirements_lead_the_rewrite_prompt(llm, repo, tmp_path):
    """Ordering matters: the plan is what the model follows, so anything that
    overrides it has to arrive before it, not after."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    llm["script"] = [CHANGED]
    _run(tmp_path, repo, retry_count=1, plan=PLAN.model_dump(), ci_failure_log=CI_LOG)
    user = llm["prompts"][0][1][1]
    assert user.index("Required behaviour") < user.index("Plan:")


BOTH_TESTS = TESTS + """

def test_parse_query_rejects_empty():
    with pytest.raises(ValueError):
        parse_query("   ")
"""
BOTH_LOG = (CI_LOG + "\nFAILED tests/test_search.py::test_parse_query_rejects_empty - Failed: ...")


def test_both_contracts_are_pinned_when_the_file_demands_two_types(repo):
    """The oscillation case: satisfying the red test by breaking the green one."""
    from app.nodes.plan import _pin_required_exceptions

    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(BOTH_TESTS)
    p = ChangePlan(target_file="search.py", summary="s",
                   steps=["Change the raise in parse_query to ValueError."],
                   rationale="r")
    _pin_required_exceptions(p, repo, CI_LOG)
    joined = "\n".join(p.steps)
    assert "Raise TypeError" in joined and "Raise ValueError" in joined
    assert "separate check per required exception" in joined
    assert "Change the raise" not in joined, "the contradicting step is gone"


def test_dropping_every_step_still_leaves_a_usable_plan(repo):
    from app.nodes.plan import _pin_required_exceptions

    (repo / "tests").mkdir()
    (repo / "tests" / "test_search.py").write_text(TESTS)
    p = ChangePlan(target_file="search.py", summary="s",
                   steps=["raise a ValueError", "raise RuntimeError instead"], rationale="r")
    _pin_required_exceptions(p, repo, CI_LOG)
    assert p.steps and all("ValueError" not in st for st in p.steps)
    assert "TypeError" in p.steps[0]
    assert len(p.steps) == 1, "nothing of the model's survives when it was all wrong"
