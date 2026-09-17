"""Turn a raw GitHub Actions job log into something a 3B model can act on.

A job log is not a test report. Every line carries a 29-character timestamp, and
the step that failed is followed by post-job steps (checkout's credential
cleanup, orphan-process cleanup) that always run. The self-heal loop first took
the last 2000 characters, which for a one-test pytest failure is roughly two
thirds git-config noise, and can lose the FAILED line that names the test.

Even a clean pytest report does not say what the test *expects*: `--tb=short`
shows `parse_query(None)` raising AttributeError, not the
`pytest.raises(TypeError)` around it. So `failing_tests` reads the failing test
functions from the workspace, where the expectation actually lives.

`focus` keeps the failing step's output. `digest` keeps the lines that say which
test failed and why. Both are short enough to repeat in the rewrite prompt.

`expectations` goes one step further. Given the test source, a 3B model still
reads `pytest.raises(TypeError)` as decoration: on the first live self-heal run
it spent all three attempts refining *when* to reject (`not query`, then
`not query.strip()`, then `query is None`) and raised `ValueError` every time.
The expected exception is therefore extracted and stated as a requirement.
"""
import ast
import re
from pathlib import Path

_TS = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ?")
_ERROR_END = "##[error]Process completed with exit code"
_PYTEST_START = re.compile(r"^=+ (FAILURES|ERRORS) =+$")
_DIGEST_LINE = re.compile(
    r"^(FAILED |ERROR |E\s{2,}|\S+\.py:\d+: |>\s|=+ .*(failed|error).* =+$)"
)


def _lines(raw: str) -> list[str]:
    return [_TS.sub("", ln.rstrip("\r")) for ln in raw.splitlines()]


def focus(raw: str, max_chars: int = 4000) -> str:
    """The failing step's output, timestamps stripped, post-job steps dropped."""
    lines = _lines(raw)

    # Cut at the first step failure; everything after it is cleanup.
    end = next((i for i, ln in enumerate(lines) if ln.startswith(_ERROR_END)), len(lines))
    body = lines[:end]

    # Within that, start at pytest's FAILURES/ERRORS section when there is one.
    start = next((i for i in range(len(body) - 1, -1, -1)
                  if _PYTEST_START.match(body[i])), None)
    if start is None:
        # Not pytest (or collection blew up first): start at the failing step.
        start = next((i for i in range(len(body) - 1, -1, -1)
                      if body[i].startswith("##[group]Run ")), 0)

    kept = [ln for ln in body[start:]
            if ln.strip() and not ln.startswith(("##[group]", "##[endgroup]"))]
    text = "\n".join(kept)
    # Tail-truncate: pytest's short summary (which test, which exception) is last.
    return text[-max_chars:]


def digest(log: str, max_chars: int = 1200) -> str:
    """Only the lines naming the failing tests and their errors."""
    kept = [ln for ln in _lines(log) if _DIGEST_LINE.match(ln)]
    return "\n".join(kept)[-max_chars:] if kept else log[-max_chars:]


_FAILED_ID = re.compile(r"^(?:FAILED|ERROR) (\S+?\.py)::(\S+?)(?: - .*)?$")


def failing_tests(repo: Path, log: str, max_chars: int = 1500) -> str:
    """Source of the failing test functions named in the log, from the workspace.

    Best effort: anything that cannot be located or parsed is skipped.
    """
    seen, chunks = set(), []
    root = Path(repo).resolve()
    for ln in _lines(log):
        m = _FAILED_ID.match(ln)
        if not m:
            continue
        rel, name = m.group(1), m.group(2).split("::")[-1].split("[")[0]
        if (rel, name) in seen:
            continue
        seen.add((rel, name))
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            continue
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                seg = ast.get_source_segment(src, node)
                if seg:
                    chunks.append(f"# {rel}\n{seg}")
                break
    return "\n\n".join(chunks)[:max_chars]

def _raises_arg(node: ast.AST) -> str | None:
    """The exception named by a pytest.raises(...) call, if this is one."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name != "raises" or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Attribute):
        return arg.attr
    if isinstance(arg, ast.Tuple):
        names = [a.id for a in arg.elts if isinstance(a, ast.Name)]
        return " or ".join(names) or None
    return None


def required_exceptions(repo: Path, log: str) -> list[tuple[str, str, str]]:
    """[(test id, exception type, the call under `raises`)] — the WHOLE contract.

    Not just the tests that failed this time. A file can hold two tests that
    demand different exception types for different inputs, and stating only the
    red one makes the loop oscillate: seen live, the model merged both guards
    into `if not isinstance(query, str) or not query.strip()` and then flipped
    that single raise between TypeError and ValueError on alternate attempts,
    turning the other test red each time. Both constraints have to be in the
    prompt at once, or the model is being asked to satisfy a contradiction.
    """
    out, seen = [], set()
    root = Path(repo).resolve()
    failing = [m.group(1) for ln in _lines(log) if (m := _FAILED_ID.match(ln))]
    for rel in dict.fromkeys(failing):                  # de-dup, keep order
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.name.startswith("test"):
                continue
            for node in ast.walk(fn):
                if exc := _raises_arg(node):
                    key = (rel, fn.name)
                    if key not in seen:
                        seen.add(key)
                        out.append((f"{rel}::{fn.name}", exc, _raises_body(fn)))
                    break
    return out


def _raises_body(fn: ast.AST) -> str:
    """The first statement inside `with pytest.raises(...)`, as source."""
    for node in ast.walk(fn):
        if isinstance(node, ast.With) and any(
                _raises_arg(item.context_expr) for item in node.items):
            for stmt in node.body:
                try:
                    return ast.unparse(stmt).strip()
                except Exception:      # pragma: no cover - unparse is total in 3.9+
                    return ""
    return ""


def expectations(repo: Path, log: str, max_chars: int = 900) -> str:
    """Every exception the target's tests require, stated together."""
    required = required_exceptions(repo, log)
    if not required:
        return ""
    lines = [f"- {test} requires {exc}{f' from `{call}`' if call else ''}"
             for test, exc, call in required]
    if len({exc for _t, exc, _c in required}) > 1:
        lines.append("These are different exception types for different inputs: "
                     "each needs its own check. One combined condition raising a "
                     "single type cannot satisfy them all.")
    return "\n".join(lines)[:max_chars]
