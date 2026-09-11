"""W3D2: live progress. Node transitions and token progress reach the
dashboard during the run, and streaming does not change what propose_diff
decides (truncation is still detected from the merged chunks)."""
from langchain_core.messages import AIMessageChunk

from app.nodes import propose_diff as diff_mod
from tests.test_main import _until, client  # noqa: F401  (fixture)


def test_nodes_are_announced_in_order_before_the_gate(client):  # noqa: F811
    c, _ = client
    with c.websocket_connect("/ws?thread_id=s1") as ws:
        ws.send_json({"type": "start", "task": "fix sandbox/search.py"})
        seen = _until(ws, "diff_proposed")
    steps = [(m["type"], m["node"]) for m in seen if m["type"].startswith("node_")]
    assert steps == [
        ("node_enter", "plan"), ("node_exit", "plan"),
        ("node_enter", "propose_diff"), ("node_exit", "propose_diff"),
        ("node_enter", "await_approval"), ("node_paused", "await_approval"),
    ]


class _Streaming:
    """A stub LLM that streams real AIMessageChunks, like ChatOllama does."""

    def __init__(self, text, done_reason="stop", n=45):
        self.text, self.done_reason, self.n = text, done_reason, n

    def stream(self, _messages):
        size = max(1, len(self.text) // self.n)
        parts = [self.text[i:i + size] for i in range(0, len(self.text), size)]
        for i, part in enumerate(parts):
            meta = {"done_reason": self.done_reason} if i == len(parts) - 1 else {}
            yield AIMessageChunk(content=part, response_metadata=meta)


def test_generate_merges_chunks_and_reports_progress(monkeypatch):
    events = []
    monkeypatch.setattr(diff_mod, "_writer", lambda: events.append)
    text = "def parse_query(q):\n    return q.split()\n" * 3
    msg = diff_mod._generate(_Streaming(text), [], attempt=1)
    assert msg.content == text, "chunks merged back into the whole file"
    assert msg.response_metadata["done_reason"] == "stop"
    assert events and all(e["type"] == "token_progress" for e in events)
    assert events[0] == {"type": "token_progress", "node": "propose_diff",
                         "attempt": 1, "max_attempts": 2, "tokens": 20}


def test_truncation_is_still_detected_when_streamed():
    msg = diff_mod._generate(_Streaming("def parse_query(q):", done_reason="length"),
                             [], attempt=1)
    assert msg.response_metadata["done_reason"] == "length"


def test_generate_works_outside_a_graph_run():
    """No stream writer outside a run: progress is dropped, not an error."""
    msg = diff_mod._generate(_Streaming("x = 1\n", n=100), [], attempt=2)
    assert msg.content == "x = 1\n"