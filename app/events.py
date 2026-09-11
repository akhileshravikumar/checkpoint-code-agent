"""In-process pub/sub between the graph and connected dashboards.

THREADS. publish() is called from the graph's worker threads (the graph runs
under asyncio.to_thread). asyncio.Queue is not thread-safe: put_nowait() from
another thread enqueues the item but never wakes the event loop, so the
message waits until something else wakes the loop — typically the worker
thread finishing. Every intermediate event (CI progress, node transitions,
token progress) then arrives all at once at the end of the run. So each
subscriber remembers its loop, and delivery goes through call_soon_threadsafe.
"""
import asyncio
import threading
from collections import defaultdict


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = (
            defaultdict(list)
        )
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        """Must be called from inside the event loop that will read the queue."""
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subs[thread_id].append((loop, q))
        return q

    def unsubscribe(self, thread_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs[thread_id] = [s for s in self._subs[thread_id] if s[1] is not q]

    def history(self, thread_id: str) -> list[dict]:
        with self._lock:
            return list(self._history[thread_id])

    def publish(self, thread_id: str, message: dict) -> None:
        """Safe to call from any thread, including the event loop's own."""
        with self._lock:
            self._history[thread_id].append(message)
            targets = list(self._subs[thread_id])
        for loop, q in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, message)
            except RuntimeError:
                pass  # loop already closed: that subscriber is gone


bus = EventBus()