"""In-process pub/sub between the graph and connected dashboards."""
import asyncio
from collections import defaultdict


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._history: dict[str, list[dict]] = defaultdict(list)

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[thread_id].append(q)
        return q

    def unsubscribe(self, thread_id: str, q: asyncio.Queue) -> None:
        if q in self._queues[thread_id]:
            self._queues[thread_id].remove(q)

    def history(self, thread_id: str) -> list[dict]:
        return list(self._history[thread_id])

    def publish(self, thread_id: str, message: dict) -> None:
        """Safe to call from a worker thread."""
        self._history[thread_id].append(message)
        for q in self._queues[thread_id]:
            q.put_nowait(message)


bus = EventBus()