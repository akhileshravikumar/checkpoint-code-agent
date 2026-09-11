"""The event bus is fed from worker threads and read on the event loop."""
import asyncio
import threading
import time

from app.events import EventBus


def test_publish_from_a_worker_thread_wakes_the_loop():
    """asyncio.Queue.put_nowait from another thread never wakes the loop; the
    message sat there until the worker finished. CI progress and every W3D2
    live update depend on this arriving immediately."""
    bus = EventBus()

    async def main():
        q = bus.subscribe("t")

        def worker():
            time.sleep(0.1)
            bus.publish("t", {"type": "x"})
            time.sleep(5)                     # the graph is still running

        threading.Thread(target=worker, daemon=True).start()
        t0 = time.monotonic()
        msg = await asyncio.wait_for(q.get(), timeout=3)
        return msg, time.monotonic() - t0

    msg, dt = asyncio.run(main())
    assert msg == {"type": "x"}
    assert dt < 1.0, f"delivered after {dt:.2f}s"


def test_history_replays_and_order_is_kept():
    bus = EventBus()

    async def main():
        q = bus.subscribe("t")
        for i in range(5):
            bus.publish("t", {"i": i})
        return [(await q.get())["i"] for _ in range(5)]

    assert asyncio.run(main()) == [0, 1, 2, 3, 4]
    assert [m["i"] for m in bus.history("t")] == [0, 1, 2, 3, 4]


def test_unsubscribed_queues_receive_nothing():
    bus = EventBus()

    async def main():
        q = bus.subscribe("t")
        bus.unsubscribe("t", q)
        bus.publish("t", {"x": 1})
        await asyncio.sleep(0.05)
        return q.qsize()

    assert asyncio.run(main()) == 0
