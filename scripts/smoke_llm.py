import time

from app.llm import get_llm
from app.tracing import configure_tracing

print("tracing:", configure_tracing())
llm = get_llm(num_predict=64)
t0 = time.perf_counter()
r = llm.invoke("Reply with exactly: READY")
print(f"{time.perf_counter() - t0:.1f}s -> {r.content!r}")