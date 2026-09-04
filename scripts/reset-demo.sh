#!/usr/bin/env bash
# Reset every piece of state the Week-1 demo touches, so a run starts from zero.
#
# There are FOUR stores, and forgetting any one of them produces a confusing
# result rather than an obvious failure:
#   1. checkpoint.sqlite   - stale threads; a resumed thread outlives the demo
#   2. /tmp/fixture        - the CLI target repo
#   3. ./.workspace        - the dashboard target repo
#   4. browser localStorage - the dashboard's remembered thread_id (manual, see end)
set -euo pipefail
cd "$(dirname "$0")/.."

FIXTURE_PY='def parse_query(q):
    return q.strip().split()
'

_fresh_repo() {           # $1 = path
  rm -rf "$1" && mkdir -p "$1"
  printf '%s' "$FIXTURE_PY" > "$1/search.py"
  git -C "$1" init -q
  git -C "$1" add -A
  git -C "$1" -c user.email=t@t -c user.name=t commit -qm "init"
}

echo "==> checkpointer"
rm -f checkpoint.sqlite checkpoint.sqlite-wal checkpoint.sqlite-shm

echo "==> CLI target      /tmp/fixture"
_fresh_repo /tmp/fixture

echo "==> dashboard target ./.workspace"
_fresh_repo ./.workspace

echo "==> python caches"
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Prove both targets are clean rather than asserting it.
fail=0
for r in /tmp/fixture ./.workspace; do
  out=$(git -C "$r" status --porcelain)
  if [ -n "$out" ]; then echo "NOT CLEAN: $r"; echo "$out"; fail=1; fi
done
[ "$fail" -eq 0 ] || { echo; echo "reset FAILED"; exit 1; }

echo
echo "clean. 0 threads, both target repos committed with no local changes."
echo
echo "One store this script cannot touch: the browser."
echo "The dashboard remembers its thread in localStorage, so before testing the"
echo "HTML version open http://localhost:8000 , and in the DevTools console run:"
echo
echo "    localStorage.removeItem('checkpoint_thread'); location.href = location.pathname;"
echo
echo "That clears the key AND drops any ?thread_id= from the URL. Loading a URL"
echo "that still carries ?thread_id= will reconnect you to a thread this script"
echo "just deleted, and you will get an empty session that looks like a bug."
