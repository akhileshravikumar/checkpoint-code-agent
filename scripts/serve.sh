#!/usr/bin/env bash
# Start the dashboard server the way a run needs it.
#
# NOT `uvicorn app.main:app --reload` on its own: the reloader watches *.py
# under the repo, .workspace/ is under the repo, and the agent writes .py files
# there on every approval. The server would restart in the middle of `execute`.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKSPACE="${WORKSPACE_DIR:-.workspace}"

if [ "${RELOAD:-0}" = "1" ]; then       # developing the server itself
  exec uvicorn app.main:app --reload \
       --reload-exclude "${WORKSPACE}/*" --reload-exclude "checkpoint.sqlite*" \
       "$@"
fi
exec uvicorn app.main:app "$@"
