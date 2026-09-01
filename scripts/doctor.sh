#!/usr/bin/env bash
# Environment preflight for Checkpoint.
set -uo pipefail
[ -f .env ] && set -a && . ./.env && set +a
PASS=0; FAIL=0
ok(){ echo "  ok    $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "-- platform --"
uname -a | grep -qi microsoft && ok "WSL2" || no "not WSL2"
[[ "$PWD" == /home/* ]] && ok "Linux filesystem" || no "running from /mnt/c (slow, CRLF risk)"

echo "-- python --"
python --version 2>&1 | grep -q "3.12" && ok "python 3.12" || no "python is not 3.12"
[ -n "${VIRTUAL_ENV:-}" ] && ok "venv active" || no "venv not active"

echo "-- ollama --"
curl -fs "${OLLAMA_BASE_URL:-http://localhost:11434}/api/version" >/dev/null \
  && ok "ollama reachable" || no "ollama unreachable"
ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}" \
  && ok "model ${OLLAMA_MODEL:-?} present" || no "model ${OLLAMA_MODEL:-?} not pulled"

echo "-- github --"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  C=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $GITHUB_TOKEN" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}")
  [ "$C" = "200" ] && ok "PAT reaches ${GITHUB_REPO}" || no "PAT -> HTTP $C"
  P=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/branches/${GITHUB_BASE_BRANCH}" \
      | jq -r '.protected')
  [ "$P" = "true" ] && ok "main is protected" || no "main NOT protected"
else no "GITHUB_TOKEN unset"; fi

echo "-- langsmith --"
if [ "${CHECKPOINT_OFFLINE:-0}" = "1" ]; then ok "offline mode (tracing skipped)"
elif [ -n "${LANGSMITH_API_KEY:-}" ]; then
  C=$(curl -s -o /dev/null -w '%{http_code}' -H "x-api-key: ${LANGSMITH_API_KEY}" \
      "${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}/info")
  [ "$C" = "200" ] && ok "langsmith reachable" || no "langsmith -> HTTP $C"
else no "LANGSMITH_API_KEY unset"; fi

echo; echo "$PASS passed, $FAIL failed"; [ "$FAIL" -eq 0 ]
