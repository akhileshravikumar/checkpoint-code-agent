#!/usr/bin/env bash
# Environment preflight for Checkpoint.
set -uo pipefail
[ -f .env ] && set -a && . ./.env && set +a
PASS=0; FAIL=0
ok(){ echo "  ok    $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
warn(){ echo "  WARN  $1"; }

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
# ADR-010: the agent should push as its own GitHub App, not as you.
TOKEN=""
if [ -n "${GITHUB_APP_ID:-}" ]; then
  KEY="${GITHUB_APP_PRIVATE_KEY_PATH/#\~/$HOME}"
  [ -f "$KEY" ] && ok "app private key present" || no "app private key missing: $KEY"
  case "$(realpath -m "$KEY")" in
    "$(pwd)"/*) no "private key is INSIDE the repo; move it out (e.g. ~/.config/checkpoint/)";;
    *) ok "private key is outside the repo";;
  esac
  if ID=$(python -c "from app.github_client import credentials; print(credentials().identity)" 2>&1); then
    ok "agent identity: $ID"
    TOKEN=$(python -c "from app.github_client import credentials; print(credentials().token)")
  else
    no "GitHub App auth failed: $(echo "$ID" | tail -1)"
  fi
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  warn "agent uses a PAT: it acts as you, including your admin bypass (docs/github-app-setup.md)"
  TOKEN="$GITHUB_TOKEN"
else
  no "no GitHub credentials (GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY_PATH)"
fi
if [ -n "$TOKEN" ]; then
  C=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}")
  [ "$C" = "200" ] && ok "agent can reach ${GITHUB_REPO}" || no "agent -> HTTP $C"
  P=$(curl -s -H "Authorization: Bearer $TOKEN" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/branches/${GITHUB_BASE_BRANCH}" \
      | jq -r '.protected')
  [ "$P" = "true" ] && ok "main is protected" || no "main NOT protected"
fi
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  A=$(gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/branches/${GITHUB_BASE_BRANCH}/protection" \
      --jq '.enforce_admins.enabled' 2>/dev/null || echo "?")
  [ "$A" = "false" ] && ok "you (admin) can push to main directly" \
                     || warn "enforce_admins=$A: your own pushes to main will be blocked too"
fi

echo "-- langsmith --"
if [ "${CHECKPOINT_OFFLINE:-0}" = "1" ]; then ok "offline mode (tracing skipped)"
elif [ -n "${LANGSMITH_API_KEY:-}" ]; then
  C=$(curl -s -o /dev/null -w '%{http_code}' -H "x-api-key: ${LANGSMITH_API_KEY}" \
      "${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}/info")
  [ "$C" = "200" ] && ok "langsmith reachable" || no "langsmith -> HTTP $C"
else no "LANGSMITH_API_KEY unset"; fi

echo; echo "$PASS passed, $FAIL failed"; [ "$FAIL" -eq 0 ]
