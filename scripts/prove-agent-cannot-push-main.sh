#!/usr/bin/env bash
# ADR-010 proof: the agent's identity cannot push to a protected branch.
#
# Never touches main. It creates a throwaway branch from main, protects it
# exactly like main (PR required, admins may bypass), tries to push an empty
# commit to it AS THE AGENT, then deletes the branch again whatever happens.
#
#   expected with the GitHub App : "OK: rejected"   (the App is not an admin)
#   with the PAT fallback        : "FAIL"            (a PAT acts as you, the admin)
#
# Needs: gh logged in as YOU (the repo admin), the venv active, run from the
# agent repo root.  Screenshot the output for the portfolio.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${GITHUB_OWNER:?set GITHUB_OWNER}"
REPO="$GITHUB_OWNER/${GITHUB_REPO:-checkpoint-sandbox}"
PROBE="protection-probe-$(date +%s)"
TMP="$(mktemp -d)"

cleanup() {
  gh api -X DELETE "repos/$REPO/branches/$PROBE/protection" >/dev/null 2>&1 || true
  gh api -X DELETE "repos/$REPO/git/refs/heads/$PROBE" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT

echo "==> throwaway branch $PROBE on $REPO, protected like main"
MAIN_SHA=$(gh api "repos/$REPO/git/ref/heads/main" --jq .object.sha)
gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/$PROBE" -f sha="$MAIN_SHA" >/dev/null
gh api -X PUT "repos/$REPO/branches/$PROBE/protection" --input - >/dev/null <<'JSON'
{"required_status_checks": null, "enforce_admins": false,
 "required_pull_request_reviews": {"required_approving_review_count": 0},
 "restrictions": null, "allow_force_pushes": false, "allow_deletions": false}
JSON

# The agent's credentials, exactly as execute gets them. Never echoed.
python - > "$TMP/creds" <<'PY'
from app.github_client import credentials
c = credentials()
print(c.identity)
print(c.token)
PY
IDENTITY=$(sed -n 1p "$TMP/creds"); TOKEN=$(sed -n 2p "$TMP/creds")
echo "==> pushing as: $IDENTITY"

git clone -q --depth 1 --branch "$PROBE" \
  "https://x-access-token:${TOKEN}@github.com/$REPO.git" "$TMP/clone"
git -C "$TMP/clone" -c user.name=probe -c user.email=probe@example.com \
  commit -q --allow-empty -m "probe: this push must be rejected"

if git -C "$TMP/clone" push -q origin "$PROBE" 2> "$TMP/err"; then
  echo "FAIL: $IDENTITY pushed to a protected branch, so it could push to main too."
  exit 1
fi
echo "OK: rejected. GitHub said:"
python - "$TMP/err" "$TOKEN" <<'PY'
import sys
err, token = open(sys.argv[1]).read(), sys.argv[2]
print("   " + "\n   ".join(l for l in err.replace(token, "***").splitlines() if l.strip())[:600])
PY
