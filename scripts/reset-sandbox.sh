#!/usr/bin/env bash
# Put checkpoint-sandbox and the agent into the known demo state (W3D0 / W3D4).
# Run between takes. Needs `gh` authenticated as the sandbox repo's admin.
#
# Destructive on the SANDBOX only: closes the agent's open PRs, deletes every
# checkpoint/* branch, and pushes a commit to sandbox main that restores the
# unvalidated parse_query and seeds the failing tests. main is red afterwards;
# that is the point — it is what makes CI fail on attempt 1.
set -euo pipefail
[ -f "$(dirname "$0")/../.env" ] && set -a && . "$(dirname "$0")/../.env" && set +a
: "${GITHUB_OWNER:?set GITHUB_OWNER (in .env or the shell)}"
REPO="$GITHUB_OWNER/${GITHUB_REPO:-checkpoint-sandbox}"
SANDBOX="${SANDBOX:-$HOME/projects/checkpoint-sandbox}"
AGENT="$(cd "$(dirname "$0")/.." && pwd)"

[ -d "$SANDBOX/.git" ] || { echo "no sandbox clone at $SANDBOX (set SANDBOX=...)"; exit 1; }

echo "==> close agent PRs, delete checkpoint/* branches on $REPO"
gh pr list --repo "$REPO" --state open --json number,headRefName \
  --jq '.[] | select(.headRefName | startswith("checkpoint/")) | .number' |
  xargs -r -n1 gh pr close --repo "$REPO" --delete-branch
gh api "repos/$REPO/git/matching-refs/heads/checkpoint/" --jq '.[].ref' |
  xargs -r -I{} gh api -X DELETE "repos/$REPO/git/{}"

echo "==> sandbox main: unvalidated parse_query + seeded failing tests"
cd "$SANDBOX"
git checkout -q main && git pull -q --ff-only
cat > sandbox/search.py <<'PY'
"""A tiny query utility, deliberately under-validated. This is the agent target."""


def parse_query(query):
    """Split a raw query string into lowercase terms."""
    return query.strip().lower().split()


def rank_results(results, terms):
    """Score each result by how many query terms appear in its title."""
    scored = []
    for r in results:
        score = sum(1 for t in terms if t in r["title"].lower())
        scored.append({**r, "score": score})
    return sorted(scored, key=lambda r: r["score"], reverse=True)


def top_n(results, terms, n=3):
    """Return the n highest-scoring results."""
    return rank_results(results, terms)[:n]
PY
python demo/seed_failing_test.py
git add -A
if ! git diff --cached --quiet; then
  git commit -qm "chore(demo): reset parse_query and seed failing tests"
  # Admin bypass: protection was set with enforce_admins=false (W2 prerequisites).
  git push -q
fi
git log --oneline -1

echo "==> agent: forget threads and the workspace"
cd "$AGENT"
rm -f checkpoint.sqlite checkpoint.sqlite-wal checkpoint.sqlite-shm
rm -rf .workspace                      # re-cloned from origin at startup

echo
echo "done. Restart uvicorn, then in the browser console:"
echo "  localStorage.removeItem('checkpoint_thread'); location.href = location.pathname"
