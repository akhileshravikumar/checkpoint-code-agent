#!/usr/bin/env bash
set -euo pipefail

: "${GH_USER:?Set GH_USER, e.g. export GH_USER=akhileshravikumar}"
: "${CHECKPOINT_PAT:?Set CHECKPOINT_PAT}"

CODE_REPO="checkpoint-code-agent"
SANDBOX_REPO="checkpoint-sandbox"

echo "GitHub user: $GH_USER"
echo

# -------------------------------------------------------------------
# 1. Create checkpoint-code-agent as PUBLIC from the current directory
# -------------------------------------------------------------------

if gh repo view "$GH_USER/$CODE_REPO" >/dev/null 2>&1; then
  echo "✓ $CODE_REPO already exists"
else
  gh repo create "$CODE_REPO" \
    --public \
    --source=. \
    --remote=origin

  echo "✓ created public repo: $CODE_REPO"
fi

# -------------------------------------------------------------------
# 2. Create checkpoint-sandbox as PUBLIC
# -------------------------------------------------------------------

if gh repo view "$GH_USER/$SANDBOX_REPO" >/dev/null 2>&1; then
  echo "✓ $SANDBOX_REPO already exists"
else
  gh repo create "$SANDBOX_REPO" \
    --public \
    --clone

  echo "✓ created public repo: $SANDBOX_REPO"
fi

# -------------------------------------------------------------------
# 3. Make sure checkpoint-sandbox has a main branch
# -------------------------------------------------------------------

if gh api "repos/$GH_USER/$SANDBOX_REPO/branches/main" \
    --silent >/dev/null 2>&1; then

  echo "✓ $SANDBOX_REPO/main exists"

else
  echo "Creating $SANDBOX_REPO/main..."

  TMP_DIR="$(mktemp -d)"

  git -C "$TMP_DIR" init -b main
  git -C "$TMP_DIR" config user.name "$GH_USER"
  git -C "$TMP_DIR" config user.email \
    "$GH_USER@users.noreply.github.com"

  printf '# %s\n' "$SANDBOX_REPO" > "$TMP_DIR/README.md"

  git -C "$TMP_DIR" add README.md
  git -C "$TMP_DIR" commit -m "Initial commit"

  git -C "$TMP_DIR" remote add origin \
    "https://github.com/$GH_USER/$SANDBOX_REPO.git"

  git -C "$TMP_DIR" push -u origin main

  rm -rf "$TMP_DIR"

  echo "✓ created $SANDBOX_REPO/main"
fi

# -------------------------------------------------------------------
# 4. Protect main on both repositories
# -------------------------------------------------------------------

for REPO in "$CODE_REPO" "$SANDBOX_REPO"; do
  echo
  echo "Protecting $REPO/main..."

  # Verify main exists first.
  if ! gh api "repos/$GH_USER/$REPO/branches/main" \
      --silent >/dev/null 2>&1; then
    echo "ERROR: $REPO/main does not exist"
    continue
  fi

  gh api -X PUT \
    "repos/$GH_USER/$REPO/branches/main/protection" \
    -H "Accept: application/vnd.github+json" \
    -F "required_pull_request_reviews[required_approving_review_count]=0" \
    -F "required_pull_request_reviews[dismiss_stale_reviews]=false" \
    -F "enforce_admins=false" \
    -F "required_status_checks=null" \
    -F "restrictions=null" \
    -F "allow_force_pushes=false" \
    -F "allow_deletions=false"

  echo "✓ protected: $REPO/main"
done

# -------------------------------------------------------------------
# 5. Verify repository visibility
# -------------------------------------------------------------------

echo
echo "Repository visibility:"

for REPO in "$CODE_REPO" "$SANDBOX_REPO"; do
  gh repo view "$GH_USER/$REPO" \
    --json nameWithOwner,isPrivate,defaultBranchRef \
    --jq '"\(.nameWithOwner): public=\(.isPrivate | not), branch=\(.defaultBranchRef.name)"'
done

# -------------------------------------------------------------------
# 6. Test CHECKPOINT_PAT
#
# This should return 200 if CHECKPOINT_PAT is allowed to see
# checkpoint-code-agent.
# -------------------------------------------------------------------

echo
echo "Testing CHECKPOINT_PAT against $CODE_REPO..."

CODE_STATUS="$(
  curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $CHECKPOINT_PAT" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$GH_USER/$CODE_REPO"
)"

echo "CHECKPOINT_PAT -> $CODE_REPO: HTTP $CODE_STATUS"

# -------------------------------------------------------------------
# 7. Test against an unrelated repo.
#
# Replace OTHER_REPO with a repository the token MUST NOT be able
# to access. This is the meaningful isolation test.
# -------------------------------------------------------------------

if [[ -n "${OTHER_REPO:-}" ]]; then
  echo
  echo "Testing CHECKPOINT_PAT isolation against $OTHER_REPO..."

  OTHER_STATUS="$(
    curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $CHECKPOINT_PAT" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/$GH_USER/$OTHER_REPO"
  )"

  echo "CHECKPOINT_PAT -> $OTHER_REPO: HTTP $OTHER_STATUS"

  if [[ "$OTHER_STATUS" == "404" ]]; then
    echo "✓ isolation confirmed: token cannot see $OTHER_REPO"
  else
    echo "⚠ expected HTTP 404, got HTTP $OTHER_STATUS"
  fi
fi