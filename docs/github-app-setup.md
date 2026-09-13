# Give the agent its own identity: a GitHub App (ADR-010)

**Why.** The agent used a fine-grained PAT created on your account. A PAT acts
*as you*, so GitHub cannot tell its pushes from yours, and it inherits your admin
bypass on `main`. With a GitHub App the agent is `checkpoint-agent-…[bot]`, a
separate, non-admin identity: you can push to `main` directly, the agent cannot,
and "the agent has never been able to write to `main`" becomes literally true.

Time: about 20 minutes. Code support is already in `app/github_client.py`.

---

## 1. Create the App

GitHub → your avatar → **Settings → Developer settings → GitHub Apps → New GitHub App**

| Field | Value |
|---|---|
| GitHub App name | `checkpoint-agent-<your-username>` (globally unique; becomes the bot name) |
| Homepage URL | `https://github.com/<you>/checkpoint-code-agent` |
| Webhook → Active | **unchecked** (the agent polls; it needs no webhook) |
| Repository permissions → Contents | **Read and write** |
| Repository permissions → Pull requests | **Read and write** |
| Repository permissions → Actions | **Read-only** (failed-job logs for self-healing) |
| Repository permissions → Metadata | Read-only (mandatory, preselected) |
| Everything else | **No access**. In particular *not* Administration and *not* Workflows |
| Where can this GitHub App be installed? | **Only on this account** |

**Create GitHub App.** The code asks GitHub for a token with exactly these four
permissions, so a permission added to the App later by mistake never reaches a token.

## 2. App ID and private key

On the App's page:

1. Copy the **App ID** (a number near the top).
2. Scroll to **Private keys → Generate a private key**. A `.pem` file downloads.

Move it **outside the repository** and lock it down (from WSL):

```bash
mkdir -p ~/.config/checkpoint && chmod 700 ~/.config/checkpoint
mv /mnt/c/Users/<you>/Downloads/checkpoint-agent-*.private-key.pem \
   ~/.config/checkpoint/checkpoint-agent.pem
chmod 600 ~/.config/checkpoint/checkpoint-agent.pem
```

The key is the App's password. `*.pem` is gitignored as a backstop, and
`doctor.sh` fails if the key is inside the repo, but keep it out anyway.

## 3. Install it on the sandbox only

App page → **Install App** → your account → **Only select repositories** →
`checkpoint-sandbox` → **Install**. Do not install it on `checkpoint-code-agent`.

(The installation id in the resulting URL is optional; the code looks it up.)

## 4. Point the agent at it

`.env`:

```bash
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY_PATH=~/.config/checkpoint/checkpoint-agent.pem
GITHUB_APP_INSTALLATION_ID=0
GITHUB_TOKEN=                  # empty: the App replaces the PAT
```

## 5. Branch protection: you bypass, the App cannot

`main` keeps its protection (PR required, required `test` check). Admins may
bypass it only while `enforce_admins` is off. Check:

```bash
gh api repos/$GITHUB_OWNER/checkpoint-sandbox/branches/main/protection \
  --jq '.enforce_admins.enabled'          # must print: false
# if it prints true:
gh api -X DELETE repos/$GITHUB_OWNER/checkpoint-sandbox/branches/main/protection/enforce_admins
```

The App is not an admin, so for it the protection holds regardless.

## 6. CI only for the agent's commits

In **your** sandbox clone (`~/projects/checkpoint-sandbox`, never `.workspace`),
`.github/workflows/ci.yml`:

```yaml
on:
  push:
    branches: ["checkpoint/**"]   # only the agent's branches
  workflow_dispatch:              # run it by hand from the Actions tab
```

(the `jobs:` section stays as it is). `watch_ci` finds this run by the commit it
pushed, and the required `test` check on an agent PR is satisfied by it. Your
own pushes to `main` no longer start CI. Commit and push this yourself; the App
cannot, as it has no Workflows permission and the agent is barred from `.github/`.

## 7. Push as yourself

Your own pushes must use **your** login, not a token:

```bash
gh auth login          # once
gh auth setup-git      # git uses your gh login for github.com
```

Push from your own clone. `.workspace` belongs to the agent: its remote URL
carries the agent's token, and it is hard-reset on every run.

## 8. Verify

```bash
#chmod +x ./scripts/doctor.sh
./scripts/doctor.sh
#   ok    agent identity: checkpoint-agent-<you>[bot] (GitHub App)
#   ok    you (admin) can push to main directly

# chmod +x ./scripts/prove-agent-cannot-push-main.sh
./scripts/prove-agent-cannot-push-main.sh
#   ==> pushing as: checkpoint-agent-<you>[bot] (GitHub App)
#   OK: rejected. GitHub said: ... protected branch ...
```

Restart uvicorn. The startup log prints `[startup] GitHub identity: …[bot] (GitHub App)`.
Run a task and approve it: the branch, the commit and the PR are authored by the bot.

Screenshot the proof script's output. It is the evidence for the claim.

## 9. Delete the PAT

GitHub → Settings → Developer settings → **Fine-grained tokens** → delete the
Checkpoint token, and make sure `GITHUB_TOKEN` in `.env` is empty. While it
exists it still carries your admin bypass.

---

## Troubleshooting

| Message | Cause |
|---|---|
| `GITHUB_APP_PRIVATE_KEY_PATH points at …, which does not exist` | Wrong path, or `~` not where you think. Use the absolute path to check. |
| `The GitHub App is not installed on <owner>/checkpoint-sandbox` | Step 3 missing, or installed on a different repository. |
| `GitHub refused an installation token with Contents RW, …` | A permission from step 1 is missing. If you changed permissions after installing, accept the new permissions on the installation page. |
| `401` / `A JSON web token could not be decoded` | Wrong App ID, or the `.pem` belongs to a different App. |
| Commits show an unknown author | The bot-id lookup failed; cosmetic only. The pusher is still the App. |
| `prove-agent-cannot-push-main.sh` prints **FAIL** | The agent is still using the PAT (`doctor.sh` warns about this), or the App was given admin rights. |
