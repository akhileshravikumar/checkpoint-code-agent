"""GitHub integration: workspace clone, branch, commit, PR, Actions polling.

IDENTITY (ADR-010). The agent authenticates as its own GitHub App, installed on
the sandbox repository only, with four permissions: Contents RW, Pull requests
RW, Actions R, Metadata R. Branch protection on `main` lets repository admins
bypass it (enforce_admins=false) so the owner can push directly; the App is
not an admin, so for it the protection holds. That is what makes "the agent
has never been able to write to main" literally true.

A fine-grained PAT (GITHUB_TOKEN) still works as a fallback, but a PAT acts as
the person who created it, so it inherits that person's admin bypass. The
startup log says which identity is in use.

SECRET HANDLING. Pushing over HTTPS puts the token in the remote URL, which
means it appears in argv and in git's own error output. Every error raised from
this module goes into state["error"], which is checkpointed to SQLite and
traced to LangSmith. So `_redact` is not decoration: without it, one failed
`git remote set-url` writes the token into two durable stores and a
third-party SaaS. Installation tokens are minted at runtime, so callers cannot
pass them to `_redact`; every minted token is registered and always redacted.
"""
from __future__ import annotations
 
import hashlib
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
 
import httpx
from github import Auth, Github, GithubException, GithubIntegration
 
from app.config import get_settings
 
API = "https://api.github.com"
 
 
@dataclass
class PullRequest:
    number: int
    url: str
    head_sha: str
    branch: str
 
 
def checkout_local_branch(repo: str | Path, branch: str) -> bool:
    """Switch `repo` to the local copy of origin/<branch>, if one exists.

    No network and no token: it uses the remote-tracking ref the last push
    left behind. Called on a re-plan so the retry is planned against the
    agent branch (attempt N-1 included), not against main. That matters after
    a server restart, because startup resets the workspace to main.

    Returns False, and changes nothing, when there is no such ref, which is
    always the case for a local test fixture or a Week-1 workspace.
    """
    repo = str(repo)
    ref = f"refs/remotes/origin/{branch}"
    probe = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--verify", "--quiet", ref],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return False
    r = subprocess.run(
        ["git", "-C", repo, "checkout", "-f", "-B", branch, ref],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"could not check out {branch}: {r.stderr.strip()}")
    return True


# Every token this process has minted or loaded. Redacted everywhere, whether
# or not the caller knew about it.
_SECRETS: set[str] = set()


def _register_secret(secret: str) -> None:
    if secret:
        _SECRETS.add(secret)


def _redact(text: str, *secrets: str) -> str:
    """Replace any known secret, and any user:pass in a URL."""
    for sec in {*secrets, *_SECRETS}:
        if sec:
            text = text.replace(sec, "***")
    return re.sub(r"://[^/\s:]+:[^@/\s]+@", "://***:***@", text)


# ---------- credentials ----------

# The installation token scope requested from GitHub. Narrower than or equal
# to what the App was granted; asking for exactly this means a token cannot
# carry a permission added to the App later by mistake.
APP_TOKEN_PERMISSIONS = {
    "contents": "write",
    "pull_requests": "write",
    "actions": "read",
    "metadata": "read",
}


@dataclass
class Credentials:
    token: str
    identity: str            # for logs: who GitHub will see
    git_name: str
    git_email: str
    expires_at: float = float("inf")


_cache: dict[tuple, Credentials] = {}
_cache_lock = threading.Lock()


def _app_configured(s) -> bool:
    return bool(getattr(s, "github_app_id", "") and getattr(s, "github_app_private_key_path", ""))


def _mint_app_credentials(s) -> Credentials:
    """Exchange the App's private key for a one-hour installation token."""
    key_path = Path(s.github_app_private_key_path).expanduser()
    if not key_path.is_file():
        raise RuntimeError(
            f"GITHUB_APP_PRIVATE_KEY_PATH points at {key_path}, which does not exist. "
            "Download a private key from the App settings page (see docs/github-app-setup.md)."
        )
    gi = GithubIntegration(auth=Auth.AppAuth(str(s.github_app_id), key_path.read_text()))
    try:
        installation_id = int(getattr(s, "github_app_installation_id", 0) or 0)
        if not installation_id:
            try:
                installation_id = gi.get_repo_installation(s.github_owner, s.github_repo).id
            except GithubException as exc:
                raise _app_error(exc, s, not_found=(
                    f"The GitHub App is not installed on {s.repo_slug}. Install it on "
                    "that repository only (see docs/github-app-setup.md).")) from None
        try:
            grant = gi.get_access_token(installation_id, permissions=APP_TOKEN_PERMISSIONS)
        except GithubException as exc:
            raise _app_error(exc, s, not_found=(
                f"Installation {installation_id} does not exist for this App. Set "
                "GITHUB_APP_INSTALLATION_ID=0 to look it up."), unprocessable=(
                "GitHub refused an installation token with Contents RW, Pull requests RW, "
                "Actions R and Metadata R. Check the App's permissions, and accept the "
                "updated permissions on the installation if you changed them.")) from None
        slug = gi.get_app().slug
    finally:
        gi.close()

    _register_secret(grant.token)
    bot = f"{slug}[bot]"
    expires = grant.expires_at
    if isinstance(expires, datetime):
        expires = (expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)).timestamp()
    else:
        expires = time.time() + 3600
    return Credentials(
        token=grant.token,
        identity=f"{bot} (GitHub App)",
        git_name=bot,
        git_email=_bot_email(bot, grant.token),
        expires_at=expires,
    )


def _app_error(exc: GithubException, s, *, not_found: str, unprocessable: str = "") -> RuntimeError:
    """Turn GitHub's status codes into the one fix that applies."""
    if exc.status == 401:
        msg = (f"GitHub rejected the App's signed request: GITHUB_APP_ID={s.github_app_id} "
               "is wrong, or the .pem belongs to a different App.")
    elif exc.status == 404:
        msg = not_found
    elif exc.status == 422 and unprocessable:
        msg = unprocessable
    else:
        msg = "GitHub App authentication failed."
    detail = exc.data.get("message") if isinstance(exc.data, dict) else exc.data
    return RuntimeError(_redact(f"{msg} (HTTP {exc.status}: {detail})"))


def _bot_email(bot: str, token: str) -> str:
    """The noreply address GitHub links to the App's bot account.

    With it, commits show the App's avatar and name instead of an unknown
    author. The numeric id is only known by asking; fall back without it.
    """
    try:
        r = httpx.get(f"{API}/users/{bot}", timeout=10.0,
                      headers={"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json"})
        r.raise_for_status()
        return f"{r.json()['id']}+{bot}@users.noreply.github.com"
    except Exception:
        return f"{bot}@users.noreply.github.com"


def credentials(s=None) -> Credentials:
    """The credentials the agent uses: the GitHub App if configured, else the PAT.

    Installation tokens last an hour; one is reused until it has less than ten
    minutes left, so a long watch_ci poll never runs on an expiring token.
    """
    s = s or get_settings()
    if _app_configured(s):
        key = ("app", str(s.github_app_id), str(getattr(s, "github_app_installation_id", "")),
               s.repo_slug)
        with _cache_lock:
            cached = _cache.get(key)
            if cached and cached.expires_at - time.time() > 600:
                return cached
            fresh = _cache[key] = _mint_app_credentials(s)
            return fresh
    if s.github_token:
        _register_secret(s.github_token)
        return Credentials(
            token=s.github_token,
            identity="personal access token (acts as you, including your admin bypass)",
            git_name="checkpoint-agent",
            git_email="checkpoint-agent@users.noreply.github.com",
        )
    raise RuntimeError(
        "No GitHub credentials. Set GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY_PATH "
        "(recommended, see docs/github-app-setup.md), or GITHUB_TOKEN."
    )
 
 
class GitHubClient:
    def __init__(self) -> None:
        s = get_settings()
        self.s = s
        # getattr, not s.checkpoint_offline: the fake settings in
        # tests/test_github_client.py have no such attribute.
        if getattr(s, "checkpoint_offline", False):
            raise RuntimeError(
                "CHECKPOINT_OFFLINE=1: GitHub is disabled. "
                "plan/propose_diff/approve run locally; execute is unavailable."
            )
        creds = credentials(s)
        self.token = creds.token
        self.identity = creds.identity
        self._git_name, self._git_email = creds.git_name, creds.git_email
        # Absolute: every _git call passes -C, and a relative workspace_dir
        # silently follows the process's cwd.
        self.workspace = Path(s.workspace_dir).expanduser().resolve()
        # x-access-token works for both a PAT and an App installation token.
        self.remote = (
            f"https://x-access-token:{self.token}@github.com/{s.repo_slug}.git"
        )
        self.gh = Github(auth=Auth.Token(self.token))
        try:
            self.repo = self.gh.get_repo(s.repo_slug)
        except Exception:
            self.gh.close()
            raise
        self._http = httpx.Client(
            base_url=API,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )
 
    # ---------- local workspace ----------
 
    def ensure_workspace(self) -> Path:
        """Clone or refresh the local working copy the agent patches.
 
        Hard-resets to origin/<base> every time: the agent always plans against
        clean upstream state, never against leftovers from a rejected proposal.
        """
        path, s = self.workspace, self.s
 
        if path.exists() and not (path / ".git").is_dir():
            raise RuntimeError(
                f"{path} exists but is not a git repository. Remove it and retry."
            )
 
        # A .workspace left over from Week 1 is a local `git init` with no
        # origin, so `remote set-url` would fail. Anything that is not a clone
        # of the target repo gets replaced rather than repaired.
        if (path / ".git").is_dir():
            try:
                current = self._git("remote", "get-url", "origin")
            except RuntimeError:
                current = ""
            if s.repo_slug.lower() not in current.lower():
                shutil.rmtree(path)
 
        if not (path / ".git").is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._run(["git", "clone", self.remote, str(path)])
 
        self._git("remote", "set-url", "origin", self.remote)
        # With an App these are the bot's name and noreply address, so the
        # commits are attributed to the agent, not to a person.
        self._git("config", "user.name", self._git_name)
        self._git("config", "user.email", self._git_email)
        self._git("fetch", "origin", s.github_base_branch)
        self._git("checkout", "-f", s.github_base_branch)
        self._git("reset", "--hard", f"origin/{s.github_base_branch}")
        self._git("clean", "-fd")
        return path
 
    def _run(self, cmd: list[str]) -> str:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            shown = _redact(" ".join(cmd), self.token)
            raise RuntimeError(f"{shown} failed: {_redact(r.stderr.strip(), self.token)}")
        return r.stdout.strip()
 
    def _git(self, *args: str) -> str:
        return self._run(["git", "-C", str(self.workspace), *args])
 
    # ---------- branch / commit / PR ----------
 
    @staticmethod
    def branch_name(task: str, seed: str = "") -> str:
        """Deterministic when given a seed — pass the thread_id.
 
        Without a stable seed this is time-based, so a retried `execute` mints a
        second branch and opens a second PR for one approval. See ADR-005: any
        entry point that can run twice needs an identity that does not move.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40]
        basis = f"{task}{seed}" if seed else f"{task}{time.time()}"
        digest = hashlib.sha1(basis.encode()).hexdigest()[:7]
        return f"checkpoint/{slug}-{digest}"
 
    def branch_exists(self, name: str) -> bool:
        """Remote check — the local clone is disposable, the remote is truth."""
        r = self._http.get(f"/repos/{self.s.repo_slug}/branches/{name}")
        return r.status_code == 200
 
    def remote_sha(self, branch: str) -> str:
        r = self._http.get(f"/repos/{self.s.repo_slug}/branches/{branch}")
        r.raise_for_status()
        return r.json()["commit"]["sha"]

    def remote_head(self, branch: str) -> tuple[str, str]:
        """(sha, commit message) of the remote branch tip.

        execute reads the Checkpoint-Attempt trailer from the message to tell
        "this approval was already pushed" apart from "this is a new attempt".
        """
        r = self._http.get(f"/repos/{self.s.repo_slug}/branches/{branch}")
        r.raise_for_status()
        commit = r.json()["commit"]
        return commit["sha"], commit["commit"]["message"]

    def create_branch(self, name: str) -> None:
        self._git("checkout", "-b", name)

    def checkout_remote_branch(self, name: str) -> None:
        """Put the workspace on the tip of an existing remote branch.

        A retry after a CI failure has to commit on top of the previous
        attempt, not on top of main. ensure_workspace() leaves us on main.
        """
        self._git("fetch", "origin", f"+refs/heads/{name}:refs/remotes/origin/{name}")
        self._git("checkout", "-f", "-B", name, f"origin/{name}")
 
    def commit_and_push(self, branch: str, message: str, paths: list[str]) -> str:
        self._git("add", *paths)
        self._git("commit", "-m", message)
        self._git("push", "-u", "origin", branch)
        return self._git("rev-parse", "HEAD")
 
    def find_pr_for_branch(self, branch: str) -> PullRequest | None:
        """An open PR already raised from this branch, if any.
 
        execute can be re-entered — a resumed thread, a crash between push and
        create_pull, a retried approval. Without this the second pass raises
        422 "A pull request already exists" and the node reports failure for
        work that actually succeeded.
        """
        for pr in self.repo.get_pulls(
            state="open", base=self.s.github_base_branch,
            head=f"{self.repo.owner.login}:{branch}",
        ):
            return PullRequest(pr.number, pr.html_url, pr.head.sha, branch)
        return None
 
    def open_pr(self, branch: str, title: str, body: str) -> PullRequest:
        if existing := self.find_pr_for_branch(branch):
            return existing
        pr = self.repo.create_pull(
            title=title, body=body,
            base=self.s.github_base_branch, head=branch,
        )
        return PullRequest(pr.number, pr.html_url, pr.head.sha, branch)
 
    # ---------- Actions ----------
 
    def latest_run_for_sha(self, sha: str) -> dict | None:
        r = self._http.get(
            f"/repos/{self.s.repo_slug}/actions/runs",
            params={"head_sha": sha, "per_page": 1},
        )
        r.raise_for_status()
        runs = r.json().get("workflow_runs", [])
        return runs[0] if runs else None
 
    def failure_log(self, run_id: int, max_chars: int = 4000) -> str:
        """Fetch the log of the first failed job, tail-truncated for the prompt."""
        jobs_r = self._http.get(
            f"/repos/{self.s.repo_slug}/actions/runs/{run_id}/jobs"
        )
        if jobs_r.status_code != 200:
            return ""
        failed = next(
            (j for j in jobs_r.json().get("jobs", [])
             if j.get("conclusion") == "failure"),
            None,
        )
        if not failed:
            return ""
        # Job logs are plain text behind a redirect to a pre-signed blob URL.
        # (Run-level logs would be a zip — that endpoint is deliberately unused.)
        r = self._http.get(
            f"/repos/{self.s.repo_slug}/actions/jobs/{failed['id']}/logs",
            follow_redirects=True,
        )
        if r.status_code != 200:
            return ""
        return r.text[-max_chars:]
 
    def close(self) -> None:
        self._http.close()
        self.gh.close()