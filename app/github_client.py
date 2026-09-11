"""GitHub integration: workspace clone, branch, commit, PR, Actions polling.
 
Auth is a fine-grained PAT scoped to exactly one repository with four
permissions (ARCHITECTURE.md §7). If something here needs a broader scope,
the bug is in this file, not in the token.
 
SECRET HANDLING. Pushing with a fine-grained PAT over HTTPS means the token
appears in the remote URL, which means it appears in argv and in git's own
error output. Every error raised from this module goes into state["error"],
which is checkpointed to SQLite and traced to LangSmith. So `_redact` is not
decoration: without it, one failed `git remote set-url` writes the PAT into
two durable stores and a third-party SaaS.
"""
from __future__ import annotations
 
import hashlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
 
import httpx
from github import Auth, Github
 
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


def _redact(text: str, *secrets: str) -> str:
    """Replace any occurrence of a secret, and any user:pass in a URL."""
    for sec in secrets:
        if sec:
            text = text.replace(sec, "***")
    return re.sub(r"://[^/\s:]+:[^@/\s]+@", "://***:***@", text)
 
 
class GitHubClient:
    def __init__(self) -> None:
        s = get_settings()
        self.s = s
        if not s.github_token:
            raise RuntimeError("GITHUB_TOKEN is empty — check .env")
        # Absolute: every _git call passes -C, and a relative workspace_dir
        # silently follows the process's cwd.
        self.workspace = Path(s.workspace_dir).expanduser().resolve()
        self.remote = (
            f"https://x-access-token:{s.github_token}@github.com/{s.repo_slug}.git"
        )
        self.gh = Github(auth=Auth.Token(s.github_token))
        try:
            self.repo = self.gh.get_repo(s.repo_slug)
        except Exception:
            self.gh.close()
            raise
        self._http = httpx.Client(
            base_url=API,
            headers={
                "Authorization": f"Bearer {s.github_token}",
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
        self._git("config", "user.name", "checkpoint-agent")
        self._git("config", "user.email",
                  "checkpoint-agent@users.noreply.github.com")
        self._git("fetch", "origin", s.github_base_branch)
        self._git("checkout", "-f", s.github_base_branch)
        self._git("reset", "--hard", f"origin/{s.github_base_branch}")
        self._git("clean", "-fd")
        return path
 
    def _run(self, cmd: list[str]) -> str:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            shown = _redact(" ".join(cmd), self.s.github_token)
            raise RuntimeError(
                f"{shown} failed: {_redact(r.stderr.strip(), self.s.github_token)}"
            )
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