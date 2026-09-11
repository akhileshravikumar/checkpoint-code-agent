"""ADR-010: the agent authenticates as its own GitHub App.

No network: GithubIntegration and the bot-id lookup are faked. The private
key is a real RSA key, so JWT signing itself is exercised.
"""
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from github import Auth, GithubException

from app import github_client as gc

INSTALL_TOKEN = "ghs_INSTALLATIONTOKEN_DO_NOT_LEAK"


@pytest.fixture
def pem(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "agent.pem"
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    return path


def _settings(pem="", app_id="123456", token="", installation=0, workspace="/tmp/ws"):
    class S:
        github_app_id = app_id
        github_app_private_key_path = str(pem)
        github_app_installation_id = installation
        github_token = token
        github_owner, github_repo = "akhileshravikumar", "checkpoint-sandbox"
        repo_slug = "akhileshravikumar/checkpoint-sandbox"
        github_base_branch = "main"
        workspace_dir = workspace
        checkpoint_offline = False
    return S()


@pytest.fixture
def fake_github(monkeypatch):
    """A GithubIntegration that records what it was asked for."""
    calls = {"minted": 0, "jwt": None, "permissions": None, "expires_in": 3600}

    class Grant:
        def __init__(self):
            self.token = INSTALL_TOKEN
            self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=calls["expires_in"])

    class GI:
        def __init__(self, auth):
            calls["jwt"] = auth.create_jwt()       # real signing with the real key

        def get_repo_installation(self, owner, repo):
            calls["looked_up"] = f"{owner}/{repo}"
            return type("I", (), {"id": 99})()

        def get_access_token(self, installation_id, permissions=None):
            calls["minted"] += 1
            calls["installation"] = installation_id
            calls["permissions"] = permissions
            return Grant()

        def get_app(self):
            return type("A", (), {"slug": "checkpoint-agent"})()

        def close(self):
            pass

    monkeypatch.setattr(gc, "GithubIntegration", GI)
    monkeypatch.setattr(gc, "_bot_email",
                        lambda bot, token: f"4242+{bot}@users.noreply.github.com")
    gc._cache.clear()
    yield calls
    gc._cache.clear()


def test_app_credentials_are_an_installation_token_for_the_bot(pem, fake_github):
    c = gc.credentials(_settings(pem))
    assert c.token == INSTALL_TOKEN
    assert c.identity == "checkpoint-agent[bot] (GitHub App)"
    assert c.git_name == "checkpoint-agent[bot]"
    assert c.git_email == "4242+checkpoint-agent[bot]@users.noreply.github.com"
    assert fake_github["looked_up"] == "akhileshravikumar/checkpoint-sandbox"
    assert fake_github["installation"] == 99
    assert fake_github["jwt"].count(".") == 2, "a signed JWT was produced from the key"


def test_the_token_asks_for_exactly_four_permissions(pem, fake_github):
    gc.credentials(_settings(pem))
    assert fake_github["permissions"] == {
        "contents": "write", "pull_requests": "write",
        "actions": "read", "metadata": "read",
    }


def test_an_explicit_installation_id_skips_the_lookup(pem, fake_github):
    gc.credentials(_settings(pem, installation=7))
    assert fake_github["installation"] == 7
    assert "looked_up" not in fake_github


def test_a_token_is_reused_until_it_nears_expiry(pem, fake_github):
    s = _settings(pem)
    gc.credentials(s)
    gc.credentials(s)
    assert fake_github["minted"] == 1, "reused while it has more than 10 minutes left"
    fake_github["expires_in"] = 300
    gc._cache.clear()
    gc.credentials(s)
    gc.credentials(s)
    assert fake_github["minted"] == 3, "a token with <10 minutes left is replaced"


def test_minted_tokens_are_always_redacted(pem, fake_github):
    gc.credentials(_settings(pem))
    msg = f"push failed: remote said {INSTALL_TOKEN} is bad"
    assert INSTALL_TOKEN not in gc._redact(msg), "callers don't know the token; redact anyway"
    assert INSTALL_TOKEN not in gc._redact(msg, "")


def test_the_app_is_preferred_over_a_pat(pem, fake_github):
    c = gc.credentials(_settings(pem, token="github_pat_FALLBACK"))
    assert c.token == INSTALL_TOKEN


def test_the_pat_fallback_says_it_acts_as_you():
    c = gc.credentials(_settings(app_id="", token="github_pat_FALLBACK"))
    assert c.token == "github_pat_FALLBACK"
    assert "acts as you" in c.identity


def test_no_credentials_is_a_clear_error():
    with pytest.raises(RuntimeError, match="GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY_PATH"):
        gc.credentials(_settings(app_id="", token=""))


def test_a_missing_key_file_names_the_path(tmp_path, fake_github):
    with pytest.raises(RuntimeError, match="does not exist"):
        gc.credentials(_settings(tmp_path / "nope.pem"))


@pytest.mark.parametrize("status, expected", [
    (404, "not installed on akhileshravikumar/checkpoint-sandbox"),
    (401, "GITHUB_APP_ID=123456 is wrong, or the .pem belongs to a different App"),
    (500, "GitHub App authentication failed. (HTTP 500"),
])
def test_installation_lookup_failures_name_the_fix(pem, fake_github, monkeypatch, status, expected):
    def fail(self, owner, repo):
        raise GithubException(status, {"message": "nope"}, None)
    monkeypatch.setattr(gc.GithubIntegration, "get_repo_installation", fail)
    with pytest.raises(RuntimeError, match=expected.replace("(", r"\(").replace(".", r"\.")):
        gc.credentials(_settings(pem))


def test_a_permission_mismatch_is_named(pem, fake_github, monkeypatch):
    def refuse(self, installation_id, permissions=None):
        raise GithubException(422, {"message": "The permissions requested are not granted"}, None)
    monkeypatch.setattr(gc.GithubIntegration, "get_access_token", refuse)
    with pytest.raises(RuntimeError, match="Check the App's permissions"):
        gc.credentials(_settings(pem))


def test_a_real_key_signs_a_real_jwt(pem):
    """No fakes: PyGithub + PyJWT + cryptography load the .pem and sign."""
    jwt = Auth.AppAuth("123456", pem.read_text()).create_jwt()
    header, payload, signature = jwt.split(".")
    assert header and payload and signature


def test_the_client_commits_as_the_bot(pem, fake_github, tmp_path, monkeypatch):
    """ensure_workspace configures git with the App's bot identity."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    (seed / "search.py").write_text("x = 1\n")
    for c in (["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
              ["remote", "add", "origin", str(bare)], ["push", "-q", "-u", "origin", "main"]):
        subprocess.run(["git", "-C", str(seed), *c], check=True)

    s = _settings(pem, workspace=tmp_path / "ws")
    monkeypatch.setattr(gc, "get_settings", lambda: s)
    monkeypatch.setattr(gc, "Github", lambda **k: type(
        "G", (), {"get_repo": lambda self, x: None, "close": lambda self: None})())
    monkeypatch.setattr(gc.httpx, "Client", lambda **k: type(
        "C", (), {"close": lambda self: None})())
    client = gc.GitHubClient()
    assert client.identity.startswith("checkpoint-agent[bot]")
    assert INSTALL_TOKEN in client.remote
    client.remote = str(bare)
    client.ensure_workspace()
    assert client._git("config", "user.name") == "checkpoint-agent[bot]"
    assert client._git("config", "user.email").endswith("+checkpoint-agent[bot]@users.noreply.github.com")
