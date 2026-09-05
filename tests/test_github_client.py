"""Exercise the git half of GitHubClient offline, against a local bare 'remote'."""
import subprocess
import pytest
from app import github_client as gc
 
TOKEN = "github_pat_SUPERSECRET_DO_NOT_LEAK"
 
@pytest.fixture
def origin(tmp_path):
    """A bare repo standing in for github.com."""
    bare = tmp_path/"origin.git"
    subprocess.run(["git","init","-q","--bare",str(bare)],check=True)
    seed = tmp_path/"seed"
    subprocess.run(["git","init","-q","-b","main",str(seed)],check=True)
    (seed/"search.py").write_text("def parse_query(q):\n    return q.split()\n")
    for c in (["add","-A"],["-c","user.email=t@t","-c","user.name=t","commit","-qm","init"],
              ["remote","add","origin",str(bare)],["push","-q","-u","origin","main"]):
        subprocess.run(["git","-C",str(seed),*c],check=True)
    return bare
 
@pytest.fixture
def client(tmp_path, origin, monkeypatch):
    """Real GitHubClient with the network bits stubbed out."""
    ws = tmp_path/"workspace"
    class S:
        github_token = TOKEN
        workspace_dir = ws
        repo_slug = "akhileshravikumar/checkpoint-sandbox"
        github_base_branch = "main"
    monkeypatch.setattr(gc, "get_settings", lambda: S())
    monkeypatch.setattr(gc, "Github", lambda **k: type("G",(),{"get_repo":lambda s,x:None,"close":lambda s:None})())
    monkeypatch.setattr(gc, "httpx", type("H",(),{"Client":lambda **k: type("C",(),{"close":lambda s:None})()}))
    c = gc.GitHubClient()
    c.remote = str(origin)          # local bare repo instead of https://...
    return c
 
def test_ensure_workspace_clones(client):
    p = client.ensure_workspace()
    assert (p/"search.py").exists()
    assert client._git("rev-parse","--abbrev-ref","HEAD") == "main"
 
def test_survives_a_week1_workspace_with_no_origin(client, tmp_path):
    """The guide's version dies here: `git remote set-url origin` -> No such remote."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    subprocess.run(["git","init","-q",str(ws)],check=True)
    (ws/"search.py").write_text("leftover\n")
    subprocess.run(["git","-C",str(ws),"add","-A"],check=True)
    subprocess.run(["git","-C",str(ws),"-c","user.email=t@t","-c","user.name=t",
                    "commit","-qm","week1"],check=True)
    p = client.ensure_workspace()                      # must not raise
    assert (p/"search.py").read_text().startswith("def parse_query")
 
def test_hard_reset_discards_leftovers(client):
    p = client.ensure_workspace()
    (p/"search.py").write_text("REJECTED PROPOSAL LEFTOVER\n")
    (p/"junk.py").write_text("x=1\n")
    client.ensure_workspace()
    assert (p/"search.py").read_text().startswith("def parse_query")
    assert not (p/"junk.py").exists()
 
def test_token_never_appears_in_an_error(client, tmp_path):
    """The leak path: the token is in ARGV, and _git interpolates argv into the
    exception. That exception becomes state["error"] -> SQLite -> LangSmith."""
    missing = tmp_path / "not-a-repo"
    missing.mkdir()
    client.workspace = missing                       # make the next git call fail
    url = "https://x-access-token:%s@github.com/o/r.git" % TOKEN
    with pytest.raises(RuntimeError) as e:
        client._git("remote", "set-url", "origin", url)
    msg = str(e.value)
    assert TOKEN not in msg, f"TOKEN LEAKED INTO THE EXCEPTION: {msg}"
    assert "***" in msg
    print("\n   redacted ->", msg[:150])
 
    # and prove the unredacted version really would have leaked it
    naive = f"git remote set-url origin {url} failed: ..."
    assert TOKEN in naive
    print("   the guide's version would have raised:", naive[:110], "...")
 
def test_branch_name_is_deterministic_with_a_seed(client):
    a = client.branch_name("add validation to search.py", seed="d35fa34a")
    b = client.branch_name("add validation to search.py", seed="d35fa34a")
    c = client.branch_name("add validation to search.py", seed="other")
    assert a == b, "same task+thread must give the same branch"
    assert a != c
    assert a.startswith("checkpoint/add-validation-to-search-py-")
    print("\n   branch ->", a)