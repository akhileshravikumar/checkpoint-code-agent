"""Terminal harness for the graph. No web server involved."""
import uuid

import typer
from langgraph.types import Command
from rich.console import Console
from rich.syntax import Syntax

from app.graph import build_graph, make_checkpointer
from app.tracing import configure_tracing

cli = typer.Typer()
console = Console()


def _show(payload: dict) -> None:
    console.rule("[bold]Proposed change")
    console.print(f"[bold]{payload['plan']['summary']}[/bold]")
    for i, s in enumerate(payload["plan"]["steps"], 1):
        console.print(f"  {i}. {s}")
    console.print(Syntax(payload["diff"], "diff", theme="ansi_dark"))
    console.print(f"[dim]+{payload['stats']['additions']} "
                  f"-{payload['stats']['deletions']}[/dim]")


@cli.command()
def run(task: str, repo: str = ".", thread: str = ""):
    configure_tracing()
    thread_id = thread or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(make_checkpointer())

    console.print(f"[dim]thread_id: {thread_id}[/dim]")
    result = graph.invoke(
        {"task": task, "repo_path": repo, "retry_count": 0}, config=config
    )

    while interrupts := result.get("__interrupt__"):
        _show(interrupts[0].value)
        choice = typer.prompt("approve / reject / edit").strip().lower()
        note = typer.prompt("note") if choice.startswith("e") else ""
        decision = {"a": "approved", "r": "rejected", "e": "edit_requested"}[choice[0]]
        result = graph.invoke(
            Command(resume={"decision": decision, "note": note}), config=config
        )

    if err := result.get("error"):
        console.print(f"[red]{err}[/red]")
    else:
        console.print(f"[green]done[/green] ({result.get('approval_status')})")


@cli.command()
def resume(thread: str, repo: str = "."):
    """Resume a thread that was interrupted, e.g. after the process was killed."""
    configure_tracing()
    config = {"configurable": {"thread_id": thread}}
    graph = build_graph(make_checkpointer())
    snap = graph.get_state(config)
    console.print(f"next node: {snap.next}")
    if snap.interrupts:
        _show(snap.interrupts[0].value)
    console.print(
    "[dim]resume with: python -m app.cli run '' --thread "
    + thread
    + "[/dim]"
)


if __name__ == "__main__":
    cli()