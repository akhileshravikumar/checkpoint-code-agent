"""Terminal harness for the graph. No web server involved."""
import uuid

import typer
from langgraph.types import Command
from rich.console import Console
from rich.syntax import Syntax

from app.graph import build_graph, make_checkpointer
from app.state import new_task
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


def _ask() -> dict:
    while True:
        choice = typer.prompt("approve / reject / edit").strip().lower()
        if choice[:1] and choice[0] in "are":
            break
        console.print("[yellow]Answer approve, reject or edit.[/yellow]")
    decision = {"a": "approved", "r": "rejected", "e": "edit_requested"}[choice[0]]
    note = typer.prompt("note") if decision == "edit_requested" else ""
    return {"decision": decision, "note": note}


def _drive(graph, config, first) -> dict:
    """Run the approval loop until the graph stops interrupting."""
    result = first
    while interrupts := result.get("__interrupt__"):
        _show(interrupts[0].value)
        result = graph.invoke(Command(resume=_ask()), config=config)
    return result


def _report(result: dict) -> None:
    if err := result.get("error"):
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]done[/green] ({result.get('approval_status')})")


@cli.command()
def run(task: str, repo: str = ".", thread: str = ""):
    """Start a new thread. To continue an existing one, use `resume`."""
    configure_tracing()
    thread_id = thread or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(make_checkpointer())

    # Passing an input dict to invoke() writes it into state and re-enters at
    # START — it does NOT resume a pending interrupt. Doing that on a paused
    # thread silently overwrites `task` and `repo_path` with these arguments
    # and replans from scratch. Refuse, and point at the command that works.
    snap = graph.get_state(config)
    if snap.next:
        console.print(
            f"[red]Thread {thread_id} is already in progress "
            f"(paused before: {', '.join(snap.next)}).[/red]\n"
            f"[dim]Continue it with: python -m app.cli resume --thread {thread_id}[/dim]"
        )
        raise typer.Exit(1)

    console.print(f"[dim]thread_id: {thread_id}[/dim]")
    result = graph.invoke(new_task(task, repo), config=config)
    _report(_drive(graph, config, result))


@cli.command()
def resume(thread: str):
    """Resume a thread that was interrupted, e.g. after the process was killed.

    Nothing is re-planned: the graph picks up inside await_approval with the
    state exactly as it was checkpointed, including task and repo_path.
    """
    configure_tracing()
    config = {"configurable": {"thread_id": thread}}
    graph = build_graph(make_checkpointer())

    snap = graph.get_state(config)
    if not snap.created_at:
        console.print(f"[red]No checkpoint found for thread {thread}.[/red]")
        raise typer.Exit(1)
    if not snap.next:
        console.print(f"[yellow]Thread {thread} already finished.[/yellow]")
        _report(snap.values)
        return

    console.print(f"[dim]task: {snap.values.get('task')!r}[/dim]")
    console.print(f"[dim]repo: {snap.values.get('repo_path')!r}[/dim]")
    console.print(f"[dim]next node: {', '.join(snap.next)}[/dim]")

    if snap.interrupts:
        _show(snap.interrupts[0].value)
        result = graph.invoke(Command(resume=_ask()), config=config)
    else:
        # Paused without an interrupt (crash mid-node). None re-runs the
        # pending task from its checkpointed input.
        result = graph.invoke(None, config=config)

    _report(_drive(graph, config, result))


@cli.command()
def status(thread: str):
    """Show where a thread is parked, without touching it."""
    config = {"configurable": {"thread_id": thread}}
    snap = build_graph(make_checkpointer()).get_state(config)
    if not snap.created_at:
        console.print(f"[red]No checkpoint found for thread {thread}.[/red]")
        raise typer.Exit(1)
    console.print(f"task        : {snap.values.get('task')!r}")
    console.print(f"repo_path   : {snap.values.get('repo_path')!r}")
    console.print(f"next        : {', '.join(snap.next) or '(finished)'}")
    console.print(f"retry_count : {snap.values.get('retry_count', 0)}")
    console.print(f"checkpointed: {snap.created_at}")
    if err := snap.values.get("error"):
        console.print(f"[red]error       : {err}[/red]")


if __name__ == "__main__":
    cli()
