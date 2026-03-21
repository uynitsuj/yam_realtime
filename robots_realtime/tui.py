"""Rich TUI for live session monitoring.

Renders at 10 Hz.  Keyboard shortcuts work in the same terminal.
"""

from __future__ import annotations

import sys
import termios
import threading
import time
import tty
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _make_table(session) -> Table:
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("NODE",   style="bold")
    table.add_column("STATUS", justify="center")
    table.add_column("HZ",     justify="right")
    table.add_column("TOPICS", style="dim")

    for st in session.node_statuses():
        dot   = Text("● ", style="green") if st.alive else Text("○ ", style="red")
        label = Text("live" if st.alive else "dead", style="green" if st.alive else "red")
        status_cell = dot + label

        hz_val = f"{st.hz:>6.1f}" if st.hz > 0 else Text("  ---", style="dim")

        # topic suffixes from the node definition
        from robots_realtime.nodes.base import Node  # avoid circular at module level
        topics = ", ".join(st._timestamps.keys()) or "—"

        table.add_row(st.name, status_cell, hz_val, topics)

    return table


def _recording_line(session) -> Text:
    if not session.is_recording:
        return Text("○  idle", style="dim")

    start = session.episode_start_time or time.time()
    elapsed = int(time.time() - start)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    clock  = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    t = Text()
    t.append("● ", style="bold red")
    t.append(clock, style="bold white")
    t.append(f"  {session.save_root}", style="dim")
    return t


def _help_line() -> Text:
    t = Text(justify="right", style="dim")
    t.append("[r]", style="bold white"); t.append(" record  ")
    t.append("[d]", style="bold white"); t.append(" discard  ")
    t.append("[q]", style="bold white"); t.append(" quit")
    return t


def _render(session) -> Panel:
    node_table = _make_table(session)
    rec_line   = _recording_line(session)
    help_line  = _help_line()

    from rich.rule import Rule
    from rich import box as rbox

    content = Table.grid(expand=True)
    content.add_row(node_table)
    content.add_row(Rule(style="dim"))
    content.add_row(Columns([rec_line, help_line], expand=True))

    return Panel(content, title="[bold]robots_realtime[/bold]", border_style="dim")


# ── Keyboard reader ───────────────────────────────────────────────────────────

def _read_keys(session, stop_event: threading.Event) -> None:
    """Read single keypresses from stdin without echoing."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            if _stdin_ready():
                ch = sys.stdin.read(1)
                if ch == "r":
                    session.toggle_recording()
                elif ch == "d":
                    session.end_episode(save=False)
                elif ch in ("q", "\x03"):  # q or Ctrl-C
                    stop_event.set()
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _stdin_ready() -> bool:
    import select
    return bool(select.select([sys.stdin], [], [], 0.05)[0])


# ── Entry point ───────────────────────────────────────────────────────────────

def run_tui(session, refresh_hz: float = 10.0) -> None:
    """Block and render the TUI until the user quits or session stops."""
    stop_event = threading.Event()

    key_thread = threading.Thread(
        target=_read_keys, args=(session, stop_event), daemon=True
    )
    key_thread.start()

    console = Console()
    with Live(
        _render(session),
        console=console,
        refresh_per_second=refresh_hz,
        screen=True,
    ) as live:
        while not stop_event.is_set() and not session._stop_event.is_set():
            live.update(_render(session))
            time.sleep(1.0 / refresh_hz)

    session.stop()
