"""Rich TUI for live session monitoring.

Renders at 10 Hz.  Keyboard shortcuts work in the same terminal.
"""

from __future__ import annotations

import re
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


_ERROR_PATTERN = re.compile(
    r"\b(ERROR|CRITICAL)\b|Traceback \(most recent call last\)|^\s*\w*(Error|Exception):"
)


def _make_table(session, nodes_with_errors: set[str] | None = None) -> Table:
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("NODE",   style="bold")
    table.add_column("STATUS", justify="center")
    table.add_column("STEP(HZ)",   justify="right")
    table.add_column("PUB(HZ)",    justify="right")
    table.add_column("TOPICS", style="dim")

    errs = nodes_with_errors or set()
    for st in session.node_statuses():
        dot   = Text("● ", style="green") if st.alive else Text("○ ", style="red")
        label = Text("live" if st.alive else "dead", style="green" if st.alive else "red")
        status_cell = Text()
        if st.name in errs:
            status_cell.append("⚠ ", style="bold red")
        status_cell += dot + label

        step_val = f"{st.step_hz:>6.1f}" if st.step_hz > 0 else Text("  ---", style="dim")
        pub_val  = f"{st.pub_hz:>6.1f}"  if st.pub_hz  > 0 else Text("  ---", style="dim")

        topics = ", ".join(t for t in st._timestamps.keys() if not t.startswith("_")) or "—"

        name_cell = Text(st.name, style="bold red" if st.name in errs else "bold")
        table.add_row(name_cell, status_cell, step_val, pub_val, topics)

    return table


def _recording_line(session) -> Text:
    if session.is_paused:
        # Big visible indicator — gates RobotNode commands, so the operator
        # should know at a glance that motors are held.
        t = Text()
        t.append("⏸  PAUSED", style="bold yellow")
        hint = "  (robot commands held — press [space] to resume"
        if getattr(session, "_record_on_unpause", False):
            hint += " + auto-record"
        hint += ")"
        t.append(hint, style="dim")
        return t
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


def _help_line(session=None) -> Text:
    t = Text(justify="right", style="dim")
    t.append("[r]", style="bold white"); t.append(" record  ")
    t.append("[d]", style="bold white"); t.append(" discard  ")
    t.append("[space]", style="bold white")
    if session is not None and session.is_paused:
        t.append(" resume  ")
    else:
        t.append(" pause  ")
    t.append("[q]", style="bold white"); t.append(" quit")
    return t


def _endpoints_text(session) -> Text | None:
    eps = getattr(session, "web_endpoints", [])
    if not eps:
        return None
    t = Text(style="cyan dim")
    t.append("  ".join(eps))
    return t


def _tail_file(path: Path, n: int) -> list[str]:
    """Return the last n lines of a file efficiently."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            chunk = min(4096, size)
            f.seek(-chunk, 2)
            data = f.read()
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def _scan_log_tail(
    log_dir: Path | None, n_lines: int
) -> tuple[list[tuple[str, str, bool]], set[str]]:
    """Tail each *.log and classify lines.

    Returns (tagged_lines, nodes_with_errors):
      tagged_lines: list of (node_name, line, is_error)
      nodes_with_errors: set of node names that emitted an error within the tail
    """
    tagged: list[tuple[str, str, bool]] = []
    err_nodes: set[str] = set()
    if log_dir is None or not log_dir.exists():
        return tagged, err_nodes

    for lf in sorted(log_dir.glob("*.log")):
        node_name = lf.stem
        for line in _tail_file(lf, n_lines):
            is_err = bool(_ERROR_PATTERN.search(line))
            if is_err:
                err_nodes.add(node_name)
            tagged.append((node_name, line, is_err))
    return tagged, err_nodes


def _log_text(tagged_lines: list[tuple[str, str, bool]], n_lines: int) -> Text:
    """Render the tail as a single Text block, coloring error lines red."""
    tail = tagged_lines[-n_lines:]
    out = Text(overflow="fold")
    for i, (node, line, is_err) in enumerate(tail):
        if i > 0:
            out.append("\n")
        out.append(f"[{node}] ", style="dim")
        out.append(line, style="bold red" if is_err else "dim")
    return out


def _render(session, n_log_lines: int = 8) -> Panel:
    log_dir = getattr(session, "log_dir", None)
    tagged_lines, err_nodes = _scan_log_tail(log_dir, n_log_lines)

    node_table = _make_table(session, nodes_with_errors=err_nodes)
    rec_line   = _recording_line(session)
    help_line  = _help_line(session)

    from rich.rule import Rule

    content = Table.grid(expand=True)
    content.add_row(node_table)

    eps_text = _endpoints_text(session)
    if eps_text is not None:
        content.add_row(eps_text)

    content.add_row(Rule(style="dim"))
    content.add_row(Columns([rec_line, help_line], expand=True))

    if log_dir is not None:
        content.add_row(Rule(style="dim"))
        content.add_row(Text(f"  logs: {log_dir}", style="dim"))
        content.add_row(_log_text(tagged_lines, n_lines=n_log_lines))

    return Panel(content, title="[bold]robots_realtime[/bold]", border_style="dim")


# ── Keyboard reader ───────────────────────────────────────────────────────────

def _read_keys(session, stop_event: threading.Event) -> None:
    """Read single keypresses from stdin without echoing.

    Terminal setup (setcbreak) is owned by run_tui, not here.
    """
    while not stop_event.is_set():
        if _stdin_ready():
            ch = sys.stdin.read(1)
            if ch == "r":
                session.toggle_recording()
            elif ch == "d":
                session.end_episode(save=False)
            elif ch == " ":
                session.toggle_pause()
            elif ch == "q":
                stop_event.set()
                break


def _stdin_ready() -> bool:
    import select
    return bool(select.select([sys.stdin], [], [], 0.05)[0])


# ── Entry point ───────────────────────────────────────────────────────────────

def run_tui(session, refresh_hz: float = 10.0) -> None:
    """Block and render the TUI until the user quits or session stops."""
    stop_event = threading.Event()

    # Save terminal settings here in the main thread so the finally block
    # reliably restores them even if a daemon key thread is killed abruptly.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # single-char reads, no echo, keeps OPOST + ISIG intact

    try:
        key_thread = threading.Thread(
            target=_read_keys, args=(session, stop_event), daemon=True
        )
        key_thread.start()

        console = Console()
        try:
            with Live(
                _render(session),
                console=console,
                refresh_per_second=refresh_hz,
                screen=True,
            ) as live:
                while not stop_event.is_set() and not session._stop_event.is_set():
                    # Reserve ~10 lines for the table / chrome above the log panel.
                    n_log = max(4, console.height - 12)
                    live.update(_render(session, n_log_lines=n_log))
                    time.sleep(1.0 / refresh_hz)
        except KeyboardInterrupt:
            pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    session.stop()
