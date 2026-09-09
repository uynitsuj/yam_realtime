"""CLI entry point.

Usage:
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml --no-tui
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml --save-root /data/rec

    # Legacy Python module path (backward compatibility):
    uv run -m robots_realtime configs.sessions.yam_sim_dummy  --no-tui
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys


_MAIN_PID = os.getpid()


def _descendant_pids(root: int) -> list[int]:
    """All live descendants of `root` (children first), via /proc — no psutil dependency."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out: list[int] = []
    stack = [root]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def _force_exit(sig, frame):
    """SIGTERM handler for the *main* rr-session process: give session.stop() 3 s, then hard-kill
    our own descendants (node / broker subprocesses) and exit.

    This handler is installed at import time, so forked children (multiprocessing nodes, the
    message-bus broker) inherit it. A child must NOT run the group kill: the broker receives a
    SIGTERM from `MessageBus.stop()` during a normal quit, and the old `os.killpg(getpgid(0))`
    then SIGKILLed the whole process group — including the shell / tmux pane that launched the
    session. Children just exit; only descendants of the main process are ever killed, never
    ancestors such as the launching shell.
    """
    if os.getpid() != _MAIN_PID:
        os._exit(0)

    import threading
    import time

    def _kill_descendants():
        time.sleep(3.0)
        for pid in _descendant_pids(_MAIN_PID):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        os._exit(1)

    threading.Thread(target=_kill_descendants, daemon=True).start()


signal.signal(signal.SIGTERM, _force_exit)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="robots_realtime",
        description="Launch a robots_realtime session.",
    )
    parser.add_argument(
        "session",
        help=(
            "Path to a YAML session config file (e.g. configs/sessions/yam_sim_dummy.yaml), "
            "or a dotted Python module path containing make_session() "
            "(e.g. configs.sessions.yam_sim_dummy)."
        ),
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help="Override the session's default save_root for recordings.",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the Rich TUI and just block until Ctrl-C.",
    )
    args = parser.parse_args()

    session_arg: str = args.session

    # Determine whether this is a YAML file path or a Python module path
    is_yaml = session_arg.endswith(".yaml") or session_arg.endswith(".yml")
    is_file = os.path.exists(session_arg)

    if is_yaml or is_file:
        # YAML file path
        from robots_realtime.runtime.config import load_session
        try:
            session = load_session(session_arg)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading session config '{session_arg}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Legacy Python module path
        try:
            mod = importlib.import_module(session_arg)
        except ModuleNotFoundError as e:
            print(f"Error: could not import '{session_arg}': {e}", file=sys.stderr)
            sys.exit(1)

        if not hasattr(mod, "make_session"):
            print(
                f"Error: '{session_arg}' has no make_session() function.",
                file=sys.stderr,
            )
            sys.exit(1)

        session = mod.make_session()

    # Allow save-root override
    if args.save_root:
        from pathlib import Path
        session._save_root = Path(args.save_root)

    session.start()

    if args.no_tui:
        print(f"Session running. Ctrl-C to stop.  Recordings → {session.save_root}")
        session.wait()
    else:
        from robots_realtime.runtime.tui import run_tui
        run_tui(session)

    os._exit(0)


if __name__ == "__main__":
    main()
