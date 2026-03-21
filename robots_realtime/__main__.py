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


def _force_exit(sig, frame):
    """SIGTERM handler: give session.stop() 3 s then hard-kill the process group."""
    import threading
    import time

    def _kill_group():
        time.sleep(3.0)
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except Exception:
            os._exit(1)

    threading.Thread(target=_kill_group, daemon=True).start()


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
        from robots_realtime.config.loader import load_session
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
        from robots_realtime.tui import run_tui
        run_tui(session)

    os._exit(0)


if __name__ == "__main__":
    main()
