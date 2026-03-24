"""CLI entry point: rr-replay — replay a sim episode in Viser."""


def main() -> None:
    import importlib.util
    from pathlib import Path

    import robots_realtime

    scripts_dir = Path(robots_realtime.__file__).parent.parent / "scripts"
    script_path = scripts_dir / "replay_episode.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"replay_episode.py not found at {script_path}. "
            "Run from the robots_realtime repo root or reinstall the package."
        )

    spec = importlib.util.spec_from_file_location("replay_episode", script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.main()
