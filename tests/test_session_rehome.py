from pathlib import Path

from robots_realtime.runtime.session import Session


class FakePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def publish(self, suffix: str, data: dict, **kwargs: object) -> None:
        self.calls.append((suffix, data, kwargs))


def test_save_and_rehome_saves_before_publishing_request() -> None:
    session = Session.__new__(Session)
    publisher = FakePublisher()
    events = []
    session._control_publisher = publisher
    session.end_episode = lambda save=True: events.append(("save", save)) or Path("/tmp/episode")

    result = session.save_and_rehome()

    assert result == Path("/tmp/episode")
    assert events == [("save", True)]
    assert publisher.calls[0][0:2] == ("rehome", {"request": True})
    assert publisher.calls[0][2]["record"] is False


def test_save_and_rehome_requires_running_control_publisher() -> None:
    session = Session.__new__(Session)
    session._control_publisher = None
    session.end_episode = lambda save=True: None

    try:
        session.save_and_rehome()
    except RuntimeError as exc:
        assert "control publisher" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
