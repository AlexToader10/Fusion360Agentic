"""Offline tests for the Streamlit background WebSocket reader and the backend event log.

No sockets are opened. `frontend_demo.stream.websocket` is replaced with a scripted fake module, and the reconnect
sleep is replaced so the tests stay fast. Threads are checked with join(timeout=...), never with long sleeps.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

import frontend_demo.stream
from agent_backend.events import EventLog
from frontend_demo.stream import EventStream

JOIN_TIMEOUT = 2


class _FakeTimeout(Exception):
    """Stands in for websocket.WebSocketTimeoutException."""


class _FakeConnection:
    """Plays back a script. Items are JSON strings, exceptions to raise, or nothing left to play.

    When the script runs out, recv() waits briefly and then raises a timeout, the way an idle socket would.
    """

    def __init__(self, *script: Any) -> None:
        self.script: deque[Any] = deque(script)
        self.timeout: float | None = None
        self.closed = False
        self._idle = threading.Event()

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recv(self) -> str:
        if not self.script:
            self._idle.wait(0.01)
            raise _FakeTimeout()
        item = self.script.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class _FakeWebSocketModule:
    """Stands in for the `websocket` module. Each create_connection call consumes one attempt.

    An attempt is a _FakeConnection, or an exception that create_connection raises. Once the attempts are used up,
    it returns an idle connection, so a stray reconnect cannot spin through errors forever.
    """

    WebSocketTimeoutException = _FakeTimeout

    def __init__(self, *attempts: Any) -> None:
        self.attempts: deque[Any] = deque(attempts)
        self.urls: list[str] = []

    def create_connection(self, url: str, timeout: float | None = None) -> _FakeConnection:
        self.urls.append(url)
        attempt = self.attempts.popleft() if self.attempts else _FakeConnection()
        if isinstance(attempt, BaseException):
            raise attempt
        return attempt


@pytest.fixture(autouse=True)
def _no_reconnect_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontend_demo.stream, "time", SimpleNamespace(sleep=lambda seconds: None))


def _install(monkeypatch: pytest.MonkeyPatch, *attempts: Any) -> _FakeWebSocketModule:
    fake = _FakeWebSocketModule(*attempts)
    monkeypatch.setattr(frontend_demo.stream, "websocket", fake)
    return fake


def _msg(event: dict[str, Any]) -> str:
    return json.dumps(event)


def _run_state(status: str, seq: int) -> str:
    return _msg({"kind": "run_state", "seq": seq, "last_seq": seq, "run": {"status": status}})


def _url() -> Callable[[int], str]:
    run_id = uuid4()
    return lambda after: f"ws://testserver/runs/{run_id}/events?after={after}"


def _join(stream: EventStream) -> None:
    stream.thread.join(timeout=JOIN_TIMEOUT)
    assert not stream.alive, "reader thread did not exit in time"


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [event["kind"] for event in events]


def test_missing_websocket_library_reports_stream_error_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontend_demo.stream, "websocket", None)
    stream = EventStream(_url()).start()

    _join(stream)

    events = stream.drain()
    assert _kinds(events) == ["stream_error"]
    assert "websocket-client is not installed" in events[0]["message"]


def test_scripted_events_arrive_in_order_and_terminal_state_stops_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    url = _url()
    fake = _install(
        monkeypatch,
        _FakeConnection(
            _msg({"kind": "status", "seq": 1, "stage": "assumptions", "message": "Starting"}),
            _msg({"kind": "token", "seq": 2, "stage": "assumptions", "message": "Hel"}),
            _run_state("completed", seq=3),
        ),
    )
    stream = EventStream(url).start()

    _join(stream)

    events = stream.drain()
    assert _kinds(events) == ["status", "token", "run_state"]
    assert [event.get("seq") for event in events] == [1, 2, 3]
    assert stream.after == 3
    assert stream.stop_event.is_set()
    assert fake.urls == [url(0)]


def test_timeout_during_recv_is_swallowed_and_reading_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _FakeConnection(
            _FakeTimeout(),
            _FakeTimeout(),
            _msg({"kind": "token", "seq": 1, "message": "after the quiet period"}),
            _run_state("completed", seq=2),
        ),
    )
    stream = EventStream(_url()).start()

    _join(stream)

    events = stream.drain()
    assert _kinds(events) == ["token", "run_state"]
    assert "stream_error" not in _kinds(events)


def test_connection_error_reconnects_and_delivers_later_events(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(
        monkeypatch,
        _FakeConnection(ConnectionResetError("socket dropped")),
        _FakeConnection(
            _msg({"kind": "token", "seq": 1, "message": "resumed"}),
            _run_state("completed", seq=2),
        ),
    )
    stream = EventStream(_url()).start()

    _join(stream)

    events = stream.drain()
    assert _kinds(events) == ["stream_error", "token", "run_state"]
    assert events[0]["message"] == "socket dropped"
    assert events[1]["message"] == "resumed"
    assert len(fake.urls) == 2
    assert stream.after == 2


def test_reconnect_resumes_after_the_last_event_read(monkeypatch: pytest.MonkeyPatch) -> None:
    url = _url()
    fake = _install(
        monkeypatch,
        _FakeConnection(_msg({"kind": "token", "seq": 4, "message": "Hel"}), ConnectionResetError("socket dropped")),
        _FakeConnection(_run_state("completed", seq=5)),
    )
    stream = EventStream(url).start()

    _join(stream)

    assert fake.urls == [url(0), url(4)]
    assert [event.get("seq") for event in stream.drain()] == [4, None, 5]


def test_drain_empties_queue_and_second_call_returns_empty_list() -> None:
    stream = EventStream(_url())
    stream.queue.put({"kind": "status", "seq": 1})
    stream.queue.put({"kind": "token", "seq": 2})

    first = stream.drain()
    second = stream.drain()

    assert [event["seq"] for event in first] == [1, 2]
    assert second == []
    assert stream.queue.empty()


def test_stop_is_idempotent_and_reader_thread_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeConnection())
    stream = EventStream(_url()).start()
    assert stream.alive

    stream.stop()
    stream.stop()

    _join(stream)
    assert stream.stop_event.is_set()
    assert stream.alive is False
    stream.stop()
    assert stream.alive is False


@pytest.mark.parametrize("status", ["stopped", "failed", "completed"])
def test_terminal_run_states_stop_the_reader(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    _install(monkeypatch, _FakeConnection(_run_state(status, seq=1)))
    stream = EventStream(_url()).start()

    _join(stream)

    assert stream.stop_event.is_set()
    events = stream.drain()
    assert _kinds(events) == ["run_state"]
    assert events[0]["run"]["status"] == status


def test_running_run_state_does_not_stop_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeConnection(_run_state("running", seq=1)))
    stream = EventStream(_url()).start()
    try:
        event = stream.queue.get(timeout=JOIN_TIMEOUT)

        assert event["run"]["status"] == "running"
        assert not stream.stop_event.is_set()
        assert stream.alive
    finally:
        stream.stop()
        _join(stream)


def test_repeated_connect_failures_queue_errors_and_thread_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        ConnectionRefusedError("first refusal"),
        ConnectionRefusedError("second refusal"),
        _FakeConnection(
            _msg({"kind": "status", "seq": 5, "message": "back online"}),
            _run_state("completed", seq=6),
        ),
    )
    stream = EventStream(_url()).start()

    _join(stream)

    events = stream.drain()
    assert _kinds(events) == ["stream_error", "stream_error", "status", "run_state"]
    assert [event["message"] for event in events[:2]] == ["first refusal", "second refusal"]
    assert stream.after == 6


def test_event_log_stamps_monotonic_sequence_numbers() -> None:
    log = EventLog()

    stamped = [log.add({"kind": "status", "message": f"step {n}"}) for n in range(3)]

    assert [event["seq"] for event in stamped] == [1, 2, 3]
    assert all(event["at"] for event in stamped)
    assert stamped[1]["message"] == "step 1"
    assert log.last_seq == 3


def test_event_log_after_returns_only_later_events() -> None:
    log = EventLog()
    for n in range(4):
        log.add({"kind": "token", "message": str(n)})

    assert [event["seq"] for event in log.after(2)] == [3, 4]
    assert [event["seq"] for event in log.after()] == [1, 2, 3, 4]
    assert log.after(4) == []
    assert log.after(99) == []


def test_event_log_is_bounded_and_sequence_keeps_counting() -> None:
    log = EventLog(max_events=3)
    for n in range(5):
        log.add({"kind": "token", "message": str(n)})

    assert [event["seq"] for event in log.after(0)] == [3, 4, 5]
    assert log.last_seq == 5
    assert log.add({"kind": "status", "message": "next"})["seq"] == 6
