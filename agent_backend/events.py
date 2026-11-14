"""Live progress events for a pipeline run.

Stages run in threads (the work is network I/O against OpenRouter), so a stage can call its reporter directly.
The run manager stores the events and the WebSocket endpoint streams them to the frontend.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

EventSink = Callable[[dict[str, Any]], None]


class Reporter:
    """What a stage uses to report progress. All methods are safe to call from worker threads."""

    def __init__(self, stage: str, emit: EventSink):
        self.stage = stage
        self._emit = emit

    def status(self, message: str, **data: Any) -> None:
        self._send("status", message, data)

    def token(self, text: str, **data: Any) -> None:
        self._send("token", text, data)

    def child(self, **data: Any) -> Reporter:
        """A reporter that adds constant fields (for example a persona id) to every event."""
        return _ChildReporter(self, data)

    def _send(self, kind: str, message: str, data: dict[str, Any]) -> None:
        event: dict[str, Any] = {"kind": kind, "stage": self.stage, "message": message}
        if data:
            event["data"] = data
        self._emit(event)

    @classmethod
    def noop(cls, stage: str) -> Reporter:
        return cls(stage, lambda _: None)


class _ChildReporter(Reporter):
    def __init__(self, parent: Reporter, data: dict[str, Any]):
        super().__init__(parent.stage, parent._emit)
        self._parent = parent
        self._data = data

    def _send(self, kind: str, message: str, data: dict[str, Any]) -> None:
        merged = {**self._data, **data}
        self._parent._send(kind, message, merged)

    def child(self, **data: Any) -> Reporter:
        return _ChildReporter(self, data)


class EventLog:
    """Append-only event log with monotonic sequence numbers and a bounded size."""

    def __init__(self, max_events: int = 6000):
        self.max_events = max_events
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            stamped = {
                "seq": self._seq,
                "at": datetime.now(timezone.utc).isoformat(),
                **event,
            }
            self._events.append(stamped)
            if len(self._events) > self.max_events:
                del self._events[: len(self._events) - self.max_events]
            return stamped

    def after(self, seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if event["seq"] > seq]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq
