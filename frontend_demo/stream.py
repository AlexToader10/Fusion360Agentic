"""Background WebSocket reader for the Streamlit frontend.

Streamlit reruns the script on every interaction, so the socket lives in a daemon thread and pushes events into a
queue. The script drains the queue on each rerun. That keeps the Stop button clickable while tokens are streaming.
"""

from __future__ import annotations

import json
import queue
import threading
import time

try:
    import websocket  # type: ignore[import-not-found]
except ImportError:  # the frontend still works through HTTP polling; only the live trace is missing
    websocket = None


class EventStream:
    def __init__(self, url: str, after: int = 0):
        self.url = url
        self.after = after
        self.queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pipeline-events", daemon=True)

    def start(self) -> EventStream:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def drain(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self.queue.get_nowait())
            except queue.Empty:
                return events

    def _run(self) -> None:
        if websocket is None:
            self.queue.put({"kind": "stream_error", "message": "websocket-client is not installed"})
            return
        while not self.stop_event.is_set():
            try:
                connection = websocket.create_connection(self.url, timeout=10)
                connection.settimeout(1.0)
                while not self.stop_event.is_set():
                    try:
                        raw = connection.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    event = json.loads(raw)
                    seq = event.get("seq")
                    if isinstance(seq, int):
                        self.after = max(self.after, seq)
                    self.queue.put(event)
                    if event.get("kind") == "run_state" and event.get("run", {}).get("status") in {
                        "completed",
                        "failed",
                        "stopped",
                    }:
                        self.stop_event.set()
                        break
                connection.close()
            except Exception as error:  # noqa: BLE001 - a dropped socket reconnects; the UI keeps polling
                if self.stop_event.is_set():
                    break
                self.queue.put({"kind": "stream_error", "message": str(error)})
                time.sleep(0.5)
