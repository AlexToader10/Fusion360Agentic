"""The live events of the run in this session, and the trace of the stage that is running."""

from functools import partial

import streamlit as st

from api_common.assets.common import PipelineState, Run, RunStatus
from frontend_demo.session import client
from frontend_demo.stream import EventStream
from frontend_demo.views import persona_names, quote

MAX_EVENTS = 3000  # token events can be chatty
ANSWER_TAIL = 900  # characters of a streaming answer kept on screen


def events(run: Run) -> list[dict]:
    """Keep the stream open while the run is running, close it otherwise, and return every event read so far."""
    log = st.session_state.setdefault("events", [])
    stream = st.session_state.get("stream")
    if run.status == RunStatus.RUNNING:
        if stream is None or not stream.alive:
            after = max((event.get("seq", 0) for event in log), default=0)
            stream = st.session_state.stream = EventStream(partial(client.events_url, run.id), after).start()
    elif stream is not None and stream.alive:
        stream.stop()
    if stream is not None:
        log.extend(stream.drain())
        del log[:-MAX_EVENTS]
    return log


def trace(log: list[dict], stage: str, state: PipelineState) -> None:
    """What the running stage reported: its last status lines and, in interviews, the answers as they are written."""
    stage_events = [event for event in log if event.get("stage") == stage]
    statuses = [event["message"] for event in stage_events if event.get("kind") == "status"][-6:]
    answers: dict[str, str] = {}
    for event in stage_events:
        if event.get("kind") == "token":
            persona_id = (event.get("data") or {}).get("persona_id", "Live answer")
            answers[persona_id] = answers.get(persona_id, "") + event.get("message", "")
    names = persona_names(state) if state.personas else {}

    # Not a with block: leaving one marks the status complete, and this stage is still running
    box = st.status(f"Working on {stage}", state="running", expanded=True)
    if statuses:
        box.markdown("\n".join(f"- {message}" for message in statuses))
    else:
        box.caption("Waiting for the model…")
    for persona_id, answer in answers.items():
        if answer.strip():
            box.markdown(f"**{names.get(persona_id, persona_id)}**\n\n{quote(answer[-ANSWER_TAIL:])}")
