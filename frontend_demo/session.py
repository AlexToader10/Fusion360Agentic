"""What the screens share: the backend client and the run kept in the browser session."""

import os

import streamlit as st

from api_common.client.client import BackendClient, BackendError

client = BackendClient(os.getenv("BACKEND_URL", "http://localhost:8000"))


def act(action, *args) -> None:
    """Call the backend from a button and keep its refusal to show on the next rerun."""
    try:
        action(*args)
    except BackendError as error:
        st.session_state.error = str(error)


def reset_run() -> None:
    stream = st.session_state.get("stream")
    if stream is not None:
        stream.stop()
    for key in ("run_id", "chat", "events", "stream", "error", "drafts"):
        st.session_state.pop(key, None)
