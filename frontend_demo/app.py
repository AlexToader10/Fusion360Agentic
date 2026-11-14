"""Streamlit frontend: starts a run, streams the pipeline live and lets the user stop, edit and resume it."""

from pathlib import Path

import httpx
import streamlit as st

from api_common.client.client import BackendError
from frontend_demo import run, start
from frontend_demo.session import reset_run

st.set_page_config(page_title="Business idea validation", page_icon="🌱")
st.html(Path(__file__).with_name("style.css"))

try:
    if "run_id" in st.session_state:
        run.page()
    else:
        start.page()
except (BackendError, httpx.HTTPError) as error:
    st.error(f"The request to the validation service failed: {error}")
    st.button("New run", on_click=reset_run)
