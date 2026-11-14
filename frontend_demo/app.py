"""Streamlit frontend: starts a run, shows every stage as it finishes and lets the user hold, edit and resume it."""

import os
import time

import httpx
import streamlit as st
from pydantic import ValidationError
from pypdf import PdfReader

from api_common.assets.common import STAGE_OUTPUTS, PipelineRequest, Run, RunStatus
from api_common.client.client import BackendClient, BackendError
from frontend_demo.views import VIEWS

TABLE_STAGES = ("assumptions", "personas", "questions")  # outputs that are one flat list, edited as a table
STATUS_BADGE = {
    RunStatus.RUNNING: ":blue-badge[running]",
    RunStatus.STOPPED: ":orange-badge[on hold]",
    RunStatus.COMPLETED: ":green-badge[completed]",
    RunStatus.FAILED: ":red-badge[failed]",
}

client = BackendClient(os.getenv("BACKEND_URL", "http://localhost:8000"))


def act(action, *args) -> None:
    """Call the backend from a button and keep its refusal to show on the next rerun."""
    try:
        action(*args)
    except BackendError as error:
        st.session_state.error = str(error)


def start_form() -> None:
    st.title("Validate a business idea")
    st.caption("Describe the idea or upload the business plan. Simulated customers are interviewed to test its assumptions.")
    with st.form("start"):
        plan = st.file_uploader("Business plan (PDF)", type="pdf")
        idea = st.text_area("Detailed business idea", height=150)
        audience = st.text_area("Target audience")
        info = st.text_area("Additional info")
        count = st.number_input("Number of personas to interview (optional)", min_value=1, max_value=30, value=None)
        if not st.form_submit_button("Start validation", type="primary"):
            return
    if not plan and not (idea and audience):
        st.error("Upload a business plan or fill in the business idea and the target audience.")
        return
    plan_text = "\n".join(page.extract_text() for page in PdfReader(plan).pages) if plan else ""
    request = PipelineRequest(
        business_idea=idea, target_audience=audience, additional_info=info, business_plan=plan_text, persona_count=count
    )
    st.session_state.run_id = client.start_run(request).id
    st.session_state.chat = []
    st.rerun()


def sidebar(run: Run, current: str | None) -> None:
    with st.sidebar:
        st.header("Validation run")
        st.markdown(STATUS_BADGE[run.status])
        if run.status == RunStatus.RUNNING:
            st.button("Stop", icon=":material/pause:", type="primary", on_click=act, args=(client.stop_run, run.id))
        elif run.status != RunStatus.COMPLETED:
            st.button("Resume", icon=":material/play_arrow:", type="primary", on_click=act, args=(client.resume_run, run.id))
        st.button("New run", icon=":material/add:", on_click=st.session_state.clear)
        st.divider()
        for stage in STAGE_OUTPUTS:
            done = getattr(run.state, stage) is not None
            icon = ":material/check_circle:" if done else ":material/sync:" if stage == current else ":material/circle:"
            st.markdown(f"{icon} {stage.capitalize()}")


def edit_form(run: Run, stage: str) -> None:
    output_model = STAGE_OUTPUTS[stage]
    output = getattr(run.state, stage)
    with st.expander("Edit", icon=":material/edit:"), st.form(f"edit_{stage}"):
        if stage in TABLE_STAGES:
            rows = st.data_editor(output.model_dump()[stage], num_rows="dynamic", hide_index=True)
        else:
            json_text = st.text_area("Output as JSON", output.model_dump_json(indent=2), height=400)
        st.caption("Saving clears the stages after this one. They run again when you resume.")
        if not st.form_submit_button("Save"):
            return
        try:
            if stage in TABLE_STAGES:
                edited = output_model.model_validate({stage: rows})
            else:
                edited = output_model.model_validate_json(json_text)
        except ValidationError as error:
            st.error(str(error))
            return
        act(client.edit_stage, run.id, stage, edited)
        st.rerun()


def stages(run: Run, current: str | None) -> None:
    for number, stage in enumerate(STAGE_OUTPUTS, start=1):
        st.subheader(f"{number}. {stage.capitalize()}")
        if getattr(run.state, stage) is not None:
            VIEWS[stage](run.state)
            if run.status != RunStatus.RUNNING:
                edit_form(run, stage)
        elif stage == current:
            st.status("Running…", state="running")
        else:
            st.caption("Waiting for the previous stages.")


def chat(run: Run) -> None:
    st.divider()
    st.subheader("Ask for edits")
    for role, message in st.session_state.chat:
        st.chat_message(role).markdown(message)
    running = run.status == RunStatus.RUNNING
    placeholder = "Stop the run to ask for edits" if running else "Ask the agent to change a finished stage"
    if prompt := st.chat_input(placeholder, disabled=running):
        st.chat_message("user").markdown(prompt)
        with st.spinner("The agent is editing the run…"):
            reply = client.chat(run.id, prompt).reply
        st.session_state.chat += [("user", prompt), ("assistant", reply)]
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Business idea validation", page_icon=":material/fact_check:", layout="wide")
    if "run_id" not in st.session_state:
        start_form()
        return

    run = client.get_run(st.session_state.run_id)
    pending = [stage for stage in STAGE_OUTPUTS if getattr(run.state, stage) is None]
    current = pending[0] if pending and run.status == RunStatus.RUNNING else None

    sidebar(run, current)
    st.title("Business idea validation")
    if error := st.session_state.pop("error", None) or run.error:
        st.error(error)
    stages(run, current)
    chat(run)

    if run.status == RunStatus.RUNNING:
        time.sleep(2)
        st.rerun()


try:
    main()
except (BackendError, httpx.HTTPError) as error:
    st.error(f"Backend error: {error}")
    st.button("New run", on_click=st.session_state.clear)
