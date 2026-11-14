"""The two ways to change a finished stage: its edit form and the chat with the edit agent."""

import streamlit as st
from pydantic import ValidationError

from api_common.assets.common import STAGE_OUTPUTS, Run, RunStatus
from frontend_demo.session import act, client
from frontend_demo.views import text

TABLE_STAGES = ("assumptions", "personas", "questions")  # outputs that are one flat list, edited as a table
AVATAR = {"user": ":material/person:", "assistant": ":material/psychiatry:"}  # the agent wears the seedling


def form(run: Run, stage: str) -> None:
    output_model = STAGE_OUTPUTS[stage]
    output = getattr(run.state, stage)
    with st.expander(f"Edit {stage}", icon=":material/edit:"), st.form(f"edit_{stage}", border=False):
        if stage in TABLE_STAGES:
            rows = st.data_editor(output.model_dump()[stage], num_rows="dynamic", hide_index=True)
        else:
            json_text = st.text_area(f"{stage.capitalize()} as JSON", output.model_dump_json(indent=2), height=400)
        st.caption("Saving clears the stages after this one. Resume the run to generate them again.")
        if not st.form_submit_button("Save changes"):
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


def chat(run: Run) -> None:
    history = st.session_state.get("chat", [])
    if history:
        st.header("Ask for edits", anchor=False)
    for role, message in history:
        st.chat_message(role, avatar=AVATAR[role]).markdown(text(message))
    running = run.status == RunStatus.RUNNING
    placeholder = "Stop the run to ask for edits" if running else "Ask the agent to change a finished stage"
    if prompt := st.chat_input(placeholder, disabled=running):
        st.chat_message("user", avatar=AVATAR["user"]).markdown(text(prompt))
        with st.spinner("The agent is editing the run…"):
            reply = client.chat(run.id, prompt).reply
        st.session_state.chat = history + [("user", prompt), ("assistant", reply)]
        st.rerun()
