"""The two ways to change a finished stage: its editor and the chat with the edit agent."""

import streamlit as st

from api_common.assets.common import STAGE_OUTPUTS, Run, RunStatus
from frontend_demo.editors import EDITORS
from frontend_demo.session import act, client
from frontend_demo.views import text

AVATAR = {"user": ":material/person:", "assistant": ":material/psychiatry:"}  # the agent wears the seedling


def start(run: Run, stage: str) -> None:
    """Open the editor of a stage on a draft: a copy of its output that the run does not see before it is saved."""
    st.session_state.drafts[stage] = getattr(run.state, stage).model_dump()


def form(run: Run, stage: str) -> None:
    """The editor of a stage. Save changes sends the edited draft to the run, Cancel throws the draft away."""
    drafts = st.session_state.drafts
    stages = list(STAGE_OUTPUTS)
    cleared = [later for later in stages[stages.index(stage) + 1 :] if getattr(run.state, later) is not None]
    # style.css finds the fields of an editor through the key of this container
    with st.container(key=f"editor_{stage}"), st.form(f"edit_{stage}", enter_to_submit=False, border=False):
        edited = EDITORS[stage](drafts[stage], run.state)
        if cleared:
            st.caption(f"Saving clears {', '.join(cleared)}. Resume the run to generate them again.")
        with st.container(horizontal=True):
            save = st.form_submit_button("Save changes", key=f"save_{stage}", type="primary")
            st.form_submit_button("Cancel", key=f"cancel_{stage}", on_click=drafts.pop, args=(stage,))
    if save:
        act(client.edit_stage, run.id, stage, STAGE_OUTPUTS[stage].model_validate(edited))
        del drafts[stage]
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
