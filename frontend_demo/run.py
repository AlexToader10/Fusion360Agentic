"""The run screen: progress in the sidebar, the finished stages as one document and the unfinished stage under them."""

import time

import streamlit as st

from api_common.assets.common import STAGE_OUTPUTS, Run, RunStatus
from frontend_demo import edit, live
from frontend_demo.session import act, client, reset_run
from frontend_demo.views import VIEWS

TITLE = {
    RunStatus.RUNNING: "Testing your idea",
    RunStatus.STOPPED: "Run stopped",
    RunStatus.COMPLETED: "Here's how your idea held up",
    RunStatus.FAILED: "The run failed",
}
BADGE = {  # label, icon and color of the status badge
    RunStatus.RUNNING: ("Running", ":material/sync:", "primary"),
    RunStatus.STOPPED: ("Stopped", ":material/pause:", "yellow"),
    RunStatus.COMPLETED: ("Completed", ":material/check:", "green"),
    RunStatus.FAILED: ("Failed", ":material/error:", "red"),
}
DONE_MARK = ":green[:material/check_circle:]"
WAITING_MARK = ":material/radio_button_unchecked:"
UNFINISHED_MARK = {  # of the first stage without an output
    RunStatus.RUNNING: ":green[:material/radio_button_checked:]",
    RunStatus.STOPPED: ":yellow[:material/pause_circle:]",
    RunStatus.FAILED: ":red[:material/error:]",
}
BRIEF_LENGTH = 240


def control(run: Run, place: str, width: str = "content") -> None:
    """The one action the run offers now: stop it while it runs, resume it once it is stopped or has failed."""
    if run.status == RunStatus.RUNNING:
        action, label, icon = client.stop_run, "Stop run", ":material/stop_circle:"
    elif run.status != RunStatus.COMPLETED:
        action, label, icon = client.resume_run, "Resume run", ":material/play_arrow:"
    else:
        return
    key = f"control_{place}"
    st.button(label, icon=icon, type="primary", on_click=act, args=(action, run.id), key=key, width=width)


def sidebar(run: Run, unfinished: str | None) -> None:
    """The stages as progress and as a table of contents, with the controls under them."""
    steps = []
    for stage in STAGE_OUTPUTS:
        title = stage.capitalize()
        if getattr(run.state, stage) is not None:
            steps.append(f"{DONE_MARK} [{title}](#{stage})")
        elif stage == unfinished:
            steps.append(f"{UNFINISHED_MARK[run.status]} **{title}**")
        else:
            steps.append(f":gray[{WAITING_MARK} {title}]")

    with st.sidebar:
        st.header("Business idea validation", anchor=False)
        label, icon, color = BADGE[run.status]
        st.badge(label, icon=icon, color=color)
        with st.container(key="steps"):
            st.markdown("  \n".join(steps))
        control(run, "sidebar", width="stretch")
        st.button("New run", icon=":material/add:", on_click=reset_run, width="stretch")


def unfinished_stage(run: Run, stage: str, log: list[dict]) -> None:
    """The first stage without an output: its live trace while it runs, otherwise what to do about it."""
    if run.status == RunStatus.RUNNING:
        st.header(stage.capitalize(), anchor=stage)
        live.trace(log, stage, run.state)
    elif run.status == RunStatus.STOPPED:
        st.warning(
            f"Stopped at {stage}. Edit any finished stage, then resume. The stages after an edit run again.",
            icon=":material/pause_circle:",
        )
    else:
        st.error(f"{run.error}\n\nResume the run to try {stage} again.", icon=":material/error:")
    control(run, "page")


def finished_stage(run: Run, stage: str, drafts: dict) -> None:
    """A finished stage under its header: its output with an Edit button, or its editor once that was pressed."""
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom"):
        st.header(stage.capitalize(), anchor=stage)
        if run.status != RunStatus.RUNNING and stage not in drafts:
            key = f"start_{stage}"
            st.button("Edit", icon=":material/edit:", type="tertiary", on_click=edit.start, args=(run, stage), key=key)
    if stage in drafts:
        edit.form(run, stage)
    else:
        VIEWS[stage](run.state)


def page() -> None:
    run = client.get_run(st.session_state.run_id)
    drafts = st.session_state.setdefault("drafts", {})
    if run.status == RunStatus.RUNNING:
        drafts.clear()  # an editor left open is out of date once the run moves again
    log = live.events(run)
    unfinished = next((stage for stage in STAGE_OUTPUTS if getattr(run.state, stage) is None), None)

    sidebar(run, unfinished)
    st.title(TITLE[run.status], anchor=False)
    brief = run.state.request.business_idea or "Business plan uploaded as the brief."
    st.caption(brief[:BRIEF_LENGTH] + ("…" if len(brief) > BRIEF_LENGTH else ""))
    if error := st.session_state.pop("error", None):
        st.error(error)

    for stage in STAGE_OUTPUTS:
        if getattr(run.state, stage) is None:
            break
        finished_stage(run, stage, drafts)
    if unfinished:
        unfinished_stage(run, unfinished, log)
    edit.chat(run)

    if run.status == RunStatus.RUNNING:
        time.sleep(0.8)
        st.rerun()
