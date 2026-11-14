"""Streamlit frontend: starts a run, streams the pipeline live and lets the user hold, edit and resume it."""

import os
import time
from functools import partial

import httpx
import streamlit as st
from pydantic import ValidationError
from pypdf import PdfReader

from api_common.assets.common import STAGE_OUTPUTS, PipelineRequest, Run, RunStatus
from api_common.client.client import BackendClient, BackendError
from frontend_demo.stream import EventStream
from frontend_demo.views import VIEWS, text

TABLE_STAGES = ("assumptions", "personas", "questions")  # outputs that are one flat list, edited as a table
STATUS_BADGE = {
    RunStatus.RUNNING: ":blue-badge[running]",
    RunStatus.STOPPED: ":orange-badge[on hold]",
    RunStatus.COMPLETED: ":green-badge[completed]",
    RunStatus.FAILED: ":red-badge[failed]",
}
STAGE_ICON = {
    "assumptions": "🎯",
    "personas": "👥",
    "questions": "❓",
    "interviews": "🎙️",
    "validations": "⚖️",
    "conclusions": "🧠",
}
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
    for key in ("run_id", "chat", "events", "stream", "error"):
        st.session_state.pop(key, None)


def new_run() -> None:
    reset_run()
    st.rerun()


def read_plan(plan) -> str:
    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(plan).pages).strip()
    except Exception as error:  # noqa: BLE001 - a malformed PDF should not take the form down
        st.error(f"Could not read the PDF: {error}")
        return ""


def start_form() -> None:
    st.title("Validate a business idea")
    st.caption(
        "Describe the idea, or drop in the business plan you already have. Simulated customers are interviewed "
        "to test its assumptions."
    )
    with st.form("start"):
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.subheader("Start from your notes")
            idea = st.text_area(
                "Business idea",
                height=150,
                placeholder="What are you building, for whom, and what problem does it solve?",
            )
            audience = st.text_area(
                "Target audience",
                height=100,
                placeholder="Who are the customers? Segments, geography, budget, role…",
            )
            info = st.text_area(
                "Additional info",
                height=100,
                placeholder="Pricing, channel, constraints, what you already know…",
            )
        with right:
            st.markdown(
                "<div style='text-align:center;font-weight:700;letter-spacing:.18em;color:#8a8a8a;"
                "margin:0 0 1rem 0;'>⬇ &nbsp; OR &nbsp; ⬇</div>",
                unsafe_allow_html=True,
            )
            st.subheader("…or upload the plan")
            plan = st.file_uploader(
                "Business plan (PDF)",
                type="pdf",
                help="A market-analysis deck, a business plan or a one-pager. The PDF text becomes the brief.",
            )
            st.caption("Already have the market analysis? Skip the three fields and let the plan speak.")

        st.markdown("**Run size**")
        first, second, third = st.columns(3)
        assumption_count = first.number_input(
            "Assumptions",
            min_value=1,
            max_value=20,
            value=None,
            help="Leave empty to let the model pick 6 to 8.",
        )
        persona_count = second.number_input(
            "Personas",
            min_value=1,
            max_value=30,
            value=None,
            help="Leave empty to let the model pick 8 to 10.",
        )
        question_count = third.number_input(
            "Interview questions",
            min_value=1,
            max_value=40,
            value=None,
            help="Leave empty to let the model pick 6 to 10.",
        )

        submitted = st.form_submit_button("Start validation", type="primary", use_container_width=True)

    if not submitted:
        return
    if not plan and not (idea and audience):
        st.error("Upload a business plan, or fill in both the business idea and the target audience.")
        return

    plan_text = read_plan(plan) if plan else ""
    if plan and not plan_text:
        st.error("That PDF has no extractable text. Paste the idea into the fields instead.")
        return

    request = PipelineRequest(
        business_idea=idea,
        target_audience=audience,
        additional_info=info,
        business_plan=plan_text,
        assumption_count=assumption_count,
        persona_count=persona_count,
        question_count=question_count,
    )
    with st.spinner("Starting the validation run…"):
        run = client.start_run(request)
    reset_run()
    st.session_state.run_id = run.id
    st.session_state.chat = []
    st.session_state.events = []
    st.rerun()


def ensure_stream(run: Run) -> EventStream | None:
    stream = st.session_state.get("stream")
    if run.status == RunStatus.RUNNING:
        if stream is None or not stream.alive:
            events = st.session_state.get("events", [])
            after = max((event.get("seq", 0) for event in events), default=0)
            stream = EventStream(partial(client.events_url, run.id), after).start()
            st.session_state.stream = stream
    elif stream is not None and stream.alive:
        stream.stop()
    return stream


def drain_events(stream: EventStream | None) -> None:
    if stream is None:
        return
    new = stream.drain()
    if not new:
        return
    events = st.session_state.setdefault("events", [])
    events.extend(new)
    if len(events) > 3000:  # token events can be chatty
        del events[:-3000]


def live_trace(events: list[dict], stage: str) -> str:
    stage_events = [event for event in events if event.get("stage") == stage]
    statuses = [event["message"] for event in stage_events if event.get("kind") == "status"][-6:]
    lines = [f"- {message}" for message in statuses]
    by_persona: dict[str, str] = {}
    for event in stage_events:
        if event.get("kind") != "token":
            continue
        persona = (event.get("data") or {}).get("persona_id", "live answer")
        by_persona[persona] = by_persona.get(persona, "") + event.get("message", "")
    for persona, streamed in by_persona.items():
        if streamed.strip():
            lines.append(f"\n\n**{persona}**\n\n{text(streamed[-900:])}")
    return "\n".join(lines)


def sidebar(run: Run, current: str | None) -> None:
    with st.sidebar:
        st.header("Validation run")
        st.markdown(STATUS_BADGE[run.status])
        if run.status == RunStatus.RUNNING:
            st.button(
                "Stop",
                icon=":material/pause:",
                type="primary",
                on_click=act,
                args=(client.stop_run, run.id),
                key="stop_sidebar",
            )
        elif run.status != RunStatus.COMPLETED:
            st.button(
                "Resume",
                icon=":material/play_arrow:",
                type="primary",
                on_click=act,
                args=(client.resume_run, run.id),
                key="resume_sidebar",
            )
        st.button("New run", icon=":material/add:", on_click=new_run, key="new_run_sidebar")
        st.divider()
        for stage in STAGE_OUTPUTS:
            done = getattr(run.state, stage) is not None
            icon = ":material/check_circle:" if done else ":material/sync:" if stage == current else ":material/circle:"
            st.markdown(f"{icon} {stage.capitalize()}")
        st.divider()
        request = run.state.request
        if request.assumption_count:
            st.caption(f"Assumptions requested: {request.assumption_count}")
        if request.persona_count:
            st.caption(f"Personas requested: {request.persona_count}")
        if request.question_count:
            st.caption(f"Questions requested: {request.question_count}")
        st.caption(f"{len(st.session_state.get('events', []))} live events received")


def edit_form(run: Run, stage: str) -> None:
    output_model = STAGE_OUTPUTS[stage]
    output = getattr(run.state, stage)
    with st.expander("Edit this stage", icon=":material/edit:"), st.form(f"edit_{stage}"):
        if stage in TABLE_STAGES:
            rows = st.data_editor(output.model_dump()[stage], num_rows="dynamic", hide_index=True)
        else:
            json_text = st.text_area("Output as JSON", output.model_dump_json(indent=2), height=400)
        st.caption("Saving pauses the run and clears the stages after this one. Resume to regenerate them.")
        if not st.form_submit_button("Save and pause"):
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


def stages(run: Run, current: str | None, events: list[dict]) -> None:
    for number, stage in enumerate(STAGE_OUTPUTS, start=1):
        output = getattr(run.state, stage)
        with st.chat_message("assistant", avatar=STAGE_ICON[stage]):
            st.markdown(f"**{number}. {stage.capitalize()}**")
            if output is not None:
                VIEWS[stage](run.state)
                if run.status != RunStatus.RUNNING:
                    edit_form(run, stage)
            elif stage == current:
                with st.status("Thinking…", state="running", expanded=True):
                    trace = live_trace(events, stage)
                    if trace:
                        st.markdown(trace)
                    else:
                        st.caption("Waiting for the model…")
            else:
                st.caption("Waiting for the previous stages.")


def live_controls(run: Run, events: list[dict]) -> None:
    with st.chat_message("assistant", avatar="⏱️"):
        if run.status == RunStatus.RUNNING:
            st.markdown("**The pipeline is running.**")
            statuses = [event["message"] for event in events if event.get("kind") == "status"]
            st.caption(statuses[-1] if statuses else "Warming up…")
            if st.button("Stop run", icon=":material/stop_circle:", type="primary", use_container_width=True, key="stop_chat"):
                act(client.stop_run, run.id)
                st.rerun()
        elif run.status == RunStatus.STOPPED:
            st.markdown("**Run paused at the first unfinished stage.**")
            st.caption("Edit any finished stage above, then resume. The stages after the edited one will run again.")
            if st.button("Resume from here", icon=":material/play_arrow:", type="primary", use_container_width=True, key="resume_chat"):
                act(client.resume_run, run.id)
                st.rerun()
        elif run.status == RunStatus.COMPLETED:
            st.markdown("**Validation complete.** The conclusions are above; ask the agent for an edit below.")
        else:
            st.markdown("**The run failed.** Check the error above, then start a new run.")


def chat(run: Run) -> None:
    st.divider()
    st.subheader("Ask for edits")
    for role, message in st.session_state.get("chat", []):
        st.chat_message(role).markdown(message)
    running = run.status == RunStatus.RUNNING
    placeholder = "Stop the run to ask for edits" if running else "Ask the agent to change a finished stage"
    if prompt := st.chat_input(placeholder, disabled=running):
        st.chat_message("user").markdown(prompt)
        with st.spinner("The agent is editing the run…"):
            reply = client.chat(run.id, prompt).reply
        st.session_state.chat = st.session_state.get("chat", []) + [("user", prompt), ("assistant", reply)]
        st.rerun()


def header(run: Run) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.title("Business idea validation")
        request = run.state.request
        brief = request.business_idea or "Business plan uploaded as the brief."
        st.caption(brief[:240] + ("…" if len(brief) > 240 else ""))
    with right:
        st.markdown(STATUS_BADGE[run.status])
        st.button("New run", icon=":material/add:", on_click=new_run, key="new_run_header", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Business idea validation", page_icon=":material/fact_check:", layout="wide")
    if "run_id" not in st.session_state:
        start_form()
        return

    run = client.get_run(st.session_state.run_id)
    stream = ensure_stream(run)
    drain_events(stream)
    events = st.session_state.get("events", [])

    pending = [stage for stage in STAGE_OUTPUTS if getattr(run.state, stage) is None]
    current = pending[0] if pending and run.status == RunStatus.RUNNING else None

    sidebar(run, current)
    header(run)
    if error := st.session_state.pop("error", None) or run.error:
        st.error(error)
    stages(run, current, events)
    live_controls(run, events)
    chat(run)

    if run.status == RunStatus.RUNNING:
        time.sleep(0.8)
        st.rerun()


try:
    main()
except (BackendError, httpx.HTTPError) as error:
    st.error(f"Backend error: {error}")
    st.button("New run", on_click=new_run)
