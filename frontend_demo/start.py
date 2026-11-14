"""The start screen: write the idea down or upload the business plan."""

import streamlit as st
from pypdf import PdfReader

from api_common.assets.common import PipelineRequest
from frontend_demo.session import client, reset_run

FIELD_HEIGHT = 120  # style.css sizes the upload box to match the three fields


def read_plan(plan) -> str:
    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(plan).pages).strip()
    except Exception as error:  # noqa: BLE001 - a malformed PDF should not take the form down
        st.error(f"Could not read the PDF: {error}")
        return ""


def page() -> None:
    with st.container(key="hero"):
        st.title("Let's test your business idea", anchor=False)
    with st.form("start", border=False):
        write, upload = st.columns([3, 2], gap="large")
        with write:
            st.subheader("Write it down", anchor=False)
            idea = st.text_area(
                "Business idea",
                height=FIELD_HEIGHT,
                placeholder="What are you building, for whom, and what problem does it solve?",
            )
            audience = st.text_area(
                "Target audience",
                height=FIELD_HEIGHT,
                placeholder="Who are the customers? Segments, geography, budget, role…",
            )
            info = st.text_area(
                "Additional info",
                height=FIELD_HEIGHT,
                placeholder=(
                    "Pricing, channel, constraints, what you already know… You can also say how big the run should be, "
                    'like "5 assumptions, 3 personas and 6 questions".'
                ),
            )
        with upload:
            st.subheader("Or upload your plan", anchor=False)
            plan = st.file_uploader(
                "Business plan (PDF)",
                type="pdf",
                help="A market-analysis deck, a business plan or a one-pager. The PDF text becomes the brief.",
            )
        submitted = st.form_submit_button("Start validation", type="primary")

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
    )
    with st.spinner("Starting the validation run…"):
        run = client.start_run(request)
    reset_run()
    st.session_state.run_id = run.id
    st.rerun()
