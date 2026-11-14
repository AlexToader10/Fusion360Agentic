"""How the output of each finished stage is displayed."""

import streamlit as st

from api_common.assets.common import PipelineState

VERDICT_BADGE = {
    "validated": ":green-badge[validated]",
    "invalidated": ":red-badge[invalidated]",
    "inconclusive": ":gray-badge[inconclusive]",
}
VERDICT_ICON = {"validated": "✅ validated", "invalidated": "❌ invalidated", "inconclusive": "❔ inconclusive"}


def text(value: str) -> str:
    return value.replace("$", "\\$")  # keep Streamlit from reading prices as LaTeX


def badges(ids: list[str]) -> str:
    return " ".join(f":blue-badge[{id}]" for id in ids)


def persona_names(state: PipelineState) -> dict[str, str]:
    return {persona.id: persona.name for persona in state.personas.personas}


def assumptions(state: PipelineState) -> None:
    for assumption in state.assumptions.assumptions:
        st.markdown(f"**{assumption.id}** · {text(assumption.statement)} :gray-badge[{assumption.category}]")
        st.caption(text(assumption.rationale))


def personas(state: PipelineState) -> None:
    columns = st.columns(3)
    for index, persona in enumerate(state.personas.personas):
        with columns[index % 3].container(border=True):
            st.markdown(f"**{persona.name}**, {persona.age}")
            st.caption(persona.occupation)
            st.markdown(text(persona.background))
            st.markdown(f"**Goals** · {text(persona.goals)}")
            st.markdown(f"**Pain points** · {text(persona.pain_points)}")
            st.markdown(badges(persona.assumption_ids))


def questions(state: PipelineState) -> None:
    for question in state.questions.questions:
        st.markdown(f"**{question.id}** · {text(question.text)} {badges(question.assumption_ids)}")


def interviews(state: PipelineState) -> None:
    names = persona_names(state)
    for interview in state.interviews.interviews:
        title = f"{names.get(interview.persona_id, interview.persona_id)} · {len(interview.exchanges)} questions"
        with st.expander(title):
            with st.popover("Research behind this persona"):
                st.markdown(text(interview.research))
            for exchange in interview.exchanges:
                st.chat_message("user", avatar="🎙️").markdown(f"**{exchange.question_id}** · {text(exchange.question)}")
                st.chat_message("assistant", avatar="🧑").markdown(text(exchange.answer))


def validations(state: PipelineState) -> None:
    names = persona_names(state)
    results = state.validations.validations
    verdicts = {(v.persona_id, verdict.assumption_id): verdict.verdict for v in results for verdict in v.verdicts}
    rows = [
        {"Assumption": f"{assumption.id} · {assumption.statement}"}
        | {
            names.get(v.persona_id, v.persona_id): VERDICT_ICON.get(verdicts.get((v.persona_id, assumption.id)), "")
            for v in results
        }
        for assumption in state.assumptions.assumptions
    ]
    st.dataframe(rows, hide_index=True)
    with st.expander("Evidence"):
        for validation in results:
            st.markdown(f"**{names.get(validation.persona_id, validation.persona_id)}**")
            for verdict in validation.verdicts:
                st.markdown(f"- {VERDICT_BADGE[verdict.verdict]} **{verdict.assumption_id}** · {text(verdict.evidence)}")


def conclusions(state: PipelineState) -> None:
    result = state.conclusions
    st.markdown(text(result.summary))
    for assumption in result.assumptions:
        st.markdown(f"{VERDICT_BADGE[assumption.verdict]} **{assumption.assumption_id}** · {text(assumption.reasoning)}")
    st.markdown("**Recommendations**")
    st.markdown("\n".join(f"- {text(recommendation)}" for recommendation in result.recommendations))


VIEWS = {
    "assumptions": assumptions,
    "personas": personas,
    "questions": questions,
    "interviews": interviews,
    "validations": validations,
    "conclusions": conclusions,
}
