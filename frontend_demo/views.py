"""How the output of each finished stage is displayed.

Short structured fields are drawn as HTML that style.css styles. Longer text the models wrote goes through st.markdown.
"""

import html
from collections import Counter
from collections.abc import Iterable

import streamlit as st

from api_common.assets.common import PipelineState

VERDICTS = ("validated", "invalidated", "inconclusive")


def text(value: str) -> str:
    return value.replace("$", "\\$")  # keep Streamlit from reading prices as LaTeX


def quote(value: str) -> str:
    """The words of a persona, as a markdown blockquote."""
    return "\n".join(f"> {line}" for line in text(value).splitlines())


def persona_names(state: PipelineState) -> dict[str, str]:
    return {persona.id: persona.name for persona in state.personas.personas}


def statements(state: PipelineState) -> dict[str, str]:
    return {assumption.id: assumption.statement for assumption in state.assumptions.assumptions}


def chips(ids: list[str], titles: dict[str, str] | None = None) -> str:
    """Ids as small pills. With titles, hovering one shows what the id stands for."""
    titles = titles or {}
    return "".join(
        f'<span class="chip" title="{html.escape(titles.get(id, ""))}">{html.escape(id)}</span>' for id in ids
    )


def mark(verdict: str, title: str = "") -> str:
    """A verdict as a shape: filled check, filled cross or empty ring, so it never rests on color alone."""
    label = html.escape(title or verdict)
    return f'<span class="mark {verdict}" role="img" aria-label="{label}" title="{label}"></span>'


def verdict_label(verdict: str) -> str:
    return f'<span class="verdict">{mark(verdict)} {verdict}</span>'


def row(lead: str, main: str = "", note: str = "", aside: str = "") -> str:
    """One line of a list: an id in the gutter, the main text, a quieter note under it and an aside at the right."""
    main = f'<p class="main">{html.escape(main)}</p>' if main else ""
    note = f'<p class="note">{html.escape(note)}</p>' if note else ""
    return f'<li><div>{lead}</div><div>{main}{note}</div><div class="aside">{aside}</div></li>'


def rows(lines: Iterable[str]) -> None:
    st.html(f'<ul class="rows">{"".join(lines)}</ul>')


def assumptions(state: PipelineState) -> None:
    rows(
        row(
            chips([assumption.id]),
            assumption.statement,
            assumption.rationale,
            f'<span class="tag">{html.escape(assumption.category)}</span>',
        )
        for assumption in state.assumptions.assumptions
    )


def personas(state: PipelineState) -> None:
    titles = statements(state)
    cards = "".join(
        f'<li><p class="name">{html.escape(persona.name)}</p>'
        f'<p class="note">{persona.age}, {html.escape(persona.occupation)}</p>'
        f"<p>{html.escape(persona.background)}</p>"
        f"<p><b>Goals</b> {html.escape(persona.goals)}</p>"
        f"<p><b>Pain points</b> {html.escape(persona.pain_points)}</p>"
        f'<p class="speaks">{chips(persona.assumption_ids, titles)}</p></li>'
        for persona in state.personas.personas
    )
    st.html(f'<ul class="personas">{cards}</ul>')


def questions(state: PipelineState) -> None:
    titles = statements(state)
    rows(
        row(chips([question.id]), question.text, aside=chips(question.assumption_ids, titles))
        for question in state.questions.questions
    )


def interviews(state: PipelineState) -> None:
    names = persona_names(state)
    for index, interview in enumerate(state.interviews.interviews):
        count = len(interview.exchanges)
        title = f"{names.get(interview.persona_id, interview.persona_id)}, {count} question{'s' * (count != 1)}"
        with st.expander(title):
            with st.popover("Research behind this persona"):
                st.markdown(text(interview.research))
            with st.container(key=f"transcript_{index}"):  # style.css lines the answers up under their questions
                for exchange in interview.exchanges:
                    rows([row(chips([exchange.question_id]), exchange.question)])
                    st.markdown(quote(exchange.answer))


def validations(state: PipelineState) -> None:
    names = persona_names(state)
    titles = statements(state)
    results = state.validations.validations
    found = {(result.persona_id, verdict.assumption_id): verdict for result in results for verdict in result.verdicts}

    legend = "".join(f"<span>{verdict_label(verdict)}</span>" for verdict in VERDICTS)
    head = "".join(
        f'<th scope="col"><span>{html.escape(names.get(result.persona_id, result.persona_id))}</span></th>'
        for result in results
    )
    body = ""
    for assumption in state.assumptions.assumptions:
        cells = ""
        for result in results:
            verdict = found.get((result.persona_id, assumption.id))
            name = names.get(result.persona_id, result.persona_id)
            hover = f"{name}: {verdict.verdict}. {verdict.evidence}" if verdict else ""
            cells += f"<td>{mark(verdict.verdict, hover) if verdict else ''}</td>"
        statement = html.escape(assumption.statement)
        body += f'<tr><th scope="row"><div>{chips([assumption.id])}<span>{statement}</span></div></th>{cells}</tr>'
    st.html(
        f'<div class="verdicts"><p class="legend">{legend}</p>'
        f"<table><thead><tr><td></td>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )

    with st.expander("Evidence"):
        for result in results:
            st.markdown(f"**{names.get(result.persona_id, result.persona_id)}**")
            rows(
                row(chips([verdict.assumption_id], titles), note=verdict.evidence, aside=verdict_label(verdict.verdict))
                for verdict in result.verdicts
            )


def conclusions(state: PipelineState) -> None:
    result = state.conclusions
    titles = statements(state)
    with st.container(key="summary"):
        st.markdown(text(result.summary))

    counts = Counter(conclusion.verdict for conclusion in result.assumptions)
    tally = "".join(f"<span>{mark(verdict)} {counts[verdict]} {verdict}</span>" for verdict in VERDICTS)
    st.html(f'<p class="tally">{tally}</p>')
    rows(
        row(
            chips([conclusion.assumption_id]),
            titles.get(conclusion.assumption_id, ""),
            conclusion.reasoning,
            verdict_label(conclusion.verdict),
        )
        for conclusion in result.assumptions
    )

    st.subheader("Recommendations", anchor=False)
    st.markdown("\n".join(f"- {text(recommendation)}" for recommendation in result.recommendations))


VIEWS = {
    "assumptions": assumptions,
    "personas": personas,
    "questions": questions,
    "interviews": interviews,
    "validations": validations,
    "conclusions": conclusions,
}
