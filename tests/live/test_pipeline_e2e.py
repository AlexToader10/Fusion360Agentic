"""One live end-to-end run through the six stages, all on one shared PipelineState.

The run calls the real OpenRouter models, including web search in the interview research, so it costs credits and
its output varies. The assertions check structure, referential integrity and candour, not wording. The stages run
in-process rather than through the process-pool Pipeline, so each output can be inspected directly.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.conclusions import ConclusionsStage
from agent_backend.stages.interviews import InterviewsStage
from agent_backend.stages.personas import PersonasStage
from agent_backend.stages.questions import QuestionsStage
from agent_backend.stages.validations import ValidationsStage
from api_common.assets.common import STAGE_OUTPUTS, PipelineRequest, PipelineState
from tests.helpers import assert_sequential_ids, assert_unique_ids, require_live

pytestmark = [pytest.mark.live, pytest.mark.web, pytest.mark.slow, pytest.mark.e2e]

STAGE_ORDER = ["assumptions", "personas", "questions", "interviews", "validations", "conclusions"]
VERDICTS = {"validated", "invalidated", "inconclusive"}
SIMULATION_NOTE = re.compile(r"simulat|real[ -](customer|people|user|world)", re.IGNORECASE)


class _Run(NamedTuple):
    state: PipelineState
    reached: list[str]


@pytest.fixture(scope="module")
def e2e_run() -> _Run:
    require_live()
    state = PipelineState(
        request=PipelineRequest(
            business_idea=(
                "A waitlist copilot that fills cancelled appointment slots by texting patients on the waitlist, "
                "so the front desk stops phoning people by hand to fill a gap in the book."
            ),
            target_audience=(
                "Owners and office managers of independent dental practices in Germany with two to six chairs, "
                "who run the appointment book themselves."
            ),
            additional_info="Pilot in one German city first. Target price is about 149 EUR per month. Must follow GDPR.",
            persona_count=2,
        )
    )
    stages = [AssumptionsStage(), PersonasStage(), QuestionsStage(), InterviewsStage(), ValidationsStage(), ConclusionsStage()]
    reached: list[str] = []
    for stage in stages:
        state = stage.run(state)
        reached.append(stage.name)
    return _Run(state, reached)


def test_run_reaches_conclusions_and_every_output_is_valid(e2e_run: _Run) -> None:
    state, reached = e2e_run
    assert reached == STAGE_ORDER
    for name, model in STAGE_OUTPUTS.items():
        output = getattr(state, name)
        assert isinstance(output, model), f"{name} output is {type(output).__name__}, expected {model.__name__}"
        assert model.model_validate(output.model_dump()) == output, f"{name} output does not round-trip"


def test_assumptions_and_questions_have_expected_counts_and_ids(e2e_run: _Run) -> None:
    state, _ = e2e_run
    assumptions = state.assumptions.assumptions
    assert 6 <= len(assumptions) <= 8, f"expected 6 to 8 assumptions, got {len(assumptions)}"
    assert_unique_ids(assumptions)
    assert_sequential_ids(assumptions, "A")

    questions = state.questions.questions
    assert 6 <= len(questions) <= 10, f"expected 6 to 10 questions, got {len(questions)}"
    assert_unique_ids(questions)
    assert_sequential_ids(questions, "Q")


def test_personas_are_exactly_two_with_ids_p1_and_p2(e2e_run: _Run) -> None:
    state, _ = e2e_run
    personas = state.personas.personas
    assert len(personas) == 2, f"expected exactly 2 personas, got {len(personas)}"
    assert_unique_ids(personas)
    assert_sequential_ids(personas, "P")


def test_referential_integrity_across_the_chain(e2e_run: _Run) -> None:
    state, _ = e2e_run
    assumption_ids = {a.id for a in state.assumptions.assumptions}
    persona_ids = {p.id for p in state.personas.personas}
    question_ids = {q.id for q in state.questions.questions}

    for question in state.questions.questions:
        unknown = set(question.assumption_ids) - assumption_ids
        assert not unknown, f"{question.id} cites unknown assumptions {sorted(unknown)}"
    for persona in state.personas.personas:
        unknown = set(persona.assumption_ids) - assumption_ids
        assert not unknown, f"{persona.id} cites unknown assumptions {sorted(unknown)}"
    for interview in state.interviews.interviews:
        assert interview.persona_id in persona_ids, f"interview for unknown persona {interview.persona_id}"
        for exchange in interview.exchanges:
            assert exchange.question_id in question_ids, f"exchange cites unknown question {exchange.question_id}"
    for validation in state.validations.validations:
        assert validation.persona_id in persona_ids, f"validation for unknown persona {validation.persona_id}"
        for verdict in validation.verdicts:
            assert verdict.assumption_id in assumption_ids, f"verdict cites unknown assumption {verdict.assumption_id}"
    for conclusion in state.conclusions.assumptions:
        assert conclusion.assumption_id in assumption_ids, f"conclusion cites unknown assumption {conclusion.assumption_id}"


def test_every_assumption_is_tested_judged_and_concluded(e2e_run: _Run) -> None:
    state, _ = e2e_run
    assumption_ids = {a.id for a in state.assumptions.assumptions}

    tested = {assumption_id for question in state.questions.questions for assumption_id in question.assumption_ids}
    assert assumption_ids <= tested, f"assumptions no question tests: {sorted(assumption_ids - tested)}"

    for validation in state.validations.validations:
        judged = Counter(verdict.assumption_id for verdict in validation.verdicts)
        assert dict(judged) == {assumption_id: 1 for assumption_id in assumption_ids}, (
            f"{validation.persona_id} verdicts are not exactly one per assumption: {dict(judged)}"
        )

    concluded = Counter(conclusion.assumption_id for conclusion in state.conclusions.assumptions)
    assert dict(concluded) == {assumption_id: 1 for assumption_id in assumption_ids}, (
        f"conclusions are not exactly one per assumption: {dict(concluded)}"
    )


def test_one_interview_per_persona_with_exchanges_and_no_repeated_questions(e2e_run: _Run) -> None:
    state, _ = e2e_run
    persona_ids = {p.id for p in state.personas.personas}
    interviews = state.interviews.interviews

    assert Counter(interview.persona_id for interview in interviews) == {persona_id: 1 for persona_id in persona_ids}

    for interview in interviews:
        assert interview.exchanges, f"{interview.persona_id} has an interview with no exchanges"
        asked = Counter(exchange.question_id for exchange in interview.exchanges)
        repeated = sorted(question_id for question_id, count in asked.items() if count > 1)
        assert not repeated, f"{interview.persona_id} is asked the same question twice: {repeated}"


def test_no_stage_invents_ids_an_earlier_stage_did_not_produce(e2e_run: _Run) -> None:
    state, _ = e2e_run
    assumption_ids = {a.id for a in state.assumptions.assumptions}
    persona_ids = {p.id for p in state.personas.personas}
    question_ids = {q.id for q in state.questions.questions}

    references = {
        "questions.assumption_ids": (
            {aid for q in state.questions.questions for aid in q.assumption_ids},
            assumption_ids,
        ),
        "personas.assumption_ids": (
            {aid for p in state.personas.personas for aid in p.assumption_ids},
            assumption_ids,
        ),
        "interviews.persona_id": ({i.persona_id for i in state.interviews.interviews}, persona_ids),
        "exchanges.question_id": (
            {e.question_id for i in state.interviews.interviews for e in i.exchanges},
            question_ids,
        ),
        "validations.persona_id": ({v.persona_id for v in state.validations.validations}, persona_ids),
        "verdicts.assumption_id": (
            {verdict.assumption_id for v in state.validations.validations for verdict in v.verdicts},
            assumption_ids,
        ),
        "conclusions.assumption_id": (
            {c.assumption_id for c in state.conclusions.assumptions},
            assumption_ids,
        ),
    }
    invented = {label: sorted(refs - pool) for label, (refs, pool) in references.items() if refs - pool}
    assert not invented, f"later stages reference ids no earlier stage produced: {invented}"

    assert {i.persona_id for i in state.interviews.interviews} == persona_ids
    assert {v.persona_id for v in state.validations.validations} == persona_ids
    assert {c.assumption_id for c in state.conclusions.assumptions} == assumption_ids


def test_conclusions_are_candid_and_actionable(e2e_run: _Run) -> None:
    state, _ = e2e_run
    conclusions = state.conclusions

    assert SIMULATION_NOTE.search(conclusions.summary), (
        f"summary does not say the evidence is simulated or that real customers are needed: {conclusions.summary!r}"
    )
    assert len(conclusions.recommendations) >= 3, f"expected at least 3 recommendations, got {len(conclusions.recommendations)}"
    assert all(recommendation.strip() for recommendation in conclusions.recommendations)
    for conclusion in conclusions.assumptions:
        assert conclusion.verdict in VERDICTS, f"{conclusion.assumption_id} has verdict {conclusion.verdict!r}"
