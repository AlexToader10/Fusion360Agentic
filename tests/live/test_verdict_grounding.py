"""Live check that ValidationsStage grounds each verdict in what the interviewee actually said.

The interview is built with a known ground truth: A1 is clearly validated, A2 is clearly invalidated and A3 is
clearly inconclusive. The stage is run once per module and its result is reused by every test. A second
module-scoped run validates an interview with no answers, which must give inconclusive verdicts only.

A contradiction case (a persona who cancelled scheduling software for cost but would pay for time savings) was
left out on purpose. It would need a third validation call, and this module keeps to two calls because each one
is a paid, stochastic model call.
"""

from __future__ import annotations

import re

import pytest

from agent_backend.stages.validations import ValidationsStage
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.interviews import Interview, InterviewsOutput
from api_common.assets.validations import InterviewValidation, ValidationsOutput
from tests.helpers import assert_unique_ids, make_assumption, make_exchange, make_interview, token_set

pytestmark = [pytest.mark.live]

PERSONA_ID = "P1"
EMPTY_PERSONA_ID = "P2"
ASSUMPTION_IDS = ("A1", "A2", "A3")

STATEMENTS = {
    "A1": "Practices lose revenue from unfilled cancellations.",
    "A2": "Practices will pay 49 EUR/month for a waitlist copilot.",
    "A3": "Patients accept short-notice slots offered from a waitlist.",
}

QUESTIONS = {
    "A1": "Walk me through the last time a patient cancelled on you.",
    "A2": "Would you pay for a tool that fills cancelled slots from a waitlist? What would it cost you?",
    "A3": "Would your patients take a short-notice slot? Have you tried offering one?",
}

ANSWERS = {
    "A1": (
        "Last Tuesday a patient cancelled two hours before. Nobody called the waitlist; the chair stayed empty "
        "and we lost about 180 euros. It happens maybe three times a month."
    ),
    "A2": (
        "Nothing. We use the calendar in our practice software. I would not pay extra for a waitlist tool; "
        "49 euros a month is too much for something my assistant can do with phone calls."
    ),
    "A3": "I don't know. We have never really tried it, so I cannot say.",
}

EXPECTED = {"A1": "validated", "A2": "invalidated", "A3": "inconclusive"}

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his i if in into is
    it its just me more most my no nor not of off on or our out over she so some such than that the their them then
    there these they this those to too under until up us very was we were what when where which while who why will
    with would you your""".split()
)

# The stage never receives the persona's web research, so evidence must not refer to research or sources.
RESEARCH_WORDS = frozenset({"research", "web", "survey", "sources", "study", "online", "internet", "report"})

NOT_ENOUGH_EVIDENCE = re.compile(
    r"\b(no|not|insufficient|nothing|cannot|can't|none|without|lacks?|didn't|unknown|unclear)\b"
)


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(token_set(text) - STOPWORDS)


def _request() -> PipelineRequest:
    return PipelineRequest(
        business_idea="A waitlist copilot that fills cancelled dental appointments within minutes.",
        target_audience="Independent dental practices with one to three chairs.",
    )


def _assumptions() -> AssumptionsOutput:
    categories = {"A1": "problem", "A2": "willingness to pay", "A3": "solution"}
    return AssumptionsOutput(
        assumptions=[
            make_assumption(
                assumption_id,
                category=categories[assumption_id],
                statement=STATEMENTS[assumption_id],
                rationale=f"The business depends on {assumption_id}.",
            )
            for assumption_id in ASSUMPTION_IDS
        ]
    )


def _grounded_interview() -> Interview:
    exchanges = [
        make_exchange(
            "Q1",
            question=QUESTIONS["A1"],
            answer=ANSWERS["A1"],
        ),
        make_exchange(
            "Q2",
            question=QUESTIONS["A2"],
            answer=ANSWERS["A2"],
        ),
        make_exchange(
            "Q3",
            question=QUESTIONS["A3"],
            answer=ANSWERS["A3"],
        ),
    ]
    return make_interview(
        PERSONA_ID,
        exchanges,
        research="Independent dental practices often run a paper waitlist and mention no-shows in trade surveys.",
    )


def _run_validation(interview: Interview) -> ValidationsOutput:
    state = PipelineState(
        request=_request(),
        assumptions=_assumptions(),
        interviews=InterviewsOutput(interviews=[interview]),
    )
    return ValidationsStage().run(state).validations


@pytest.fixture(scope="module")
def validation_result() -> ValidationsOutput:
    return _run_validation(_grounded_interview())


@pytest.fixture(scope="module")
def empty_validation_result() -> ValidationsOutput:
    return _run_validation(make_interview(EMPTY_PERSONA_ID, [], research="No research on this persona."))


def _only_persona(output: ValidationsOutput, persona_id: str) -> InterviewValidation:
    matches = [validation for validation in output.validations if validation.persona_id == persona_id]
    assert len(matches) == 1, f"expected one validation for {persona_id}, got {len(matches)}:\n{output.model_dump_json(indent=2)}"
    return matches[0]


def _verdicts_by_id(validation: InterviewValidation) -> dict[str, str]:
    return {verdict.assumption_id: verdict.verdict for verdict in validation.verdicts}


def test_exactly_one_validation_for_the_persona(validation_result: ValidationsOutput) -> None:
    assert len(validation_result.validations) == 1, validation_result.model_dump_json(indent=2)
    assert validation_result.validations[0].persona_id == PERSONA_ID


def test_verdicts_cover_each_assumption_exactly_once(validation_result: ValidationsOutput) -> None:
    validation = _only_persona(validation_result, PERSONA_ID)
    assert_unique_ids(validation.verdicts, attr="assumption_id")
    ids = [verdict.assumption_id for verdict in validation.verdicts]
    assert set(ids) <= set(ASSUMPTION_IDS), f"verdicts reference unknown assumption ids: {ids}"
    assert set(ids) == set(ASSUMPTION_IDS), f"expected a verdict for each of {ASSUMPTION_IDS}, got {ids}"


def test_verdicts_match_the_planted_ground_truth(validation_result: ValidationsOutput) -> None:
    validation = _only_persona(validation_result, PERSONA_ID)
    actual = _verdicts_by_id(validation)
    if actual != EXPECTED:
        pytest.fail(
            "verdicts disagree with the planted ground truth "
            f"{EXPECTED}.\nFull validation:\n{validation.model_dump_json(indent=2)}"
        )


def test_evidence_shares_content_with_the_answer(validation_result: ValidationsOutput) -> None:
    validation = _only_persona(validation_result, PERSONA_ID)
    checked = 0
    for verdict in validation.verdicts:
        if verdict.verdict == "inconclusive":
            continue
        checked += 1
        evidence = verdict.evidence
        assert len(evidence) >= 15, f"{verdict.assumption_id} evidence is too short: {evidence!r}"
        shared = _content_tokens(evidence) & _content_tokens(ANSWERS[verdict.assumption_id])
        assert len(shared) >= 2, (
            f"{verdict.assumption_id} evidence shares only {sorted(shared)} with the answer.\n"
            f"Evidence: {evidence!r}\nAnswer: {ANSWERS[verdict.assumption_id]!r}"
        )
    assert checked >= 2, "expected at least A1 and A2 to be non-inconclusive"


def test_evidence_adds_to_the_assumption_and_avoids_research(validation_result: ValidationsOutput) -> None:
    validation = _only_persona(validation_result, PERSONA_ID)
    for verdict in validation.verdicts:
        evidence = verdict.evidence
        leaked = _content_tokens(evidence) & RESEARCH_WORDS
        assert not leaked, f"{verdict.assumption_id} evidence mentions research sources: {sorted(leaked)} in {evidence!r}"
        if verdict.verdict == "inconclusive":
            continue
        statement_tokens = _content_tokens(STATEMENTS[verdict.assumption_id])
        new_tokens = _content_tokens(evidence) - statement_tokens
        assert len(new_tokens) >= 2, (
            f"{verdict.assumption_id} evidence mostly restates the assumption. "
            f"New content: {sorted(new_tokens)}. Evidence: {evidence!r}"
        )


def test_empty_interview_gives_only_inconclusive_verdicts(empty_validation_result: ValidationsOutput) -> None:
    validation = _only_persona(empty_validation_result, EMPTY_PERSONA_ID)
    assert_unique_ids(validation.verdicts, attr="assumption_id")
    dump = validation.model_dump_json(indent=2)
    assert {verdict.assumption_id for verdict in validation.verdicts} == set(ASSUMPTION_IDS), dump
    for verdict in validation.verdicts:
        assert verdict.verdict == "inconclusive", f"{verdict.assumption_id} is {verdict.verdict} with no answers:\n{dump}"
        assert NOT_ENOUGH_EVIDENCE.search(verdict.evidence.lower()), (
            f"{verdict.assumption_id} evidence does not say why there is no support: {verdict.evidence!r}"
        )
