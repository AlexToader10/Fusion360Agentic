"""Live red-team checks of the questions generator and its bias judge.

Part A feeds a handcrafted set of leading and neutral questions straight to the real judge. Part B runs the real
assumptions and questions stages on an idea that carries a unique product canary, then checks the generated
questions for pitching, hypothetical framing, leading openers and missing coverage.

Budget: one questions-stage run, plus two direct judge calls (Part A and the final pass of Part B).
"""

from __future__ import annotations

import re

import pytest

from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.questions import MAX_JUDGE_ROUNDS, QuestionsStage
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.questions import Question, QuestionsOutput
from tests.helpers import extract_ids

pytestmark = [pytest.mark.live]

CANARY = "ChairFill 3000"

LEADING_TEXTS = [
    "Don't you find it frustrating that scheduling takes so long?",
    "Wouldn't you pay $50 a month for a tool that saves you ten hours?",
    "How much would you love an AI that answers your phone?",
    "You'd agree that missed calls cost you money, right?",
    "Most dentists say they lose patients to voicemail. Do you?",
    "If we built a waitlist app, would you use it?",
]

NEUTRAL_TEXTS = [
    "Tell me about your scheduling process.",
    "What tools does your front desk use, if any?",
    "Who is involved in buying decisions at the practice?",
    "What happens when someone calls the practice?",
    "Describe a typical morning at the practice.",
    "What do you wish were different about your day, if anything?",
]

LEADING_IDS = [f"Q{n}" for n in range(1, 7)]
NEUTRAL_IDS = [f"Q{n}" for n in range(7, 13)]

MIN_LEADING_RECALL = 5
MAX_NEUTRAL_FALSE_POSITIVES = 1
MAX_REMAINING_SUGGESTIONS = 4

FIRST_PERSON_PITCH = re.compile(r"\b(we|our)\b", re.IGNORECASE)
FIRST_PERSON_I = re.compile(r"\bI\b")
# "would" and "could" count only before a person or it, as in "would you use it". Past-tense phrasing such as
# "what did you hope would happen" is not future framing, so a bare "would" is allowed.
HYPOTHETICAL = re.compile(
    r"\b(wouldn't|couldn't|would|could)\s+(you|we|i|it|they)\b|\bif we\b|\bimagine\b|\bwhat if\b",
    re.IGNORECASE,
)
LEADING_OPENERS = ("don't you", "wouldn't you", "you'd agree", "how much would you love", "right?")
PAST_OR_CONCRETE = re.compile(
    r"\b(last time|recently|walk me through|tell me about|what did|how did|who|what happens|describe"
    r"|have you|did you|do you|how do you|what do you|what changes)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'").strip()


def _question_ids(suggestion: str) -> list[str]:
    return extract_ids(suggestion)["Q"]


@pytest.fixture(scope="module")
def handcrafted_bias_set() -> QuestionsOutput:
    leading = [
        Question(id=f"Q{n}", text=text, assumption_ids=["A1"]) for n, text in enumerate(LEADING_TEXTS, start=1)
    ]
    neutral = [
        Question(id=f"Q{n}", text=text, assumption_ids=["A2"])
        for n, text in enumerate(NEUTRAL_TEXTS, start=len(LEADING_TEXTS) + 1)
    ]
    return QuestionsOutput(questions=leading + neutral)


@pytest.fixture(scope="module")
def generated_state() -> PipelineState:
    request = PipelineRequest(
        business_idea=(
            f"{CANARY}: a monthly subscription that rents ergonomic office chairs to home workers and swaps "
            "them out when back pain starts."
        ),
        target_audience="Remote workers aged 25 to 55 who work from home at least three days a week.",
        additional_info="Pilot in one city first. Target price is about 60 EUR per month.",
    )
    state = PipelineState(request=request)
    state = AssumptionsStage().run(state)
    return QuestionsStage().run(state)


# Part A: judge recall and restraint


def test_judge_flags_leading_questions_and_spares_neutral_ones(handcrafted_bias_set: QuestionsOutput) -> None:
    suggestions = QuestionsStage._judge(OpenRouterClient("questions_judge"), handcrafted_bias_set)

    for suggestion in suggestions:
        assert _question_ids(suggestion), f"judge suggestion names no question id: {suggestion!r}"

    flagged = {ident for suggestion in suggestions for ident in _question_ids(suggestion)}
    leading_hits = [ident for ident in LEADING_IDS if ident in flagged]
    neutral_flags = [ident for ident in NEUTRAL_IDS if ident in flagged]

    assert len(leading_hits) >= MIN_LEADING_RECALL, (
        f"judge caught {len(leading_hits)}/6 leading questions, needs {MIN_LEADING_RECALL}. "
        f"Missed: {[i for i in LEADING_IDS if i not in flagged]}. Suggestions: {suggestions}"
    )
    assert len(neutral_flags) <= MAX_NEUTRAL_FALSE_POSITIVES, (
        f"judge flagged {len(neutral_flags)}/6 neutral questions as leading: {neutral_flags}. Suggestions: {suggestions}"
    )


# Part B: generator quality


def test_generated_question_count_ids_and_coverage(generated_state: PipelineState) -> None:
    questions = generated_state.questions.questions
    assumptions = generated_state.assumptions.assumptions

    assert 6 <= len(questions) <= 10, f"expected 6 to 10 questions, got {len(questions)}"

    ids = [q.id for q in questions]
    duplicates = sorted({ident for ident in ids if ids.count(ident) > 1})
    assert not duplicates, f"duplicate question ids: {duplicates}"

    real_assumption_ids = {a.id for a in assumptions}
    unknown = sorted({a for q in questions for a in q.assumption_ids} - real_assumption_ids)
    assert not unknown, f"questions reference unknown assumption ids: {unknown}"

    covered = {a for q in questions for a in q.assumption_ids}
    uncovered = sorted(real_assumption_ids - covered)
    assert not uncovered, f"assumptions not tested by any question: {uncovered}"


def test_generated_questions_avoid_pitch_hypotheticals_and_leading_openers(generated_state: PipelineState) -> None:
    questions = generated_state.questions.questions

    canary_hits = [q.id for q in questions if "chairfill" in _normalise(q.text).lower()]
    assert not canary_hits, f"questions mention the product canary {CANARY!r}: {canary_hits}"

    pitch_hits = [
        (q.id, q.text)
        for q in questions
        if FIRST_PERSON_PITCH.search(_normalise(q.text)) or FIRST_PERSON_I.search(_normalise(q.text))
    ]
    assert not pitch_hits, f"questions use first-person pitch words: {pitch_hits}"

    hypothetical_hits = [(q.id, q.text) for q in questions if HYPOTHETICAL.search(_normalise(q.text))]
    assert not hypothetical_hits, f"questions use hypothetical or future framing: {hypothetical_hits}"

    opener_hits = [
        (q.id, q.text)
        for q in questions
        if any(opener in _normalise(q.text).lower() for opener in LEADING_OPENERS)
    ]
    assert not opener_hits, f"questions open with a leading yes/no: {opener_hits}"


def test_generated_questions_ask_about_past_behaviour_and_end_with_question_mark(
    generated_state: PipelineState,
) -> None:
    questions = generated_state.questions.questions
    total = len(questions)

    concrete = [q for q in questions if PAST_OR_CONCRETE.search(_normalise(q.text))]
    assert len(concrete) / total >= 0.8, (
        f"only {len(concrete)}/{total} questions ask about past behaviour or concrete experience. "
        f"Others: {[(q.id, q.text) for q in questions if q not in concrete]}"
    )

    with_mark = [q for q in questions if _normalise(q.text).endswith("?")]
    assert len(with_mark) / total >= 0.8, (
        f"only {len(with_mark)}/{total} questions end with a question mark. "
        f"Others: {[(q.id, q.text) for q in questions if q not in with_mark]}"
    )


def test_final_judge_pass_converges(generated_state: PipelineState) -> None:
    suggestions = QuestionsStage._judge(OpenRouterClient("questions_judge"), generated_state.questions)

    assert len(suggestions) <= MAX_REMAINING_SUGGESTIONS, (
        f"after {MAX_JUDGE_ROUNDS} edit rounds the judge still returns {len(suggestions)} suggestions "
        f"(max {MAX_REMAINING_SUGGESTIONS}): {suggestions}"
    )
    for suggestion in suggestions:
        assert _question_ids(suggestion), f"judge suggestion names no question id: {suggestion!r}"
