"""Live check of the interviewer's question-picking policy, separate from the persona actor.

Three calls reach the real interviewer model, each made once by a module-scoped fixture:
1. all six questions remain, three relevant and three absurd, and nothing has been asked yet;
2. the same persona with only the absurd questions left, which should end the interview;
3. the three relevant questions were already answered and one relevant follow-up remains.

The persona model never runs, so the answers in the conversation are fixed text. Nothing here may call the model
again, so a test that needs new data must go through a fixture.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pytest

from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.interviews import INTERVIEWER_SYSTEM
from api_common.assets.assumptions import Assumption, AssumptionsOutput
from api_common.assets.interviews import Exchange, QuestionPick
from api_common.assets.personas import Persona
from api_common.assets.questions import Question, QuestionsOutput
from tests.helpers import require_live

pytestmark = [pytest.mark.live]

_ID_SHAPE = re.compile(r"Q\d+")

_PERSONA = Persona(
    id="P1",
    name="Hanna Vogel",
    age=52,
    occupation="Owner of an independent bakery",
    background=(
        "Runs a two-person bakery with a counter and a wholesale delivery to two cafés. Keeps every order in a "
        "paper notebook by the till. Has no interest in new software and distrusts subscriptions."
    ),
    goals="Get every custom cake order out on time without double bookings.",
    pain_points="Notes get smudged or lost in the Saturday rush, and phone orders are written on scraps of paper.",
    assumption_ids=["A1", "A2", "A3"],
)

_ASSUMPTIONS = AssumptionsOutput(
    assumptions=[
        Assumption(
            id="A1",
            statement="Small bakeries lose custom orders that are written on paper notebooks.",
            category="problem",
            rationale="Without a felt problem, no owner pays for order software.",
        ),
        Assumption(
            id="A2",
            statement="Phone orders are a main way customers place cake orders with small bakeries.",
            category="customer",
            rationale="The product has to fit how orders actually arrive.",
        ),
        Assumption(
            id="A3",
            statement="Bakery owners will pay a monthly fee for a shared order list.",
            category="willingness to pay",
            rationale="The margin of a small bakery cannot absorb a large software cost.",
        ),
    ]
)

_RELEVANT = [
    Question(
        id="Q1",
        text=(
            "When a customer orders a birthday cake, where do you write the order down, "
            "and what happens to that note on a busy Saturday?"
        ),
        assumption_ids=["A1"],
    ),
    Question(
        id="Q2",
        text="How do most of your customers place orders with you, and what do you do when a phone order comes in during a rush?",
        assumption_ids=["A2"],
    ),
    Question(
        id="Q3",
        text=(
            "What do you spend on bakery tools and software in a year, "
            "and who decides when a new tool is worth buying?"
        ),
        assumption_ids=["A3"],
    ),
]

_ABSURD = [
    Question(
        id="Q4",
        text=(
            "What is your deep-sea welding certification level, "
            "and how many hours a month do you spend welding below 300 metres?"
        ),
        assumption_ids=["A1"],
    ),
    Question(
        id="Q5",
        text="How do you keep relative humidity steady inside a lunar greenhouse module during the lunar night?",
        assumption_ids=["A2"],
    ),
    Question(
        id="Q6",
        text=(
            "How does your enterprise HR policy handle parental leave approvals "
            "across regional subsidiaries and their works councils?"
        ),
        assumption_ids=["A3"],
    ),
]

_FOLLOW_UP = Question(
    id="Q7",
    text=(
        "Think of the last time an order was lost or double-booked. What did it cost you, "
        "and would one shared order list have prevented it?"
    ),
    assumption_ids=["A1", "A2"],
)

_RELEVANT_IDS = frozenset(question.id for question in _RELEVANT)
_ABSURD_IDS = frozenset(question.id for question in _ABSURD)
_FOLLOW_UP_IDS = frozenset([_FOLLOW_UP.id])
_TEXT_BY_ID = {question.id: question.text for question in [*_RELEVANT, *_ABSURD, _FOLLOW_UP]}

_ANSWERED = [
    Exchange(
        question_id="Q1",
        question=_RELEVANT[0].text,
        answer="Mostly on a scrap of paper by the till, sometimes on the back of a delivery note. On Saturdays it gets buried.",
    ),
    Exchange(
        question_id="Q2",
        question=_RELEVANT[1].text,
        answer="Half the cake orders come by phone. When it rings during a rush I write it down and hope I read my own handwriting later.",
    ),
    Exchange(
        question_id="Q3",
        question=_RELEVANT[2].text,
        answer="Not much beyond the oven service and the till. I do not buy software, I have never seen the point.",
    ),
]


def _transcript(exchanges: Sequence[Exchange]) -> str:
    return "\n\n".join(f"{e.question}\nAnswer: {e.answer}" for e in exchanges)


def _prompt(exchanges: Sequence[Exchange], remaining: Sequence[Question]) -> str:
    return (
        f"Persona:\n{_PERSONA.model_dump_json(indent=2)}\n\n"
        f"Assumptions:\n{_ASSUMPTIONS.model_dump_json(indent=2)}\n\n"
        f"Conversation so far:\n{_transcript(exchanges) or 'Nothing asked yet.'}\n\n"
        f"Remaining questions:\n{QuestionsOutput(questions=list(remaining)).model_dump_json(indent=2)}"
    )


def _pick(exchanges: Sequence[Exchange], remaining: Sequence[Question]) -> QuestionPick:
    return OpenRouterClient("interviewer").parse(INTERVIEWER_SYSTEM, _prompt(exchanges, remaining), QuestionPick)


def _describe(pick: QuestionPick) -> str:
    text = _TEXT_BY_ID.get(pick.question_id or "", "<no question>")
    return f"full response: {pick.model_dump_json()}\npicked question text: {text}"


@pytest.fixture(scope="module")
def relevant_pick() -> QuestionPick:
    require_live()
    return _pick([], [*_RELEVANT, *_ABSURD])


@pytest.fixture(scope="module")
def absurd_pick() -> QuestionPick:
    require_live()
    return _pick([], _ABSURD)


@pytest.fixture(scope="module")
def follow_up_pick() -> QuestionPick:
    require_live()
    return _pick(_ANSWERED, [_FOLLOW_UP])


def test_picks_one_of_the_relevant_questions(relevant_pick: QuestionPick) -> None:
    assert relevant_pick.question_id is not None, (
        f"interviewer ended the interview although relevant questions remained\n{_describe(relevant_pick)}"
    )
    assert relevant_pick.question_id in _RELEVANT_IDS, (
        f"interviewer picked {relevant_pick.question_id!r}, which is not one of the relevant questions "
        f"{sorted(_RELEVANT_IDS)}\n{_describe(relevant_pick)}"
    )


def test_absurd_only_remaining_ends_the_interview(absurd_pick: QuestionPick) -> None:
    assert absurd_pick.question_id is None, (
        "interviewer forced an irrelevant question instead of ending the interview\n"
        f"expected question_id None, got {absurd_pick.question_id!r}\n{_describe(absurd_pick)}"
    )


def test_pick_is_always_a_remaining_id(
    relevant_pick: QuestionPick,
    absurd_pick: QuestionPick,
    follow_up_pick: QuestionPick,
) -> None:
    checks = [
        (relevant_pick, frozenset(_RELEVANT_IDS | _ABSURD_IDS)),
        (absurd_pick, _ABSURD_IDS),
        (follow_up_pick, _FOLLOW_UP_IDS),
    ]
    for pick, remaining_ids in checks:
        if pick.question_id is None:
            continue
        assert pick.question_id in remaining_ids, (
            f"interviewer returned {pick.question_id!r}, which is not among the remaining ids "
            f"{sorted(remaining_ids)}; invented ids are not allowed\n{_describe(pick)}"
        )


def test_pick_is_a_bare_id_not_prose(
    relevant_pick: QuestionPick,
    absurd_pick: QuestionPick,
    follow_up_pick: QuestionPick,
) -> None:
    for pick in (relevant_pick, absurd_pick, follow_up_pick):
        if pick.question_id is None:
            continue
        assert _ID_SHAPE.fullmatch(pick.question_id), (
            f"question_id {pick.question_id!r} is not a bare id such as Q3; the model wrote prose or an invented format\n"
            f"{_describe(pick)}"
        )


def test_follow_up_is_picked_or_interview_ends(follow_up_pick: QuestionPick) -> None:
    assert follow_up_pick.question_id in {_FOLLOW_UP.id, None}, (
        f"with only the relevant follow-up {_FOLLOW_UP.id} remaining, the interviewer must pick it or return null; "
        f"got {follow_up_pick.question_id!r}\n{_describe(follow_up_pick)}"
    )
