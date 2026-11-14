"""Shared fixtures and the offline/live gate.

A plain `pytest` run is offline. Tests that are not marked live or web cannot open network connections, and they
see a dummy OPENROUTER_API_KEY. Tests marked live or web run only with RUN_LIVE_TESTS=1 and OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
import socket

import pytest
from dotenv import load_dotenv

from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.conclusions import ConclusionsOutput
from api_common.assets.interviews import Exchange, Interview, InterviewsOutput
from api_common.assets.personas import PersonasOutput
from api_common.assets.questions import QuestionsOutput
from api_common.assets.validations import AssumptionVerdict, InterviewValidation, ValidationsOutput, Verdict
from tests.helpers import (
    live_skip_reason,
    make_assumption,
    make_conclusion,
    make_persona,
    make_question,
)

load_dotenv()

MARKERS = {
    "live": "calls the real OpenRouter models; costs credits and is stochastic",
    "web": "live test that also uses OpenRouter web search",
    "slow": "takes several seconds or more even when offline",
    "e2e": "drives the whole pipeline from request to conclusions",
}


def pytest_configure(config: pytest.Config) -> None:
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_report_header(config: pytest.Config) -> str | None:
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return "live tests enabled: they call OpenRouter and cost credits"
    return None


def _needs_live(item: pytest.Item) -> bool:
    return item.get_closest_marker("live") is not None or item.get_closest_marker("web") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if _needs_live(item) and (reason := live_skip_reason()) is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


_connect = socket.socket.connect


def _refuse_connection(self: socket.socket, *args: object, **kwargs: object) -> None:
    if self.family == socket.AF_UNIX:  # local IPC: the queue the pipeline shares with its stage processes
        return _connect(self, *args, **kwargs)
    raise RuntimeError(
        "network access is blocked in offline tests. Mark the test live or web, or request the allow_network fixture"
    )


@pytest.fixture
def allow_network() -> None:
    """Opt out of the network block, e.g. for a test that talks to a local server on localhost."""


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if _needs_live(request.node) or "allow_network" in request.fixturenames:
        return
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "offline/test-model")
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_connection)


@pytest.fixture
def sample_request() -> PipelineRequest:
    return PipelineRequest(
        business_idea=(
            "Weekly meal kits that pre-portion locally sourced ingredients and a recipe that can be cooked "
            "in 20 minutes, so busy households stop improvising dinner on school nights."
        ),
        target_audience=(
            "Dual-income parents in suburban areas with children aged 4 to 12, who cook dinner at home "
            "at least three nights a week."
        ),
        additional_info="Pilot in two suburbs first. Target price is about 9 EUR per portion.",
        persona_count=3,
    )


@pytest.fixture
def sample_assumptions() -> AssumptionsOutput:
    return AssumptionsOutput(
        assumptions=[
            make_assumption(
                "A1",
                category="problem",
                statement="Dual-income parents find weeknight dinner planning stressful enough to pay to reduce it.",
                rationale="Without a felt problem, nobody pays for the kit.",
            ),
            make_assumption(
                "A2",
                category="customer",
                statement="Parents of children aged 4 to 12 in suburban areas plan the weekly meals themselves.",
                rationale="The buyer must be the person who decides the menu, or the kit reaches the wrong person.",
            ),
            make_assumption(
                "A3",
                category="solution",
                statement="A pre-portioned kit with a recipe under 20 minutes is preferred to planning and shopping alone.",
                rationale="The product only beats cooking from scratch if the time saved is real.",
            ),
            make_assumption(
                "A4",
                category="willingness to pay",
                statement="Households will pay about 9 EUR per portion for the kit.",
                rationale="At this price the margin funds the local sourcing.",
            ),
            make_assumption(
                "A5",
                category="channel",
                statement="Parents hear about and sign up for meal kits through school or neighbourhood groups.",
                rationale="Paid social ads cost more than the margin of a weekly box.",
            ),
            make_assumption(
                "A6",
                category="retention",
                statement="Subscribers still order after the fourth week without a discount.",
                rationale="Acquisition only pays off if customers stay for several months.",
            ),
        ]
    )


@pytest.fixture
def sample_personas() -> PersonasOutput:
    return PersonasOutput(
        personas=[
            make_persona(
                "P1",
                ["A1", "A2", "A3"],
                name="Laura Meyer",
                age=41,
                occupation="Project manager",
                background="Two children aged 6 and 9, both parents work full time, shops once a week at a supermarket.",
                goals="Put a healthy dinner on the table within 30 minutes on school nights.",
                pain_points="Plans collapse by Wednesday, and she often orders takeaway she regrets.",
            ),
            make_persona(
                "P2",
                ["A1", "A4", "A5"],
                name="Tomas Berg",
                age=36,
                occupation="Nurse on rotating shifts",
                background="His partner also works shifts. He cooks for two children aged 5 and 11 on irregular evenings.",
                goals="Cook fresh food without planning a week in advance.",
                pain_points="Irregular shifts make any fixed plan useless, and delivery fees add up.",
            ),
            make_persona(
                "P3",
                ["A2", "A6"],
                name="Nadia Haddad",
                age=48,
                occupation="Freelance accountant",
                background="Works from home and feeds a teenager and her retired father. Sceptical of subscriptions.",
                goals="Keep grocery spending predictable.",
                pain_points="Cancels subscriptions she does not use, and dislikes packaging waste.",
            ),
        ]
    )


@pytest.fixture
def sample_questions() -> QuestionsOutput:
    return QuestionsOutput(
        questions=[
            make_question("Q1", ["A1"], text="Walk me through how you decided what to cook for dinner last week."),
            make_question(
                "Q2",
                ["A2"],
                text="Who usually makes the weekly food plan in your household, and how did that go last time?",
            ),
            make_question(
                "Q3",
                ["A3"],
                text="Tell me about the last time you tried a new recipe on a weeknight. What happened?",
            ),
            make_question(
                "Q4",
                ["A4"],
                text=(
                    "What did you spend on takeaway or convenience meals in the past month, "
                    "and how did you decide that was worth it?"
                ),
            ),
            make_question(
                "Q5",
                ["A5"],
                text="Where did you last buy an ingredient you had not planned to buy, and how did you hear about that shop?",
            ),
            make_question(
                "Q6",
                ["A6", "A2"],
                text="Think of a food delivery service you stopped using. What made you stop?",
            ),
        ]
    )


@pytest.fixture
def sample_interviews() -> InterviewsOutput:
    return InterviewsOutput(
        interviews=[
            Interview(
                persona_id="P1",
                research=(
                    "Suburban dual-income households often batch-plan dinners on Sunday and rely on supermarket "
                    "delivery. Subscription meal kits are known but seen as expensive."
                ),
                exchanges=[
                    Exchange(
                        question_id="Q1",
                        question="Walk me through how you decided what to cook for dinner last week.",
                        answer="On Sunday I wrote a list, and by Wednesday half of it was abandoned. Takeaway filled the gap.",
                    ),
                    Exchange(
                        question_id="Q3",
                        question="Tell me about the last time you tried a new recipe on a weeknight. What happened?",
                        answer="I tried a curry on a Tuesday. It took over an hour and the kids would not eat it.",
                    ),
                    Exchange(
                        question_id="Q4",
                        question=(
                            "What did you spend on takeaway or convenience meals in the past month, "
                            "and how did you decide that was worth it?"
                        ),
                        answer="Probably around 150 EUR. I did not think about it much, it just happens.",
                    ),
                ],
            ),
            Interview(
                persona_id="P2",
                research=(
                    "Shift workers in the region buy ready meals on late shifts and use neighbourhood parent "
                    "groups to find recommendations."
                ),
                exchanges=[
                    Exchange(
                        question_id="Q1",
                        question="Walk me through how you decided what to cook for dinner last week.",
                        answer="Honestly, I bought ready meals on the nights I was on late shifts. Cooking feels like a chore then.",
                    ),
                    Exchange(
                        question_id="Q4",
                        question=(
                            "What did you spend on takeaway or convenience meals in the past month, "
                            "and how did you decide that was worth it?"
                        ),
                        answer="Delivery fees alone are a lot. Paying 9 euros for one portion would be a stretch for us.",
                    ),
                    Exchange(
                        question_id="Q5",
                        question=(
                            "Where did you last buy an ingredient you had not planned to buy, "
                            "and how did you hear about that shop?"
                        ),
                        answer="A parents group in our neighbourhood recommended a small bakery. I went there the same week.",
                    ),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_validations() -> ValidationsOutput:
    def verdicts(persona_id: str, rows: list[tuple[str, Verdict, str]]) -> InterviewValidation:
        return InterviewValidation(
            persona_id=persona_id,
            verdicts=[
                AssumptionVerdict(assumption_id=assumption_id, verdict=verdict, evidence=evidence)
                for assumption_id, verdict, evidence in rows
            ],
        )

    return ValidationsOutput(
        validations=[
            verdicts(
                "P1",
                [
                    ("A1", "validated", "Said Sunday planning is a stressful ritual that fails by midweek."),
                    ("A2", "validated", "Said she plans the week's meals and does the shopping."),
                    ("A3", "inconclusive", "Did not say how cooking time fits her routine with a kit."),
                    ("A4", "inconclusive", "Mentioned takeaway spending but gave no price per portion."),
                    ("A5", "inconclusive", "Did not say how she hears about new food services."),
                    ("A6", "inconclusive", "Was not asked about ordering beyond the first weeks."),
                ],
            ),
            verdicts(
                "P2",
                [
                    ("A1", "validated", "Said late-shift dinners are bought ready-made because cooking feels like a chore."),
                    ("A2", "inconclusive", "Did not say who plans the household meals."),
                    ("A3", "inconclusive", "Did not discuss cooking time for a home-made meal."),
                    ("A4", "invalidated", "Said 9 EUR per portion would be a stretch for the household."),
                    ("A5", "validated", "Learned about a shop through a neighbourhood parents group."),
                    ("A6", "inconclusive", "Was not asked about repeat orders."),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_conclusions() -> ConclusionsOutput:
    return ConclusionsOutput(
        summary=(
            "The weeknight dinner problem is real for dual-income parents, but price and acquisition channel "
            "remain open. The two simulated interviews support the problem and the customer and contradict "
            "the price for at least one household."
        ),
        assumptions=[
            make_conclusion("A1", "validated", "Both interviews describe unplanned weeknight dinners."),
            make_conclusion("A2", "validated", "Laura plans the meals herself, but Tomas gave no evidence either way."),
            make_conclusion("A3", "inconclusive", "Cooking time for a kit was never discussed."),
            make_conclusion("A4", "inconclusive", "One interview rejects 9 EUR per portion, the other gives no price."),
            make_conclusion("A5", "inconclusive", "One interview points to neighbourhood groups, the other is silent."),
            make_conclusion("A6", "inconclusive", "Retention was not explored in the interviews."),
        ],
        recommendations=[
            "Test price points with real customers before committing to 9 EUR per portion.",
            "Interview parents about how they find new food services, to validate the channel assumption.",
            "Run a short pilot in one suburb to measure repeat orders, since simulated interviews cannot show retention.",
        ],
    )


@pytest.fixture
def full_state(
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_personas: PersonasOutput,
    sample_questions: QuestionsOutput,
    sample_interviews: InterviewsOutput,
    sample_validations: ValidationsOutput,
    sample_conclusions: ConclusionsOutput,
) -> PipelineState:
    return PipelineState(
        request=sample_request,
        assumptions=sample_assumptions,
        personas=sample_personas,
        questions=sample_questions,
        interviews=sample_interviews,
        validations=sample_validations,
        conclusions=sample_conclusions,
    )


@pytest.fixture
def empty_state(sample_request: PipelineRequest) -> PipelineState:
    return PipelineState(request=sample_request)
