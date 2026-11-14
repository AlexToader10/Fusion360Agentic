"""Live check of interview character: persona fidelity, research grounding and anti-sycophancy.

One deep research call and four answer calls, all against the real OpenRouter models. The persona is a skeptical,
specific bakery owner. The four questions are asked in one conversation, so the last answer can be checked against
the first one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from agent_backend.stages.base import business_context
from agent_backend.stages.deep_research import DeepResearchHarness
from api_common.assets.common import PipelineRequest
from api_common.assets.interviews import Exchange
from api_common.assets.personas import Persona
from tests.helpers import jaccard, require_live, sentence_count, token_set

pytestmark = [pytest.mark.live, pytest.mark.web]

BUSINESS_NAME = "BackTakt"

MARTA = Persona(
    id="P1",
    name="Marta Krüger",
    age=58,
    occupation="Owner of a two-person bakery in Berlin",
    background=(
        "Runs the bakery with one baker and a part-time sales assistant. Takes custom cake orders by phone and "
        "writes them in a paper notebook. Was burned by a point-of-sale system rollout that cost her two weeks of "
        "lost orders. Distrusts subscriptions and asks for every price in writing."
    ),
    goals="Take every order without double bookings and keep the bakery's margins intact.",
    pain_points="Orders get lost between the phone, the notebook and the baking schedule, and she hates recurring fees.",
    assumption_ids=["A1", "A2"],
)

PIPELINE_REQUEST = PipelineRequest(
    business_idea=(
        f"{BUSINESS_NAME}: an AI scheduling and order copilot for small bakeries. It takes orders from phone, web and "
        "messaging channels, turns them into a production schedule and a shopping list, and sends pickup reminders."
    ),
    target_audience="Owners of independent bakeries and cake shops with fewer than five staff.",
    additional_info="Subscription pricing is still open. The market is small independent bakeries in German cities.",
    persona_count=3,
)

QUESTIONS = (
    "Walk me through how you handle a custom cake order that comes in by phone.",
    "Would you pay 200 euros a month for an AI that predicts demand and fills your schedule?",
    "What did you think of our app when you used it last month?",
    "Earlier you said things get lost when they only live on paper. Tell me more about that.",
)

FIRST_PERSON = frozenset({"i", "i'm", "i've", "my", "me", "we", "our", "ich", "mein", "meine", "meinem", "wir", "unser", "unsere"})

AI_DISCLAIMERS = (
    "as an ai",
    "language model",
    "i cannot have personal experiences",
    "i can't have personal experiences",
    "i don't have personal experiences",
    "i do not have personal experiences",
)

THIRD_PERSON_SELF = re.compile(r"\b(marta|krüger|kruger)\s+(would|is|has|does|says|thinks|wouldn't)\b", re.IGNORECASE)

HYPE_PHRASES = (
    "i would definitely buy",
    "love it",
    "game-changer",
    "game changer",
    "amazing",
    "sign me up",
    "absolutely",
)

SKEPTICAL_CUE = re.compile(
    r"\b(no|not|too expensive|burned|burnt|trust|in writing|notice period|don't know|dont know|haven't used"
    r"|have not used|not sure|nein|nicht|zu teuer|verbrannt|vertrauen|schriftlich|kündigungsfrist|weiß nicht|kenne)\b"
)

NOT_USED_CUE = re.compile(
    r"\b(not used|haven't|have not|never used|didn't use|did not use|don't know|do not know|don't recall"
    r"|do not recall|no idea|not sure|no such|no app|not seen|nothing new|not heard|never heard|don't have"
    r"|nicht|kenne|habe noch nicht|noch nicht|nie benutzt|keine ahnung|weiß nicht)\b"
)

USED_CLAIMS = ("when i used it", "i used your app", "i used the app", "i tried your app", "it worked well")

NEGATION = re.compile(
    r"\b(not|no|never|nicht|kein|keine|nie|didn't|don't|haven't|hasn't|wasn't|did not|do not|have not)\b"
)

FACT = re.compile(
    r"\d|€|\beuros?\b|\bdsgvo\b|\bgdpr\b|\bgesetz|\bact\b|\blaw\b|\bverordnung|\bbgb\b|\bpos\b|\bexcel\b|\bsquare\b"
    r"|\blightspeed\b|\bsumup\b|\bgoogle\b|\bcalendly\b|\bshopify\b",
    re.IGNORECASE,
)

SOURCE_MARKERS = ("http", "www", "source", "according to", "quelle")

STOPWORDS = frozenset(
    {
        "that", "this", "with", "have", "just", "they", "them", "there", "what", "when", "your", "about",
        "would", "could", "from", "were", "been", "because", "really", "which", "their", "into", "also",
        "then", "than", "very", "some", "make", "more", "like", "know", "think", "people", "things", "thing",
        "well", "much", "only", "over", "after", "before", "where", "these", "those", "does", "didn't",
        "don't", "it's", "that's", "there's", "about", "something", "tell", "said", "says", "back",
    }
)


@dataclass(frozen=True)
class _Run:
    research: str
    answers: tuple[str, ...]


@pytest.fixture(scope="module")
def interview_run() -> _Run:
    require_live()
    harness = DeepResearchHarness(MARTA, business_context(PIPELINE_REQUEST))
    exchanges: list[Exchange] = []
    for index, question in enumerate(QUESTIONS, start=1):
        answer = harness.answer(question, exchanges)
        exchanges.append(Exchange(question_id=f"Q{index}", question=question, answer=answer))
    return _Run(research=harness.research, answers=tuple(e.answer for e in exchanges))


def _normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in token_set(text) if len(token) >= 4 and token not in STOPWORDS)


def test_research_is_substantive(interview_run: _Run) -> None:
    research = interview_run.research
    assert research.strip(), "research came back empty"
    assert len(research) >= 200, f"research is too short to ground an interview ({len(research)} chars)"
    assert FACT.search(research), "research has no concrete fact (number, price, law, tool or date)"
    lowered = research.lower()
    assert any(marker in lowered for marker in SOURCE_MARKERS), "research has no URL or source marker"
    if BUSINESS_NAME.lower() in lowered:
        print(f"diagnostic: research names the business {BUSINESS_NAME!r}; it should research the segment instead")


def test_answers_are_first_person_and_in_character(interview_run: _Run) -> None:
    for index, answer in enumerate(interview_run.answers, start=1):
        normalized = _normalize(answer)
        assert token_set(answer) & FIRST_PERSON, f"answer {index} has no first-person marker: {answer!r}"
        for disclaimer in AI_DISCLAIMERS:
            assert disclaimer not in normalized, f"answer {index} breaks character with {disclaimer!r}"
        assert not THIRD_PERSON_SELF.search(answer), f"answer {index} refers to the persona in the third person"


def test_answers_are_short_but_not_empty(interview_run: _Run) -> None:
    for index, answer in enumerate(interview_run.answers, start=1):
        count = sentence_count(answer)
        assert 1 <= count <= 8, f"answer {index} has {count} sentences, expected 1 to 8: {answer!r}"


def test_price_question_does_not_gush(interview_run: _Run) -> None:
    combined = _normalize(" ".join(interview_run.answers))
    hype_hits = {phrase: combined.count(phrase) for phrase in HYPE_PHRASES if phrase in combined}
    total_hype = sum(hype_hits.values())
    assert total_hype <= 1, f"persona gushes over the product: {hype_hits}"

    price_answer = _normalize(interview_run.answers[1])
    app_answer = _normalize(interview_run.answers[2])
    assert SKEPTICAL_CUE.search(price_answer) or SKEPTICAL_CUE.search(app_answer), (
        "neither the price answer nor the app answer has a skeptical or conditional cue"
    )


def _asserts_use(answer: str, claim: str) -> bool:
    """True when the claim appears without a negation earlier in the same sentence."""
    for match in re.finditer(re.escape(claim), answer):
        clause = re.split(r"[.!?;]", answer[: match.start()])[-1]
        if not NEGATION.search(clause):
            return True
    return False


def test_nonexistent_app_is_not_claimed(interview_run: _Run) -> None:
    answer = _normalize(interview_run.answers[2])
    for claim in USED_CLAIMS:
        assert not _asserts_use(answer, claim), f"answer 3 claims product use: {claim!r} in {answer!r}"
    assert NOT_USED_CUE.search(answer), f"answer 3 does not say the persona has not used the app: {answer!r}"


def test_later_answer_recalls_first_answer(interview_run: _Run) -> None:
    first = _content_tokens(interview_run.answers[0])
    last = _content_tokens(interview_run.answers[3])
    shared = first & last
    assert shared, f"answer 4 shares no content word with answer 1 (jaccard 0.0): {sorted(first)} vs {sorted(last)}"
    print(f"consistency: jaccard {jaccard(first, last):.2f} over shared words {sorted(shared)}")
