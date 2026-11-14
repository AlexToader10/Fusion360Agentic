"""Live check of research grounding ("The Oracle"): the deep research harness must research the persona's real-world
segment on the web, and the persona's answers must build on that research without inventing sources.

One deep research call and three answer calls, all against the real OpenRouter models. The persona is a 44-year-old
owner of an independent bicycle repair shop in Utrecht who writes job tickets on paper and distrusts subscription
software. The business is a repair-shop scheduling and quote tool.
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
from tests.helpers import jaccard, require_live

pytestmark = [pytest.mark.live, pytest.mark.web]

BUSINESS_NAME = "SpokeDesk"

PIETER = Persona(
    id="P1",
    name="Pieter Jansen",
    age=44,
    occupation="Owner of an independent bicycle repair shop in Utrecht",
    background=(
        "Runs a three-person repair shop near Utrecht Centraal with one mechanic and an apprentice on Saturdays. "
        "Writes every job on a paper werkbon clipped to the bike, and quotes by hand. Tried a booking app years ago "
        "that charged monthly and never synced with his till. Distrusts subscription software and wants every price "
        "in writing."
    ),
    goals="Quote repairs quickly and fit more bikes into the week without double-booking the workshop.",
    pain_points=(
        "Phone quotes get forgotten, customers do not turn up to collect their bikes, and the spring rush makes "
        "the paper board unreadable."
    ),
    assumption_ids=["A1", "A2"],
)

PIPELINE_REQUEST = PipelineRequest(
    business_idea=(
        f"{BUSINESS_NAME}: a scheduling and quote tool for bicycle repair shops. It takes bookings from the web and "
        "the phone, builds a fixed-price quote from a parts and labour list, and sends pickup reminders."
    ),
    target_audience="Owners of independent bicycle repair shops with fewer than five staff in Dutch cities.",
    additional_info="Pricing is still open. The first market is Utrecht and the surrounding region.",
    persona_count=3,
)

QUESTIONS = (
    "Walk me through what happens from the moment a customer brings in a bike with a problem.",
    "Would you pay 25 euros a month for software that books your repair slots and sends out quotes?",
    "Tell me about the last time a repair quote went wrong, or a customer did not come back for their bike.",
)

SOURCE_MARKERS = ("http", "www", "source", "according to", "report")

REFUSAL = re.compile(
    r"\b(i can(?:not|'t)|i'm sorry|i am sorry|i apologi[sz]e|unable to|i'm unable|i am unable|cannot help)\b"
)

VERDICT_PHRASES = (
    "this business will succeed",
    "will succeed",
    "will fail",
    "great idea",
    "good idea",
    "bad idea",
)

ANCHOR = re.compile(
    r"\d+(?:[.,]\d+)?|€|\beuros?\b|\beur\b"
    r"|\b(?:gdpr|avg|btw|bgb|vat|kvk|excel|google|whatsapp|calendly|shopify|square|sumup|quickbooks|moneybird"
    r"|lightspeed)\b",
    re.IGNORECASE,
)

CONSTRAINT_BUCKETS = {
    "cost": re.compile(
        r"\b(cost|costs|price|prices|pricing|fee|fees|subscription|subscriptions|budget|afford|margin|per month"
        r"|per year|euros?|eur|kosten|prijs)\b",
        re.IGNORECASE,
    ),
    "time": re.compile(
        r"\b(time|hours?|minutes?|schedule|scheduling|wait|waiting|lead time|busy|season|spring|weeks?|days?"
        r"|tijd|uur)\b",
        re.IGNORECASE,
    ),
    "regulation": re.compile(
        r"\b(law|laws|regulation|regulations|gdpr|avg|privacy|tax|btw|vat|bgb|warranty|consumer|consumers|rules?"
        r"|wet)\b",
        re.IGNORECASE,
    ),
    "workflow": re.compile(
        r"\b(paper|workflow|tickets?|werkbon|notebook|spreadsheet|excel|whiteboard|board|planning|process"
        r"|manual|till)\b",
        re.IGNORECASE,
    ),
    "customer behaviour": re.compile(
        r"\b(customers?|no-?shows?|cancel|cancell?ations?|reviews?|loyal|loyalty|walk-?ins?|appointments?|book"
        r"|booking|bookings|habits?|klanten)\b",
        re.IGNORECASE,
    ),
}

URL = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+", re.IGNORECASE)

AI_DISCLAIMERS = (
    "as an ai",
    "i'm an ai",
    "i am an ai",
    "language model",
    "i cannot have personal experiences",
    "i don't have personal experiences",
    "i do not have personal experiences",
)

BREAKS_CHARACTER = re.compile(r"\bai\b|\bllm\b|\bchatbot\b|\bsimulat\w*|\bpersona\b|\brole-?play\w*", re.IGNORECASE)

STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "because", "before", "being", "between", "could", "doing", "every",
        "found", "from", "further", "given", "going", "great", "having", "their", "there", "these", "they", "thing",
        "things", "think", "those", "through", "under", "until", "using", "where", "which", "while", "would", "should",
        "really", "people", "something", "because", "customer", "customers", "research", "sources", "source",
    }
)


@dataclass(frozen=True)
class _Run:
    research: str
    answers: tuple[str, ...]


@pytest.fixture(scope="module")
def oracle_run() -> _Run:
    require_live()
    harness = DeepResearchHarness(PIETER, business_context(PIPELINE_REQUEST))
    exchanges: list[Exchange] = []
    for index, question in enumerate(QUESTIONS, start=1):
        answer = harness.answer(question, exchanges)
        exchanges.append(Exchange(question_id=f"Q{index}", question=question, answer=answer))
    return _Run(research=harness.research, answers=tuple(e.answer for e in exchanges))


def _normalize(text: str) -> str:
    return text.replace("\u2019", "'").lower()


def _distinctive(text: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize(text))
    return frozenset(token for token in tokens if len(token) >= 5 and token not in STOPWORDS)


def _profile_text() -> str:
    return " ".join(
        [PIETER.occupation, PIETER.background, PIETER.goals, PIETER.pain_points, PIPELINE_REQUEST.business_idea]
    )


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?").lower()


def _urls(text: str) -> set[str]:
    return {_clean_url(url) for url in URL.findall(text)}


def test_research_is_substantial(oracle_run: _Run) -> None:
    research = oracle_run.research
    assert len(research) >= 300, f"research is too short to ground an actor ({len(research)} chars)"

    normalized = _normalize(research)
    assert not REFUSAL.search(normalized), f"research is a refusal or an apology: {research[:200]!r}"

    anchors = {match.group(0).lower() for match in ANCHOR.finditer(research)}
    assert len(anchors) >= 2, f"research has fewer than two concrete anchors (numbers, prices, tools, laws, dates): {sorted(anchors)}"

    assert any(marker in normalized for marker in SOURCE_MARKERS), "research has no source marker (URL, source, report)"
    print(f"anchors found: {sorted(anchors)}")


def test_research_describes_segment_not_verdict(oracle_run: _Run) -> None:
    normalized = _normalize(oracle_run.research)
    verdicts = [phrase for phrase in VERDICT_PHRASES if phrase in normalized]
    assert not verdicts, f"research judges the business instead of the segment: {verdicts}"

    if BUSINESS_NAME.lower() in normalized:
        print(f"diagnostic: research names the product {BUSINESS_NAME!r}; it should research the segment instead")


def test_research_covers_real_constraints(oracle_run: _Run) -> None:
    normalized = _normalize(oracle_run.research)
    buckets = sorted(name for name, pattern in CONSTRAINT_BUCKETS.items() if pattern.search(normalized))
    assert len(buckets) >= 2, f"research covers fewer than two real constraint types, found {buckets}"
    print(f"constraint buckets covered: {buckets}")


def test_answers_are_grounded_in_research(oracle_run: _Run) -> None:
    profile = _distinctive(_profile_text())
    research_tokens = _distinctive(oracle_run.research) - profile
    answer_tokens = _distinctive(" ".join(oracle_run.answers)) - profile

    shared = research_tokens & answer_tokens
    score = jaccard(research_tokens, answer_tokens)
    print(f"grounding: jaccard {score:.3f}, shared distinctive tokens {sorted(shared)}")
    assert len(shared) >= 3 or score >= 0.03, (
        f"answers share too little with the research (jaccard {score:.3f}, shared {sorted(shared)})"
    )


def test_answers_do_not_invent_citations(oracle_run: _Run) -> None:
    research_urls = _urls(oracle_run.research)
    for index, answer in enumerate(oracle_run.answers, start=1):
        invented = sorted(_urls(answer) - research_urls)
        assert not invented, f"answer {index} cites URLs that are not in the research: {invented}"


def test_answers_stay_in_character(oracle_run: _Run) -> None:
    for index, answer in enumerate(oracle_run.answers, start=1):
        normalized = _normalize(answer)
        for disclaimer in AI_DISCLAIMERS:
            assert disclaimer not in normalized, f"answer {index} breaks character with {disclaimer!r}"
        match = BREAKS_CHARACTER.search(answer)
        assert match is None, f"answer {index} refers to being an AI or a simulation: {match.group(0)!r} in {answer!r}"
