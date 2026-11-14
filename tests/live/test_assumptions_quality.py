"""Live, adversarial checks on what AssumptionsStage writes for two very different ideas.

The module-scoped fixture calls the real model exactly twice: once for a serious B2B scheduling copilot and once for
a deliberately absurd umbrella rental service for fish. Every test reads those two outputs. Nothing here may call
the model again, so a test that needs new data must go through the fixture.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from tests.helpers import assert_sequential_ids, assert_unique_ids, jaccard, require_live, token_set

pytestmark = [pytest.mark.live]

SERIOUS = "serious"
ABSURD = "absurd"

_QUESTION_OPENERS = {
    "is", "are", "am", "was", "were", "do", "does", "did", "will", "would", "can", "could", "should",
    "how", "what", "why", "who", "when", "where", "which",
}

_CATEGORY_BUCKETS: Mapping[str, tuple[str, ...]] = {
    "problem": ("problem", "pain", "need", "struggle"),
    "customer": ("customer", "segment", "audience", "buyer", "user", "persona"),
    "solution": ("solution", "product", "feature", "offer", "service"),
    "pricing": ("pric", "pay", "willing", "revenue", "monetis", "monetiz", "margin"),
    "channel": ("channel", "distribution", "acquisition", "marketing", "sales", "referral"),
}

_HOROSCOPE_PHRASES = (
    "revolutionary",
    "game-changing",
    "game changing",
    "everyone will love",
    "huge market",
    "no competition",
    "synergy",
    "paradigm shift",
)

_CONCRETE_CLAIM = re.compile(
    r"\d"
    r"|[€$£%]"
    r"|\b(eur|euros?|usd|dollars?|price|priced|per|weekly|monthly|daily|hours?|minutes?|days?|percent"
    r"|than|versus|instead|compared|more|less|fewer|times|orders?|book\w*|cancel\w*|sign\w*|pay\w*|buy\w*"
    r"|bought|churn\w*|renew\w*|repl(y|ies)|respond\w*|switch\w*|dentists?|practices?|clinics?|parents?"
    r"|households?|owners?|managers?|students?|nurses?|shops?|families|customers?|subscribers?|patients?"
    r"|fish|groups?|teams?|companies|employees?|visits?|appointments?|calls?|emails?)\b",
    re.IGNORECASE,
)

_MODAL = re.compile(r"\b(will|must|need|needs|pay|pays|would|should)\b", re.IGNORECASE)
_PROOF_CLAIM = re.compile(
    r"\b(proven|proves|already validated|demand is confirmed|market is confirmed|guaranteed|certainly will)\b",
    re.IGNORECASE,
)

_SERIOUS_REQUEST = PipelineRequest(
    business_idea=(
        "A scheduling copilot for independent dental practices in Germany. It fills last-minute cancellations "
        "from a waitlist, sends reminders through SMS and WhatsApp, and integrates with the practice management "
        "software the front desk already uses."
    ),
    target_audience="Owners and office managers of independent dental practices with one to four chairs in Germany.",
    additional_info="Subscription priced per practice per month. Pilot in one German city first.",
    persona_count=3,
)

_ABSURD_REQUEST = PipelineRequest(
    business_idea=(
        "A premium subscription that rents tiny umbrellas to fish, delivered monthly to the shallows, "
        "with a loyalty tier for fish that stay in one reef."
    ),
    target_audience="fish",
    persona_count=3,
)


@pytest.fixture(scope="module")
def assumption_runs() -> dict[str, AssumptionsOutput]:
    """The only live calls in this module: one serious idea and one absurd idea, each run once."""
    require_live()
    requests = {SERIOUS: _SERIOUS_REQUEST, ABSURD: _ABSURD_REQUEST}
    runs: dict[str, AssumptionsOutput] = {}
    for key, request in requests.items():
        state = AssumptionsStage().run(PipelineState(request=request))
        assert state.assumptions is not None, f"{key} run produced no assumptions"
        runs[key] = state.assumptions
    return runs


def _normalise_category(raw: str) -> str:
    return re.sub(r"[_\-\s]+", " ", raw.lower()).strip()


def _buckets_hit(output: AssumptionsOutput) -> set[str]:
    hit: set[str] = set()
    for assumption in output.assumptions:
        norm = _normalise_category(assumption.category)
        for bucket, keys in _CATEGORY_BUCKETS.items():
            if any(key in norm for key in keys):
                hit.add(bucket)
    return hit


def _is_declarative(statement: str) -> bool:
    text = statement.strip()
    if "?" in text or text.endswith("?"):
        return False
    words = re.findall(r"[a-zA-Z']+", text)
    return bool(words) and words[0].lower() not in _QUESTION_OPENERS


@pytest.mark.parametrize("key", [SERIOUS, ABSURD])
def test_the_list_is_short_numbered_and_never_repeats_an_id(assumption_runs: dict[str, AssumptionsOutput], key: str):
    """Six to eight riskiest assumptions, numbered A1..An in order with no gaps or duplicates."""
    assumptions = assumption_runs[key].assumptions

    assert 6 <= len(assumptions) <= 8, f"{key}: expected 6 to 8 assumptions, got {len(assumptions)}"
    assert_unique_ids(assumptions)
    assert_sequential_ids(assumptions, "A")


def test_the_five_lean_categories_show_up_for_both_ideas_in_some_form(assumption_runs: dict[str, AssumptionsOutput]):
    """Problem, customer, solution, pricing and channel should each have a home in at least four of five slots."""
    for key in (SERIOUS, ABSURD):
        hit = _buckets_hit(assumption_runs[key])
        assert len(hit) >= 4, f"{key}: only covered {sorted(hit)} of the five lean categories"


@pytest.mark.parametrize("key", [SERIOUS, ABSURD])
def test_each_statement_reads_as_a_sentence_and_each_rationale_says_why(
    assumption_runs: dict[str, AssumptionsOutput], key: str
):
    """Declarative claims only, at least four words each, and a rationale long enough to explain the stake."""
    assumptions = assumption_runs[key].assumptions
    for assumption in assumptions:
        assert _is_declarative(assumption.statement), f"{key} {assumption.id} is not declarative: {assumption.statement!r}"
        assert len(assumption.statement.split()) >= 4, f"{key} {assumption.id} statement is too short"
        assert len(assumption.rationale.strip()) >= 15, f"{key} {assumption.id} rationale is too thin"

    for i, first in enumerate(assumptions):
        for second in assumptions[i + 1 :]:
            similarity = jaccard(token_set(first.statement), token_set(second.statement))
            assert similarity < 0.8, f"{key}: {first.id} and {second.id} are near-duplicates ({similarity:.2f})"


@pytest.mark.parametrize("key", [SERIOUS, ABSURD])
def test_no_statement_is_horoscope_language_and_most_name_something_testable(
    assumption_runs: dict[str, AssumptionsOutput], key: str
):
    """Vague praise with nothing measurable is rejected. At least 80% of statements name a subject and a testable hook."""
    assumptions = assumption_runs[key].assumptions

    mushy = [
        a.id
        for a in assumptions
        if any(phrase in a.statement.lower() for phrase in _HOROSCOPE_PHRASES) and not _CONCRETE_CLAIM.search(a.statement)
    ]
    assert not mushy, f"{key}: statements are only vague praise: {mushy}"

    testable = [a for a in assumptions if len(token_set(a.statement)) >= 4 and _CONCRETE_CLAIM.search(a.statement)]
    ratio = len(testable) / len(assumptions)
    assert ratio >= 0.8, f"{key}: only {ratio:.0%} of statements are testable; weak ones: {[a.id for a in assumptions if a not in testable]}"


def test_the_dental_and_umbrella_lists_describe_different_businesses(assumption_runs: dict[str, AssumptionsOutput]):
    """Token overlap between the two outputs as a whole stays low, so the model is not recycling one template."""
    serious_tokens = frozenset().union(*(token_set(a.statement) for a in assumption_runs[SERIOUS].assumptions))
    absurd_tokens = frozenset().union(*(token_set(a.statement) for a in assumption_runs[ABSURD].assumptions))

    similarity = jaccard(serious_tokens, absurd_tokens)
    assert similarity < 0.5, f"the two assumption sets overlap too much (Jaccard {similarity:.2f})"


def test_the_fish_umbrella_idea_gets_hypotheses_not_a_verdict(assumption_runs: dict[str, AssumptionsOutput]):
    """Even an absurd idea should come back as things to test: a customer, a price, and no claim that the market is proven."""
    output = assumption_runs[ABSURD]
    assert AssumptionsOutput.model_validate(output.model_dump()) == output

    hit = _buckets_hit(output)
    assert "customer" in hit, f"no customer assumption for the fish idea; categories: {sorted(hit)}"
    assert "pricing" in hit, f"no willingness-to-pay assumption for the fish idea; categories: {sorted(hit)}"

    statements = [a.statement for a in output.assumptions]
    assert any(_MODAL.search(s) for s in statements), "no statement frames a need, a payment or a requirement"

    for a in output.assumptions:
        assert not _PROOF_CLAIM.search(a.statement), f"{a.id} claims the market is proven: {a.statement!r}"
        assert not _PROOF_CLAIM.search(a.rationale), f"{a.id} rationale claims proof: {a.rationale!r}"
