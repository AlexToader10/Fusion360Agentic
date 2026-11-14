"""Live "Groundhog Day" checks: does the model give stable answers to the same idea, and does it react to the input?

Two module-scoped fixtures make exactly four live AssumptionsStage calls, each run once and shared by every test:

- a polished paragraph and a casual, shorter version of the same business idea (paraphrase invariance);
- one product idea, "a scheduling and waitlist copilot", aimed at independent dental practices and at tattoo studios
  (counterfactual sensitivity).

The "no clone army" scenario, repeated PersonasStage runs on one broad audience, is not included: it would need a
fifth live call on top of this budget. Its structural half (unique persona names and occupations) belongs in a
single-run test that uses one PersonasStage call.

Results are stochastic. The thresholds are deliberately generous. They catch a model that ignores its input,
not normal wording drift.
"""

from __future__ import annotations

import re

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from tests.helpers import assert_sequential_ids, assert_unique_ids, jaccard, require_live, token_set

pytestmark = [pytest.mark.live, pytest.mark.slow]

POLISHED = "polished"
CASUAL = "casual"
DENTAL = "dental"
TATTOO = "tattoo"

_PARAPHRASE_AUDIENCE = "Office managers at independent dental practices in Germany with one to four chairs."

_POLISHED_IDEA = (
    "A scheduling and waitlist copilot for independent dental practices in Germany. When a patient cancels at short "
    "notice, the copilot offers the freed slot to patients on a waitlist by SMS, and it writes each confirmed booking "
    "back into the calendar software the practice already uses. It is sold as a monthly subscription per practice."
)

_CASUAL_IDEA = (
    "scheduling + waitlist copilot for small dental offices in Germany. when someone cancels it texts people on the "
    "waitlist to grab the slot, and it syncs with the calendar they already use. monthly subscription per office, "
    "sold to the office manager."
)

_COUNTERFACTUAL_IDEA = (
    "A scheduling and waitlist copilot. When a client cancels at short notice, it offers the freed slot to people on "
    "a waitlist by SMS, and it syncs each confirmed booking with the calendar the business already uses. It is sold "
    "as a monthly subscription per business in Germany."
)

_DENTAL_AUDIENCE = "Owners and front-desk staff of independent dental practices in Germany."
_TATTOO_AUDIENCE = "Owners and front-desk staff of independent tattoo studios in Germany."

_DENTAL_TERM = re.compile(r"\b(dental|dentists?|practices?)\b", re.IGNORECASE)
_TATTOO_TERM = re.compile(r"\b(tattoo\w*|artists?|studios?)\b", re.IGNORECASE)

_CORE_BUCKETS: dict[str, tuple[str, ...]] = {
    "problem": ("problem", "pain", "need", "struggle"),
    "customer": ("customer", "segment", "audience", "buyer", "user", "persona", "patient", "client"),
    "solution": ("solution", "product", "feature", "offer", "service", "copilot", "software"),
    "pricing": ("pric", "pay", "willing", "revenue", "monetis", "monetiz", "subscription", "margin", "eur", "cost"),
    "channel": ("channel", "distribution", "acquisition", "marketing", "sales", "referral", "sms", "signup", "sign up"),
}


def _request(idea: str, audience: str) -> PipelineRequest:
    return PipelineRequest(business_idea=idea, target_audience=audience)


@pytest.fixture(scope="module")
def paraphrase_runs() -> dict[str, AssumptionsOutput]:
    """Two live calls: the same business described polished and casual."""
    require_live()
    requests = {
        POLISHED: _request(_POLISHED_IDEA, _PARAPHRASE_AUDIENCE),
        CASUAL: _request(_CASUAL_IDEA, _PARAPHRASE_AUDIENCE),
    }
    return {key: _run_assumptions(key, request) for key, request in requests.items()}


@pytest.fixture(scope="module")
def audience_runs() -> dict[str, AssumptionsOutput]:
    """Two live calls: one idea, two very different audiences."""
    require_live()
    requests = {
        DENTAL: _request(_COUNTERFACTUAL_IDEA, _DENTAL_AUDIENCE),
        TATTOO: _request(_COUNTERFACTUAL_IDEA, _TATTOO_AUDIENCE),
    }
    return {key: _run_assumptions(key, request) for key, request in requests.items()}


def _run_assumptions(key: str, request: PipelineRequest) -> AssumptionsOutput:
    state = AssumptionsStage().run(PipelineState(request=request))
    assert state.assumptions is not None, f"{key} run produced no assumptions"
    return state.assumptions


def _all_statement_tokens(output: AssumptionsOutput) -> frozenset[str]:
    return frozenset().union(*(token_set(a.statement) for a in output.assumptions))


def _normalised_statement(statement: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", statement.lower()).strip()


def _covered_buckets(output: AssumptionsOutput) -> set[str]:
    covered: set[str] = set()
    for assumption in output.assumptions:
        text = f"{assumption.category} {assumption.statement}".lower()
        for bucket, keys in _CORE_BUCKETS.items():
            if any(key in text for key in keys):
                covered.add(bucket)
    return covered


@pytest.mark.parametrize("key", [POLISHED, CASUAL])
def test_each_paraphrase_is_schema_valid_and_has_six_to_eight_numbered_assumptions(
    paraphrase_runs: dict[str, AssumptionsOutput], key: str
):
    output = paraphrase_runs[key]

    assert AssumptionsOutput.model_validate(output.model_dump()) == output, f"{key}: output does not round-trip"
    assert 6 <= len(output.assumptions) <= 8, f"{key}: expected 6 to 8 assumptions, got {len(output.assumptions)}"
    assert_unique_ids(output.assumptions)
    assert_sequential_ids(output.assumptions, "A")
    for assumption in output.assumptions:
        assert assumption.statement.strip(), f"{key} {assumption.id} has an empty statement"
        assert assumption.rationale.strip(), f"{key} {assumption.id} has an empty rationale"


def test_the_polished_and_casual_paraphrases_overlap_enough_to_describe_the_same_business(
    paraphrase_runs: dict[str, AssumptionsOutput],
):
    """Same facts in two registers should give the same business. Token Jaccard over all statements stays at 0.20 or more."""
    similarity = jaccard(_all_statement_tokens(paraphrase_runs[POLISHED]), _all_statement_tokens(paraphrase_runs[CASUAL]))

    assert similarity >= 0.20, (
        f"paraphrase assumption sets are too far apart (token Jaccard {similarity:.2f}, need >= 0.20)"
    )


@pytest.mark.parametrize("key", [POLISHED, CASUAL])
def test_each_paraphrase_covers_at_least_four_of_the_five_lean_categories(
    paraphrase_runs: dict[str, AssumptionsOutput], key: str
):
    """Problem, customer, solution, pricing or willingness to pay, and channel: at least four of five must appear."""
    covered = _covered_buckets(paraphrase_runs[key])

    assert len(covered) >= 4, f"{key}: only covered {sorted(covered)} of problem, customer, solution, pricing, channel"


def test_the_two_audiences_produce_materially_different_assumptions_that_each_name_their_own_audience(
    audience_runs: dict[str, AssumptionsOutput],
):
    """Dental and tattoo versions of one idea must not be clones, and each must mention its own audience."""
    similarity = jaccard(_all_statement_tokens(audience_runs[DENTAL]), _all_statement_tokens(audience_runs[TATTOO]))
    assert similarity < 0.75, (
        f"the dental and tattoo assumption sets look like clones (token Jaccard {similarity:.2f}, need < 0.75)"
    )

    for key, term in ((DENTAL, _DENTAL_TERM), (TATTOO, _TATTOO_TERM)):
        statements = " ".join(a.statement for a in audience_runs[key].assumptions)
        assert term.search(statements), f"{key} assumptions never mention their audience: {statements!r}"


def test_the_model_does_not_reuse_the_same_statement_text_across_the_counterfactual_pair(
    audience_runs: dict[str, AssumptionsOutput],
):
    """Normalised statement text (case and punctuation removed) must mostly differ between the two audiences."""
    dental = {_normalised_statement(a.statement) for a in audience_runs[DENTAL].assumptions}
    tattoo = {_normalised_statement(a.statement) for a in audience_runs[TATTOO].assumptions}

    overlap = jaccard(dental, tattoo)
    assert overlap < 0.5, (
        f"the two audiences share too many statement texts (normalised statement Jaccard {overlap:.2f}, need < 0.5)"
    )
