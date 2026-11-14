"""Live calibration of ConclusionsStage: bad evidence must not come back sugar-coated.

Four synthetic scenarios run the real conclusions model, once each, through module-scoped fixtures:
doom, euphoria, split and void. The assertions check structure exactly where the evidence is unambiguous and
loosely where the model's wording may vary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

from agent_backend.stages.conclusions import ConclusionsStage
from api_common.assets.assumptions import Assumption, AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.conclusions import AssumptionConclusion, ConclusionsOutput
from api_common.assets.validations import AssumptionVerdict, InterviewValidation, ValidationsOutput, Verdict

pytestmark = [pytest.mark.live]

_IDS = ["A1", "A2", "A3", "A4"]

_ASSUMPTIONS = AssumptionsOutput(
    assumptions=[
        Assumption(
            id="A1",
            statement="Dual-income parents find weeknight dinner planning stressful enough to pay to reduce it.",
            category="problem",
            rationale="Without a felt problem, nobody pays for the kit.",
        ),
        Assumption(
            id="A2",
            statement="Parents of children aged 4 to 12 in suburban areas plan the weekly meals themselves.",
            category="customer",
            rationale="The buyer must be the person who decides the menu.",
        ),
        Assumption(
            id="A3",
            statement="A pre-portioned kit with a recipe under 20 minutes is preferred to planning and shopping alone.",
            category="solution",
            rationale="The product only beats cooking from scratch if the time saved is real.",
        ),
        Assumption(
            id="A4",
            statement="Households will pay about 9 EUR per portion for the kit.",
            category="willingness to pay",
            rationale="At this price the margin funds the local sourcing.",
        ),
    ]
)

_EVIDENCE = {
    "validated": "Described a routine that matches the assumption.",
    "invalidated": "Described a routine that contradicts the assumption.",
    "inconclusive": "Did not address the assumption directly.",
}

_CAVEAT_MARKERS = ("simulated", "real customers", "real-world", "real evidence", "test with real")
_RECOMMENDATION_VERBS = re.compile(
    r"\b(test|interview|measure|pilot|run|validate|talk|build|drop|change|keep)\w*\b", re.IGNORECASE
)
_ASSUMPTION_REFERENCE = re.compile(r"\bA[1-4]\b")
_REAL_CUSTOMER_REFERENCE = re.compile(
    r"\breal[- ](?:world|customers?|parents?|households?|families|people|persons?|users?|buyers?|shoppers?"
    r"|clients?|subscribers?|respondents?|interviews?|money|payments?|sales)\b",
    re.IGNORECASE,
)
_SCALE_OR_BUILD = re.compile(r"\b(scale|scaling|build|launch|expand|hire|invest|roll out)\w*\b", re.IGNORECASE)
_TESTING_WORD = re.compile(
    r"\b(test|pilot|interview|measure|validate|verify|experiment|survey|investigat|research|explor|check|ask|learn"
    r"|find out)\w*\b",
    re.IGNORECASE,
)
_IDEA_VALIDATED_CLAIM = re.compile(
    r"\b(idea|business|concept|product|plan)\b[^.]{0,40}\b(is|has been|was|are)\s+(validated|proven|confirmed)"
    r"|\b(validated|proven|confirmed)\s+(the|this|your)\s+(idea|business|concept)\b",
    re.IGNORECASE,
)
_INSUFFICIENT_MARKERS = (
    "insufficient",
    "not enough",
    "not sufficient",
    "no evidence",
    "not yet",
    "real customer",
    "real-world",
    "real world",
    "more evidence",
)


def _interview(persona_id: str, verdicts: Mapping[str, Verdict]) -> InterviewValidation:
    return InterviewValidation(
        persona_id=persona_id,
        verdicts=[
            AssumptionVerdict(assumption_id=assumption_id, verdict=verdict, evidence=_EVIDENCE[verdict])
            for assumption_id, verdict in verdicts.items()
        ],
    )


def _state(p1: Mapping[str, Verdict], p2: Mapping[str, Verdict]) -> PipelineState:
    return PipelineState(
        request=PipelineRequest(
            business_idea=(
                "Weekly meal kits that pre-portion locally sourced ingredients and a recipe that can be cooked "
                "in 20 minutes, so busy households stop improvising dinner on school nights."
            ),
            target_audience=(
                "Dual-income parents in suburban areas with children aged 4 to 12, who cook dinner at home "
                "at least three nights a week."
            ),
        ),
        assumptions=_ASSUMPTIONS,
        validations=ValidationsOutput(validations=[_interview("P1", p1), _interview("P2", p2)]),
    )


def _run(state: PipelineState) -> ConclusionsOutput:
    conclusions = ConclusionsStage().run(state).conclusions
    assert conclusions is not None, "ConclusionsStage did not set conclusions"
    return conclusions


def _by_id(output: ConclusionsOutput) -> dict[str, AssumptionConclusion]:
    return {item.assumption_id: item for item in output.assumptions}


def _verdicts(output: ConclusionsOutput) -> dict[str, Verdict]:
    return {item.assumption_id: item.verdict for item in output.assumptions}


@pytest.fixture(scope="module")
def doom_output() -> ConclusionsOutput:
    both = {"A1": "invalidated", "A2": "invalidated", "A3": "validated", "A4": "inconclusive"}
    return _run(_state(both, both))


@pytest.fixture(scope="module")
def euphoria_output() -> ConclusionsOutput:
    all_validated = {"A1": "validated", "A2": "validated", "A3": "validated", "A4": "validated"}
    return _run(_state(all_validated, all_validated))


@pytest.fixture(scope="module")
def split_output() -> ConclusionsOutput:
    p1 = {"A1": "validated", "A2": "validated", "A3": "invalidated", "A4": "inconclusive"}
    p2 = {"A1": "invalidated", "A2": "validated", "A3": "invalidated", "A4": "inconclusive"}
    return _run(_state(p1, p2))


@pytest.fixture(scope="module")
def void_output() -> ConclusionsOutput:
    none = {"A1": "inconclusive", "A2": "inconclusive", "A3": "inconclusive", "A4": "inconclusive"}
    return _run(_state(none, none))


def test_every_scenario_has_one_conclusion_per_assumption(
    doom_output: ConclusionsOutput,
    euphoria_output: ConclusionsOutput,
    split_output: ConclusionsOutput,
    void_output: ConclusionsOutput,
) -> None:
    for output in (doom_output, euphoria_output, split_output, void_output):
        ids = [item.assumption_id for item in output.assumptions]
        assert sorted(ids) == _IDS, f"expected exactly {_IDS} once each, got {ids}"
        assert all(item.reasoning.strip() for item in output.assumptions), "every conclusion needs reasoning"
        assert output.summary.strip(), "summary must not be empty"


def test_doom_rejects_failing_assumptions_and_does_not_claim_validation(doom_output: ConclusionsOutput) -> None:
    verdicts = _verdicts(doom_output)
    assert verdicts["A1"] == "invalidated"
    assert verdicts["A2"] == "invalidated"
    assert verdicts["A3"] == "validated"
    assert verdicts["A4"] == "inconclusive"
    assert not _IDEA_VALIDATED_CLAIM.search(doom_output.summary), (
        f"summary claims the idea is validated despite two invalidated assumptions: {doom_output.summary!r}"
    )


def test_euphoria_validates_everything_but_keeps_simulation_caveat(euphoria_output: ConclusionsOutput) -> None:
    assert set(_verdicts(euphoria_output).values()) == {"validated"}
    summary = euphoria_output.summary.lower()
    assert any(marker in summary for marker in _CAVEAT_MARKERS), (
        f"summary of simulated-only evidence has no caveat: {euphoria_output.summary!r}"
    )


def test_split_verdicts_follow_the_evidence(split_output: ConclusionsOutput) -> None:
    verdicts = _verdicts(split_output)
    assert verdicts["A2"] == "validated"
    assert verdicts["A3"] == "invalidated"
    assert verdicts["A4"] == "inconclusive"

    a1 = _by_id(split_output)["A1"]
    if a1.verdict == "validated":
        caveat_words = (
            "split",
            "disagree",
            "one persona",
            "one interview",
            "mixed",
            "conflict",
            "contradict",
            "however",
            "simulated",
        )
        assert any(word in a1.reasoning.lower() for word in caveat_words), (
            f"A1 is validated with a 2-2 split and no caveat: {a1.reasoning!r}"
        )
    else:
        assert a1.verdict == "inconclusive", f"A1 should be inconclusive on a 2-2 split, got {a1.verdict}"


def test_void_scenario_is_inconclusive_and_asks_for_real_evidence(void_output: ConclusionsOutput) -> None:
    assert set(_verdicts(void_output).values()) == {"inconclusive"}
    text = (void_output.summary + " " + " ".join(void_output.recommendations)).lower()
    assert any(marker in text for marker in _INSUFFICIENT_MARKERS), (
        "void scenario should say the evidence is insufficient or needs real customer tests"
    )


def test_recommendations_are_concrete_and_grounded(
    doom_output: ConclusionsOutput,
    euphoria_output: ConclusionsOutput,
    split_output: ConclusionsOutput,
    void_output: ConclusionsOutput,
) -> None:
    for output in (doom_output, euphoria_output, split_output, void_output):
        recommendations = output.recommendations
        assert len(recommendations) >= 3, f"expected at least 3 recommendations, got {len(recommendations)}"
        assert all(len(text.strip()) >= 15 for text in recommendations), "each recommendation needs substance"
        assert any(_RECOMMENDATION_VERBS.search(text) for text in recommendations), (
            f"no recommendation has a concrete action verb: {recommendations}"
        )
        assert any(
            _ASSUMPTION_REFERENCE.search(text) or _REAL_CUSTOMER_REFERENCE.search(text) for text in recommendations
        ), f"no recommendation refers to an assumption id or real customers: {recommendations}"


def test_doom_and_void_never_recommend_scaling_before_testing(
    doom_output: ConclusionsOutput, void_output: ConclusionsOutput
) -> None:
    for scenario, output in (("doom", doom_output), ("void", void_output)):
        for text in output.recommendations:
            if _SCALE_OR_BUILD.search(text):
                assert _TESTING_WORD.search(text), f"{scenario}: recommends scaling or building before testing: {text!r}"
