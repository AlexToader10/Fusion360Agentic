"""Live checks of the persona ecology: ids, segment coverage, assumption links, diversity and no product leakage.

Budget: three live stage runs. AssumptionsStage once, PersonasStage once for a broad audience with 8 personas,
and PersonasStage once more with a single persona for the boundary. Assertions are thresholds, because the
model output is stochastic.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from itertools import combinations

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.personas import PersonasStage
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.personas import Persona, PersonasOutput
from tests.helpers import assert_sequential_ids, assert_unique_ids, jaccard, token_set

pytestmark = [pytest.mark.live]

PRODUCT_NAME = "ChairFill 3000"
AUDIENCE = (
    "Independent dental practice owners, office managers and dental hygienists in Germany, "
    "from solo practices to 10-staff clinics."
)
# Segment name -> keywords matched against the lowercased occupation and background of each persona.
SEGMENTS = {
    "practice owners": ("owner", "inhaber", "zahnarzt", "zahnärzt", "dentist", "praxisinhab"),
    "office managers": ("office manag", "praxismanag", "praxisverwalt", "front office", "verwaltung"),
    "dental hygienists": ("hygien", "prophylax"),
}
SKEPTIC_CUES = ("sceptic", "skeptic", "doubt", "concern", "burned", "budget", "not interested")
TEXT_FIELDS = ("name", "occupation", "background", "goals", "pain_points")


def _request(persona_count: int) -> PipelineRequest:
    return PipelineRequest(
        business_idea=(
            f"{PRODUCT_NAME} is a subscription software and SMS service that refills cancelled dental "
            "appointments from a waitlist within minutes."
        ),
        target_audience=AUDIENCE,
        persona_count=persona_count,
    )


def _profile(persona: Persona) -> str:
    return " ".join(getattr(persona, field) for field in TEXT_FIELDS)


def _segment_text(persona: Persona) -> str:
    return f"{persona.occupation} {persona.background}".lower()


def _attitude_text(persona: Persona) -> str:
    return f"{persona.background} {persona.goals} {persona.pain_points}".lower()


def _coverage(personas: list[Persona], assumptions: AssumptionsOutput) -> dict[str, int]:
    counts = Counter(aid for persona in personas for aid in set(persona.assumption_ids))
    return {assumption.id: counts[assumption.id] for assumption in assumptions.assumptions}


@pytest.fixture(scope="module")
def live_assumptions() -> AssumptionsOutput:
    state = AssumptionsStage().run(PipelineState(request=_request(8)))
    assert state.assumptions is not None, "AssumptionsStage returned no assumptions"
    return state.assumptions


@pytest.fixture(scope="module")
def broad_personas(live_assumptions: AssumptionsOutput) -> PersonasOutput:
    state = PersonasStage().run(PipelineState(request=_request(8), assumptions=live_assumptions))
    assert state.personas is not None, "PersonasStage returned no personas"
    return state.personas


@pytest.fixture(scope="module")
def single_persona(live_assumptions: AssumptionsOutput) -> PersonasOutput:
    state = PersonasStage().run(PipelineState(request=_request(1), assumptions=live_assumptions))
    assert state.personas is not None, "PersonasStage returned no personas for persona_count=1"
    return state.personas


def test_broad_audience_yields_eight_ordered_personas(broad_personas: PersonasOutput) -> None:
    personas = broad_personas.personas
    assert len(personas) == 8, f"asked for 8 personas, got {len(personas)}: {[p.id for p in personas]}"
    assert_sequential_ids(personas, "P")
    assert_unique_ids(personas)
    assert_unique_ids(personas, attr="name")


def test_profiles_are_plausible_and_link_only_to_real_assumptions(
    broad_personas: PersonasOutput, live_assumptions: AssumptionsOutput
) -> None:
    real_ids = {assumption.id for assumption in live_assumptions.assumptions}
    problems: list[str] = []
    for persona in broad_personas.personas:
        if not 18 <= persona.age <= 80:
            problems.append(f"{persona.id} has age {persona.age}, outside 18-80")
        for field in TEXT_FIELDS + ("assumption_ids",):
            value = getattr(persona, field)
            if not value or (isinstance(value, str) and not value.strip()):
                problems.append(f"{persona.id} has an empty {field}")
        ghosts = sorted(set(persona.assumption_ids) - real_ids)
        if ghosts:
            problems.append(f"{persona.id} links to unknown assumption ids {ghosts}, real ids are {sorted(real_ids)}")
    assert not problems, "implausible personas:\n  " + "\n  ".join(problems)


def test_assumptions_are_spoken_to_by_several_personas(
    broad_personas: PersonasOutput, live_assumptions: AssumptionsOutput
) -> None:
    coverage = _coverage(broad_personas.personas, live_assumptions)
    assert coverage, "the assumptions stage produced no assumptions to cover"
    uncovered = [aid for aid, count in coverage.items() if count == 0]
    assert not uncovered, f"assumptions no persona can speak to: {uncovered}. Coverage: {coverage}"
    covered_twice = sum(1 for count in coverage.values() if count >= 2)
    share = covered_twice / len(coverage)
    assert share >= 0.7, (
        f"only {covered_twice}/{len(coverage)} assumptions ({share:.0%}) have 2+ personas, need at least 70%. "
        f"Coverage: {coverage}"
    )
    median = statistics.median(coverage.values())
    assert median >= 2, f"median coverage is {median}, need at least 2. Coverage: {coverage}"


def test_sample_is_diverse_and_not_repetitive(broad_personas: PersonasOutput) -> None:
    personas = broad_personas.personas

    occupations = {persona.occupation.strip().lower() for persona in personas}
    assert len(occupations) >= 5, (
        f"only {len(occupations)} distinct occupations, need at least 5: {sorted(occupations)}"
    )

    ages = [persona.age for persona in personas]
    assert max(ages) - min(ages) >= 15, f"age range {min(ages)}-{max(ages)} is narrower than 15 years"

    near_duplicates = []
    for left, right in combinations(personas, 2):
        score = jaccard(token_set(_profile(left)), token_set(_profile(right)))
        if score >= 0.75:
            near_duplicates.append(f"{left.id}/{right.id} (Jaccard {score:.2f})")
    assert not near_duplicates, "near-duplicate personas: " + ", ".join(near_duplicates)

    # Soft diagnostic only: attitude varies across runs, so this never fails the test.
    skeptics = [p.id for p in personas if any(cue in _attitude_text(p) for cue in SKEPTIC_CUES)]
    print(f"soft diagnostic: skepticism cues in {len(skeptics)}/{len(personas)} personas: {skeptics}")


def test_every_named_audience_segment_is_represented(broad_personas: PersonasOutput) -> None:
    personas = broad_personas.personas
    missing = [
        segment
        for segment, keywords in SEGMENTS.items()
        if not any(keyword in _segment_text(persona) for persona in personas for keyword in keywords)
    ]
    assert not missing, (
        f"audience segments with no persona: {missing}. "
        f"Occupations in the sample: {[persona.occupation for persona in personas]}"
    )


def test_personas_do_not_name_or_pitch_the_product(broad_personas: PersonasOutput) -> None:
    leaks = []
    for persona in broad_personas.personas:
        for field in TEXT_FIELDS:
            text = re.sub(r"\s+", "", getattr(persona, field).lower())
            if "chairfill" in text:
                leaks.append(f"{persona.id}.{field}: {getattr(persona, field)!r}")
    assert not leaks, f"personas mention the product {PRODUCT_NAME!r}: " + "; ".join(leaks)


def test_single_persona_request_returns_one_linked_persona(
    single_persona: PersonasOutput, live_assumptions: AssumptionsOutput
) -> None:
    personas = single_persona.personas
    assert len(personas) == 1, f"asked for 1 persona, got {len(personas)}: {[p.id for p in personas]}"
    assert_sequential_ids(personas, "P")
    persona = personas[0]
    assert 18 <= persona.age <= 80, f"{persona.id} has age {persona.age}, outside 18-80"
    real_ids = {assumption.id for assumption in live_assumptions.assumptions}
    assert persona.assumption_ids, f"{persona.id} links to no assumptions"
    ghosts = sorted(set(persona.assumption_ids) - real_ids)
    assert not ghosts, f"{persona.id} links to unknown assumption ids {ghosts}"
