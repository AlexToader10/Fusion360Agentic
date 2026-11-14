"""Live "twin universes" check: the same product must yield audience-specific output for two different audiences.

The product, a scheduling and waitlist copilot that fills last-minute cancellations, is held constant. Only the
audience changes: independent dental practices in Germany, or independent tattoo studios in Germany. The module-scoped
fixtures call the real model exactly four times: AssumptionsStage once per audience, then PersonasStage once per
audience with that audience's assumptions. Nothing here may call the model again.
"""

from __future__ import annotations

import re

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.personas import PersonasStage
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.personas import Persona, PersonasOutput
from tests.helpers import assert_sequential_ids, assert_unique_ids, jaccard, require_live, token_set

pytestmark = [pytest.mark.live, pytest.mark.slow]

PRODUCT = (
    "A scheduling and waitlist copilot that fills last-minute cancellations. When a booked appointment is cancelled, "
    "it messages the waitlist, offers the freed slot to the first person who accepts, and updates the calendar."
)
DENTAL = "dental"
TATTOO = "tattoo"
AUDIENCES = {
    DENTAL: (
        "Independent dental practices in Germany, each with one to five treatment chairs and a front desk "
        "that takes bookings by phone and email."
    ),
    TATTOO: (
        "Independent tattoo studios in Germany, each with a few tattoo artists who take booked and walk-in "
        "sessions and lose slots to late cancellations."
    ),
}
# Lowercase word-start prefixes. Matching by prefix catches German and English forms (Zahnarzt, dentist, Tätowierer).
_TERMS: dict[str, tuple[str, ...]] = {
    DENTAL: (
        "dent", "zahn", "praxis", "practice", "clinic", "klinik", "hygien", "orthodont", "surg", "receptionist", "empfang",
    ),
    TATTOO: ("tattoo", "tätowier", "tatowier", "artist", "künstler", "studio", "piercer", "piercing"),
}
_WORD = re.compile(r"[a-zäöüß0-9]+")


def _request(audience: str, persona_count: int | None = None) -> PipelineRequest:
    return PipelineRequest(business_idea=PRODUCT, target_audience=AUDIENCES[audience], persona_count=persona_count)


def _term_hits(text: str, audience: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word.startswith(_TERMS[audience])]


def _statements(output: AssumptionsOutput) -> str:
    return " ".join(item.statement for item in output.assumptions)


def _persona_text(persona: Persona) -> str:
    return " ".join([persona.name, persona.occupation, persona.background, persona.goals, persona.pain_points])


def _identity_text(persona: Persona) -> str:
    return " ".join([persona.name, persona.occupation, persona.background])


def _run_assumptions(audience: str) -> AssumptionsOutput:
    state = AssumptionsStage().run(PipelineState(request=_request(audience)))
    assert state.assumptions is not None, f"AssumptionsStage returned no assumptions for the {audience} audience"
    return state.assumptions


def _run_personas(audience: str, assumptions: AssumptionsOutput) -> PersonasOutput:
    state = PersonasStage().run(
        PipelineState(request=_request(audience, persona_count=3), assumptions=assumptions)
    )
    assert state.personas is not None, f"PersonasStage returned no personas for the {audience} audience"
    return state.personas


def _assert_persona_shape(audience: str, output: PersonasOutput) -> None:
    personas = output.personas
    assert len(personas) == 3, f"{audience}: asked for 3 personas, got {len(personas)}"
    assert_sequential_ids(personas, "P")
    assert_unique_ids(personas)
    assert_unique_ids(personas, attr="name")
    for persona in personas:
        assert 16 <= persona.age <= 90, f"{audience} {persona.id}: implausible age {persona.age}"
        for field in ("name", "occupation", "background", "goals", "pain_points"):
            assert getattr(persona, field).strip(), f"{audience} {persona.id}: empty {field}"


@pytest.fixture(scope="module")
def dental_assumptions() -> AssumptionsOutput:
    require_live()
    return _run_assumptions(DENTAL)


@pytest.fixture(scope="module")
def tattoo_assumptions() -> AssumptionsOutput:
    require_live()
    return _run_assumptions(TATTOO)


@pytest.fixture(scope="module")
def dental_personas(dental_assumptions: AssumptionsOutput) -> PersonasOutput:
    require_live()
    return _run_personas(DENTAL, dental_assumptions)


@pytest.fixture(scope="module")
def tattoo_personas(tattoo_assumptions: AssumptionsOutput) -> PersonasOutput:
    require_live()
    return _run_personas(TATTOO, tattoo_assumptions)


def test_each_audience_gets_six_to_eight_valid_assumptions(
    dental_assumptions: AssumptionsOutput, tattoo_assumptions: AssumptionsOutput
) -> None:
    for audience, output in ((DENTAL, dental_assumptions), (TATTOO, tattoo_assumptions)):
        items = output.assumptions
        assert 6 <= len(items) <= 8, f"{audience}: expected 6 to 8 assumptions, got {len(items)}"
        assert_sequential_ids(items, "A")
        assert_unique_ids(items)
        for item in items:
            assert item.statement.strip(), f"{audience} {item.id}: empty statement"
            assert item.rationale.strip(), f"{audience} {item.id}: empty rationale"
            assert item.category.strip(), f"{audience} {item.id}: empty category"
        assert AssumptionsOutput.model_validate(output.model_dump()) == output, f"{audience}: schema round-trip differs"


def test_assumptions_differ_across_audiences(
    dental_assumptions: AssumptionsOutput, tattoo_assumptions: AssumptionsOutput
) -> None:
    similarity = jaccard(token_set(_statements(dental_assumptions)), token_set(_statements(tattoo_assumptions)))
    assert similarity < 0.75, f"dental and tattoo assumptions overlap too much (token Jaccard {similarity:.2f})"
    assert _term_hits(_statements(dental_assumptions), DENTAL), "dental assumptions name no dental-specific term"
    assert _term_hits(_statements(tattoo_assumptions), TATTOO), "tattoo assumptions name no tattoo-specific term"


def test_personas_are_audience_specific(dental_personas: PersonasOutput, tattoo_personas: PersonasOutput) -> None:
    _assert_persona_shape(DENTAL, dental_personas)
    _assert_persona_shape(TATTOO, tattoo_personas)

    for audience, output in ((DENTAL, dental_personas), (TATTOO, tattoo_personas)):
        matching = [p.occupation for p in output.personas if _term_hits(p.occupation, audience)]
        assert len(matching) >= 2, (
            f"{audience}: at least 2 of 3 occupations should match the segment, "
            f"got {[p.occupation for p in output.personas]}"
        )

    dental_names = {p.name.strip().lower() for p in dental_personas.personas}
    tattoo_names = {p.name.strip().lower() for p in tattoo_personas.personas}
    shared = dental_names & tattoo_names
    assert not shared, f"persona names repeat across audiences: {sorted(shared)}"


def test_persona_vocabulary_stays_in_its_own_universe(
    dental_personas: PersonasOutput, tattoo_personas: PersonasOutput
) -> None:
    # Tolerates at most one accidental cross-term per audience. Hits are printed so the drift is visible.
    for audience, other, output in (
        (DENTAL, TATTOO, dental_personas),
        (TATTOO, DENTAL, tattoo_personas),
    ):
        cross = [word for persona in output.personas for word in _term_hits(_persona_text(persona), other)]
        if cross:
            print(f"\n[counterfactual] accidental {other} term(s) in {audience} personas: {cross}")
        assert len(cross) <= 1, f"{audience} personas mention {other} terms: {cross}"


def test_persona_links_stay_inside_their_own_universe(
    dental_assumptions: AssumptionsOutput,
    tattoo_assumptions: AssumptionsOutput,
    dental_personas: PersonasOutput,
    tattoo_personas: PersonasOutput,
) -> None:
    # Both universes number their assumptions A1..An, so an id alone cannot reveal a cross-universe link.
    # The check therefore uses two signals: every link must exist in its own set, and the linked statements must
    # speak the audience's own vocabulary at least as often as the other audience's.
    for audience, other, assumptions, output in (
        (DENTAL, TATTOO, dental_assumptions, dental_personas),
        (TATTOO, DENTAL, tattoo_assumptions, tattoo_personas),
    ):
        by_id = {item.id: item for item in assumptions.assumptions}
        own_hits = other_hits = 0
        for persona in output.personas:
            assert persona.assumption_ids, f"{audience} {persona.id} links to no assumptions"
            assert len(set(persona.assumption_ids)) == len(persona.assumption_ids), (
                f"{audience} {persona.id} repeats an assumption id: {persona.assumption_ids}"
            )
            ghosts = [ident for ident in persona.assumption_ids if ident not in by_id]
            assert not ghosts, f"{audience} {persona.id} links to ids outside its own assumptions: {ghosts}"
            for ident in persona.assumption_ids:
                statement = by_id[ident].statement
                own_hits += len(_term_hits(statement, audience))
                other_hits += len(_term_hits(statement, other))
        assert own_hits >= other_hits, (
            f"{audience} persona links point at {other}-flavoured assumptions more than {audience}-flavoured ones "
            f"(own {own_hits}, other {other_hits})"
        )


def test_persona_profiles_are_not_copied_across_audiences(
    dental_personas: PersonasOutput, tattoo_personas: PersonasOutput
) -> None:
    worst = max(
        jaccard(token_set(_identity_text(dental)), token_set(_identity_text(tattoo)))
        for dental in dental_personas.personas
        for tattoo in tattoo_personas.personas
    )
    assert worst < 0.8, f"a dental and a tattoo persona share most of their name, occupation and background ({worst:.2f})"


def test_shared_assumptions_diagnostic(
    dental_assumptions: AssumptionsOutput, tattoo_assumptions: AssumptionsOutput
) -> None:
    # Informational only: a product-level assumption such as "customers pay for cancellations filled" may
    # legitimately appear in both universes. Nothing here fails the run.
    shared: list[str] = []
    for item in dental_assumptions.assumptions:
        best = max(
            jaccard(token_set(item.statement), token_set(other.statement)) for other in tattoo_assumptions.assumptions
        )
        if best >= 0.5:
            shared.append(item.id)
    print(
        f"\n[counterfactual] {len(shared)} of {len(dental_assumptions.assumptions)} dental assumptions have a "
        f"near-match (token Jaccard >= 0.5) in the tattoo set: {shared}"
    )
