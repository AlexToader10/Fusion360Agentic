"""Live "Babel" checks: the pipeline keeps its schema and its input context when the brief is not English.

Three module-scoped fixtures make exactly three stage calls to the real models:

1. AssumptionsStage on a Spanish brief (mobile dog grooming in Madrid).
2. PersonasStage on a German brief with an emoji (zero-waste lunch subscription in Berlin), fed with synthetic
   assumptions so that the personas are not coupled to the Spanish run.
3. QuestionsStage with question_count=6 on the Spanish assumptions from call 1.

The tests only check structure and that the input context survives. They do not require a particular language:
the model may answer in English as long as the Spanish or German brief is still visible in what it writes.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Iterator

import pytest

from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.personas import PersonasStage
from agent_backend.stages.questions import QuestionsStage
from api_common.assets.assumptions import Assumption, AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.personas import Persona, PersonasOutput
from api_common.assets.questions import QuestionsOutput
from tests.helpers import assert_sequential_ids, assert_unique_ids

pytestmark = [pytest.mark.live]

SPANISH_CANARY = "MaxiPelo 7"
SPANISH_CANARY_KEY = "maxipelo"

SPANISH_REQUEST = PipelineRequest(
    business_idea=(
        f"{SPANISH_CANARY}: un servicio móvil de peluquería canina a domicilio. Una furgoneta lava, corta y "
        "cepilla a los perros en la puerta de casa."
    ),
    target_audience=(
        "Dueños de perros en Madrid que trabajan todo el día fuera de casa y no tienen tiempo para llevar a su "
        "mascota a una peluquería."
    ),
    additional_info="Piloto en dos barrios de Madrid. Precio objetivo: unos 45 euros por servicio.",
    question_count=6,
)

GERMAN_REQUEST = PipelineRequest(
    business_idea=(
        "Ein Mittagessen-Abo ohne Verpackungsmüll in Berlin 🥗: jede Woche frisch gekochte Mahlzeiten in "
        "Mehrwegboxen, die man zurückgibt und die wir waschen."
    ),
    target_audience=(
        "Büroangestellte in Berlin zwischen 25 und 55 Jahren, die mittags außer Haus essen und Verpackungsmüll "
        "vermeiden wollen."
    ),
    additional_info="Pilot zuerst in Berlin-Mitte. Zielpreis etwa 7 Euro pro Mittagessen. 🌱",
    persona_count=3,
)

GERMAN_ASSUMPTIONS = AssumptionsOutput(
    assumptions=[
        Assumption(
            id="A1",
            statement="Office workers in Berlin skip a healthy lunch because they have no time to plan or cook it.",
            category="problem",
            rationale="Without a felt lunch problem, nobody subscribes.",
        ),
        Assumption(
            id="A2",
            statement="Büroangestellte in Berlin choose and pay for their own lunch during the working week.",
            category="customer",
            rationale="The buyer must be the person who eats the lunch.",
        ),
        Assumption(
            id="A3",
            statement="Returnable reusable boxes are accepted as an easy alternative to disposable lunch packaging.",
            category="solution",
            rationale="The zero-waste promise only works if returning the boxes is effortless.",
        ),
        Assumption(
            id="A4",
            statement="Office workers in Berlin will pay about 7 EUR per lunch for a weekly subscription.",
            category="willingness to pay",
            rationale="At this price the subscription margin covers packaging returns.",
        ),
    ]
)

# A stage that receives a sensible prompt should not hand back these placeholder leaks.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}|\{\}|\bNone\b|\bnull\b|\bnan\b|\bNaN\b")
_BOILERPLATE = ("holds for the target", "business case depends on", "assumption a1", "assumption a2")

_SPANISH_SIGNAL = re.compile(
    r"[áéíóúñüÁÉÍÓÚÑÜ¿¡]|\b(para|clientes?|perros?|servicio|precio|due[ñn]os?|mascotas?|trabajan|semana)\b",
    re.IGNORECASE,
)
_SPANISH_AUDIENCE = re.compile(
    r"\b(perros?|due[ñn]os?|mascotas?|madrid|clientes?|dog|dogs|pets?|owners?|work|works|working)\b",
    re.IGNORECASE,
)
_OFFICE_BERLIN = re.compile(
    r"berlin|büro|buero|office|arbeit|angestellt|kolleg|employee|worker|mittag|lunch",
    re.IGNORECASE,
)

_CATEGORY_BUCKETS: dict[str, tuple[str, ...]] = {
    "problem": ("problem", "pain", "need", "struggle", "necesidad", "dolor"),
    "customer": ("customer", "segment", "audience", "buyer", "user", "cliente", "segmento", "dueño", "dueno"),
    "solution": ("solution", "solucion", "solución", "soluc", "product", "feature", "offer", "servicio", "producto"),
    "pricing": ("pric", "willing", "pay", "revenue", "margin", "precio", "pagar", "pagan", "dispuest"),
    "channel": ("channel", "distribution", "acquisition", "marketing", "sales", "referral", "canal", "venta"),
}


def _fold(text: str) -> str:
    """Lowercase and strip accents, so 'Solución' and 'solucion' compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _assumption_text(assumption: Assumption) -> str:
    return " ".join([assumption.statement, assumption.rationale, assumption.category])


def _persona_text(persona: Persona) -> str:
    return " ".join(
        [persona.name, persona.occupation, persona.background, persona.goals, persona.pain_points]
    )


def _category_buckets(assumptions: list[Assumption]) -> set[str]:
    hit: set[str] = set()
    for assumption in assumptions:
        category = _fold(assumption.category)
        for bucket, keywords in _CATEGORY_BUCKETS.items():
            if any(_fold(keyword) in category for keyword in keywords):
                hit.add(bucket)
    return hit


def _all_output_text(
    assumptions: AssumptionsOutput, personas: PersonasOutput, questions: QuestionsOutput
) -> Iterator[tuple[str, str]]:
    for assumption in assumptions.assumptions:
        for field in ("id", "category", "statement", "rationale"):
            yield f"{assumption.id}.{field}", getattr(assumption, field)
    for persona in personas.personas:
        for field in ("id", "name", "occupation", "background", "goals", "pain_points"):
            yield f"{persona.id}.{field}", getattr(persona, field)
    for question in questions.questions:
        yield f"{question.id}.text", question.text


@pytest.fixture(scope="module")
def spanish_state() -> PipelineState:
    """Stage call 1: AssumptionsStage on the Spanish brief."""
    return AssumptionsStage().run(PipelineState(request=SPANISH_REQUEST))


@pytest.fixture(scope="module")
def german_personas_state() -> PipelineState:
    """Stage call 2: PersonasStage on the German brief, with synthetic assumptions."""
    state = PipelineState(request=GERMAN_REQUEST, assumptions=GERMAN_ASSUMPTIONS)
    return PersonasStage().run(state)


@pytest.fixture(scope="module")
def spanish_questions_state(spanish_state: PipelineState) -> PipelineState:
    """Stage call 3: QuestionsStage with question_count=6 on the Spanish assumptions."""
    state = PipelineState(request=SPANISH_REQUEST, assumptions=spanish_state.assumptions)
    return QuestionsStage().run(state)


# Schema and size


def test_spanish_assumptions_are_schema_valid_and_sized(spanish_state: PipelineState) -> None:
    assumptions = spanish_state.assumptions.assumptions

    assert 6 <= len(assumptions) <= 8, f"expected 6 to 8 assumptions, got {len(assumptions)}"
    assert_unique_ids(assumptions)
    assert_sequential_ids(assumptions, "A")
    for assumption in assumptions:
        assert assumption.statement.strip(), f"{assumption.id} has an empty statement"
        assert assumption.rationale.strip(), f"{assumption.id} has an empty rationale"
        assert len(assumption.statement.split()) >= 4, f"{assumption.id} statement is too short: {assumption.statement!r}"
        lowered = assumption.statement.lower()
        assert not any(marker in lowered for marker in _BOILERPLATE), (
            f"{assumption.id} reads like English boilerplate: {assumption.statement!r}"
        )


def test_german_personas_are_schema_valid_and_sized(german_personas_state: PipelineState) -> None:
    personas = german_personas_state.personas.personas
    known_ids = {assumption.id for assumption in GERMAN_ASSUMPTIONS.assumptions}

    assert len(personas) == 3, f"expected exactly 3 personas, got {len(personas)}"
    assert_unique_ids(personas)
    assert_sequential_ids(personas, "P")
    for persona in personas:
        assert 18 <= persona.age <= 80, f"{persona.id} has an implausible age: {persona.age}"
        for field in ("name", "occupation", "background", "goals", "pain_points"):
            value = getattr(persona, field)
            assert value.strip(), f"{persona.id}.{field} is empty"
        assert persona.assumption_ids, f"{persona.id} links to no assumption"
        unknown = [ident for ident in persona.assumption_ids if ident not in known_ids]
        assert not unknown, f"{persona.id} links to unknown assumptions {unknown}; known: {sorted(known_ids)}"


def test_spanish_questions_are_schema_valid_and_cover_every_assumption(
    spanish_state: PipelineState, spanish_questions_state: PipelineState
) -> None:
    questions = spanish_questions_state.questions.questions
    known_ids = {assumption.id for assumption in spanish_state.assumptions.assumptions}

    assert len(questions) == 6, f"expected exactly 6 questions, got {len(questions)}"
    assert_unique_ids(questions)
    assert_sequential_ids(questions, "Q")
    covered: set[str] = set()
    for question in questions:
        assert question.text.strip(), f"{question.id} has an empty text"
        assert question.assumption_ids, f"{question.id} links to no assumption"
        unknown = [ident for ident in question.assumption_ids if ident not in known_ids]
        assert not unknown, f"{question.id} links to unknown assumptions {unknown}"
        covered.update(question.assumption_ids)

    missing = sorted(known_ids - covered)
    assert not missing, f"assumptions not covered by any question: {missing}"

    canary_hits = [q.id for q in questions if SPANISH_CANARY_KEY in _fold(q.text).replace(" ", "")]
    assert not canary_hits, f"questions pitch the product canary {SPANISH_CANARY!r}: {canary_hits}"


# Context survives the language


def test_spanish_assumptions_keep_spanish_context(spanish_state: PipelineState) -> None:
    assumptions = spanish_state.assumptions.assumptions

    language_hits = [a.id for a in assumptions if _SPANISH_SIGNAL.search(_assumption_text(a))]
    assert language_hits, (
        "no assumption carries a Spanish-language signal; the output may ignore the Spanish brief. "
        f"Statements: {[a.statement for a in assumptions]}"
    )

    audience_hits = [a.id for a in assumptions if _SPANISH_AUDIENCE.search(_assumption_text(a))]
    assert audience_hits, (
        "no assumption mentions the dog-owner audience or Madrid; the output is generic. "
        f"Statements: {[a.statement for a in assumptions]}"
    )

    if len(language_hits) * 2 < len(assumptions):
        warnings.warn(
            f"only {len(language_hits)}/{len(assumptions)} Spanish-brief assumptions carry a Spanish signal. "
            "The model may have translated the output into generic English. This is a soft check, not a failure.",
            UserWarning,
            stacklevel=1,
        )


def test_categories_cover_at_least_four_of_five(spanish_state: PipelineState) -> None:
    assumptions = spanish_state.assumptions.assumptions

    buckets = _category_buckets(assumptions)
    assert len(buckets) >= 4, (
        f"categories cover {sorted(buckets)}, need at least 4 of "
        f"{sorted(_CATEGORY_BUCKETS)}. Categories: {[a.category for a in assumptions]}"
    )


def test_german_personas_keep_berlin_office_context(german_personas_state: PipelineState) -> None:
    personas = german_personas_state.personas.personas

    matches = [p.id for p in personas if _OFFICE_BERLIN.search(f"{p.occupation} {p.background}")]
    assert matches, (
        "no persona's occupation or background reflects the Berlin office-worker context. "
        f"Occupations: {[p.occupation for p in personas]}"
    )


def test_no_placeholder_or_none_strings_leak(
    spanish_state: PipelineState,
    german_personas_state: PipelineState,
    spanish_questions_state: PipelineState,
) -> None:
    leaks = [
        (where, match.group(0))
        for where, text in _all_output_text(
            spanish_state.assumptions,
            german_personas_state.personas,
            spanish_questions_state.questions,
        )
        for match in _PLACEHOLDER.finditer(text)
    ]
    assert not leaks, f"broken placeholders or None-like strings leaked into the output: {leaks}"
