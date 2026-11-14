"""Builders for valid stage outputs, so tests can assemble a PipelineState at any point of the pipeline."""

from api_common.assets.assumptions import Assumption, AssumptionsOutput
from api_common.assets.common import STAGE_OUTPUTS, PipelineRequest, PipelineState
from api_common.assets.conclusions import AssumptionConclusion, ConclusionsOutput
from api_common.assets.interviews import Exchange, Interview, InterviewsOutput
from api_common.assets.personas import Persona, PersonasOutput
from api_common.assets.questions import Question, QuestionsOutput
from api_common.assets.validations import AssumptionVerdict, InterviewValidation, ValidationsOutput

STAGE_NAMES = list(STAGE_OUTPUTS)
VERDICTS = ("validated", "invalidated", "inconclusive")


def request(tag: str = "idea", **overrides) -> PipelineRequest:
    fields = {"business_idea": tag, "target_audience": f"audience of {tag}"} | overrides
    return PipelineRequest(**fields)


def assumptions(n: int = 3, tag: str = "") -> AssumptionsOutput:
    return AssumptionsOutput(
        assumptions=[
            Assumption(id=f"A{i}", statement=f"statement {i} {tag}".strip(), category="problem", rationale=f"why {i}")
            for i in range(1, n + 1)
        ]
    )


def personas(n: int = 2, assumption_count: int = 3, tag: str = "") -> PersonasOutput:
    return PersonasOutput(
        personas=[
            Persona(
                id=f"P{i}",
                name=f"Person {i} {tag}".strip(),
                age=20 + i,
                occupation="baker",
                background="bakes bread at $3 a loaf",
                goals="sell more",
                pain_points="flour costs",
                assumption_ids=[f"A{j}" for j in range(1, assumption_count + 1)],
            )
            for i in range(1, n + 1)
        ]
    )


def questions(n: int = 3, tag: str = "") -> QuestionsOutput:
    return QuestionsOutput(
        questions=[
            Question(id=f"Q{i}", text=f"Tell me about the last time {i} {tag}".strip(), assumption_ids=[f"A{i}"])
            for i in range(1, n + 1)
        ]
    )


def interviews(persona_count: int = 2, question_count: int = 3, tag: str = "") -> InterviewsOutput:
    return InterviewsOutput(
        interviews=[
            Interview(
                persona_id=f"P{p}",
                research=f"research on P{p} {tag}".strip(),
                exchanges=[
                    Exchange(question_id=f"Q{q}", question=f"question {q}", answer=f"P{p} answer {q}")
                    for q in range(1, question_count + 1)
                ],
            )
            for p in range(1, persona_count + 1)
        ]
    )


def validations(persona_count: int = 2, assumption_count: int = 3, tag: str = "") -> ValidationsOutput:
    return ValidationsOutput(
        validations=[
            InterviewValidation(
                persona_id=f"P{p}",
                verdicts=[
                    AssumptionVerdict(
                        assumption_id=f"A{a}", verdict=VERDICTS[(p + a) % 3], evidence=f"P{p} on A{a} {tag}".strip()
                    )
                    for a in range(1, assumption_count + 1)
                ],
            )
            for p in range(1, persona_count + 1)
        ]
    )


def conclusions(assumption_count: int = 3, tag: str = "") -> ConclusionsOutput:
    return ConclusionsOutput(
        summary=f"summary {tag}".strip(),
        assumptions=[
            AssumptionConclusion(assumption_id=f"A{a}", verdict=VERDICTS[a % 3], reasoning=f"because {a}")
            for a in range(1, assumption_count + 1)
        ],
        recommendations=["interview real customers", "charge $5"],
    )


BUILDERS = {
    "assumptions": assumptions,
    "personas": personas,
    "questions": questions,
    "interviews": interviews,
    "validations": validations,
    "conclusions": conclusions,
}


def output(stage: str, tag: str = ""):
    return BUILDERS[stage](tag=tag)


def state_through(last_stage: str | None, tag: str = "idea", **request_overrides) -> PipelineState:
    """A state whose stages up to and including last_stage have an output (None: only the request)."""
    state = PipelineState(request=request(tag, **request_overrides))
    if last_stage is None:
        return state
    for stage in STAGE_NAMES[: STAGE_NAMES.index(last_stage) + 1]:
        setattr(state, stage, output(stage, tag))
    return state


def full_state(tag: str = "idea") -> PipelineState:
    return state_through(STAGE_NAMES[-1], tag)
