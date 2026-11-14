from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.conclusions import ConclusionsOutput
from api_common.assets.interviews import InterviewsOutput
from api_common.assets.personas import PersonasOutput
from api_common.assets.questions import QuestionsOutput
from api_common.assets.validations import ValidationsOutput


class PipelineRequest(BaseModel):
    business_idea: str
    target_audience: str
    additional_info: str = ""
    business_plan: str = ""  # text of the business plan PDF
    assumption_count: int | None = Field(None, ge=1, le=20)  # number of assumptions; None lets the model decide
    persona_count: int | None = Field(None, ge=1, le=30)  # size of the persona sample; None lets the model decide
    question_count: int | None = Field(None, ge=1, le=40)  # number of interview questions; None lets the model decide


class PipelineState(BaseModel):
    """Travels through the pipeline: every stage receives it and returns it with its own output filled in."""

    model_config = ConfigDict(validate_assignment=True)

    request: PipelineRequest
    assumptions: AssumptionsOutput | None = None
    personas: PersonasOutput | None = None
    questions: QuestionsOutput | None = None
    interviews: InterviewsOutput | None = None
    validations: ValidationsOutput | None = None
    conclusions: ConclusionsOutput | None = None


# Stage name -> output model, in pipeline order. Each name is a field of PipelineState.
STAGE_OUTPUTS: dict[str, type[BaseModel]] = {
    "assumptions": AssumptionsOutput,
    "personas": PersonasOutput,
    "questions": QuestionsOutput,
    "interviews": InterviewsOutput,
    "validations": ValidationsOutput,
    "conclusions": ConclusionsOutput,
}


class RunStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(BaseModel):
    id: UUID
    status: RunStatus = RunStatus.RUNNING
    error: str | None = None
    state: PipelineState


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    run: Run
