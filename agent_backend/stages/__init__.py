from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.base import Stage
from agent_backend.stages.conclusions import ConclusionsStage
from agent_backend.stages.interviews import InterviewsStage
from agent_backend.stages.personas import PersonasStage
from agent_backend.stages.questions import QuestionsStage
from agent_backend.stages.validations import ValidationsStage

STAGES: list[Stage] = [
    AssumptionsStage(),
    PersonasStage(),
    QuestionsStage(),
    InterviewsStage(),
    ValidationsStage(),
    ConclusionsStage(),
]
