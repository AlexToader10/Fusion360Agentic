from pydantic import BaseModel


class Question(BaseModel):
    id: str  # Q1, Q2, ...
    text: str
    assumption_ids: list[str]  # assumptions this question validates or invalidates


class QuestionsOutput(BaseModel):
    questions: list[Question]


class JudgeFeedback(BaseModel):
    suggestions: list[str]  # how to adapt leading questions; empty when all are neutral
