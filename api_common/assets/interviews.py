from pydantic import BaseModel


class QuestionPick(BaseModel):
    question_id: str | None  # None ends the interview


class Exchange(BaseModel):
    question_id: str
    question: str
    answer: str


class Interview(BaseModel):
    persona_id: str
    research: str  # what the deep research harness found about the persona's real-world segment
    exchanges: list[Exchange]


class InterviewsOutput(BaseModel):
    interviews: list[Interview]
