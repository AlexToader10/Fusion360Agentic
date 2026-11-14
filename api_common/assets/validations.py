from typing import Literal

from pydantic import BaseModel

Verdict = Literal["validated", "invalidated", "inconclusive"]


class AssumptionVerdict(BaseModel):
    assumption_id: str
    verdict: Verdict
    evidence: str  # what the persona said that supports the verdict


class InterviewValidation(BaseModel):
    persona_id: str
    verdicts: list[AssumptionVerdict]


class ValidationsOutput(BaseModel):
    validations: list[InterviewValidation]
