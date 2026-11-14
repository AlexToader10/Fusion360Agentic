from pydantic import BaseModel

from api_common.assets.validations import Verdict


class AssumptionConclusion(BaseModel):
    assumption_id: str
    verdict: Verdict  # overall verdict across all interviews
    reasoning: str


class ConclusionsOutput(BaseModel):
    summary: str
    assumptions: list[AssumptionConclusion]
    recommendations: list[str]
