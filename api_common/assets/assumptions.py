from pydantic import BaseModel


class Assumption(BaseModel):
    id: str  # A1, A2, ...
    statement: str
    category: str  # problem, customer, solution, willingness to pay, channel, ...
    rationale: str  # why the business depends on it


class AssumptionsOutput(BaseModel):
    assumptions: list[Assumption]
