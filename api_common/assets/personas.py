from pydantic import BaseModel


class Persona(BaseModel):
    id: str  # P1, P2, ...
    name: str
    age: int
    occupation: str
    background: str
    goals: str
    pain_points: str
    assumption_ids: list[str]  # assumptions this persona can speak to


class PersonasOutput(BaseModel):
    personas: list[Persona]
