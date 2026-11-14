from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from api_common.assets.common import PipelineState
from api_common.assets.personas import PersonasOutput

SYSTEM = """You design personas for customer-discovery interviews. From the target audience, create {count} \
distinct, realistic people who form a representative sample of it: cover every segment the audience contains, in \
proportion to its weight, and vary their situation, habits and attitude from enthusiasts to skeptics.
Spread the assumptions so that each one can be spoken to from experience by as many personas as possible, \
at least 3 when the sample is large enough.
Use the ids P1, P2, ... and list in assumption_ids the assumptions each persona can speak to."""

DEFAULT_COUNT = "8 to 10"


class PersonasStage(Stage):
    name = "personas"

    def run(self, state: PipelineState) -> PipelineState:
        client = OpenRouterClient("personas")
        system = SYSTEM.format(count=state.request.persona_count or DEFAULT_COUNT)
        prompt = f"{business_context(state.request)}\n\nAssumptions:\n{state.assumptions.model_dump_json(indent=2)}"
        state.personas = client.parse(system, prompt, PersonasOutput)
        return state
