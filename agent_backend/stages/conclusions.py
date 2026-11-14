from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from api_common.assets.common import PipelineState
from api_common.assets.conclusions import ConclusionsOutput

SYSTEM = """You are a startup advisor. You receive a business idea, its assumptions and, for each simulated \
customer interview, a verdict per assumption.
Give every assumption an overall verdict across the interviews with your reasoning, summarise what the interviews \
say about the business, and recommend concrete next steps: what to keep, what to change and what to test with \
real customers.
Be candid. The interviews are simulated, so say where real-world evidence is still needed."""


class ConclusionsStage(Stage):
    name = "conclusions"

    def run(self, state: PipelineState) -> PipelineState:
        client = OpenRouterClient("conclusions")
        prompt = (
            f"{business_context(state.request)}\n\n"
            f"Assumptions:\n{state.assumptions.model_dump_json(indent=2)}\n\n"
            f"Verdicts per interview:\n{state.validations.model_dump_json(indent=2)}"
        )
        state.conclusions = client.parse(SYSTEM, prompt, ConclusionsOutput)
        return state
