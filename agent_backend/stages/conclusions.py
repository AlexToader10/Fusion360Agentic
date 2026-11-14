from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from api_common.assets.common import PipelineState
from api_common.assets.conclusions import ConclusionsOutput

SYSTEM = """You are a startup advisor. You receive a business idea, its assumptions and, for each simulated \
customer interview, a verdict per assumption.
Give every assumption an overall verdict across the interviews with your reasoning, summarise what the interviews \
say about the business, and recommend concrete next steps: what to keep, what to change and what to test with \
real customers.
Aggregate the verdicts you receive: when every interview gives the same verdict, give that verdict; when they \
conflict, use inconclusive unless one side is clearly stronger. Do not invent evidence beyond the verdicts.
Be candid. The interviews are simulated, so say where real-world evidence is still needed."""


class ConclusionsStage(Stage):
    name = "conclusions"

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report = report or Reporter.noop(self.name)
        client = OpenRouterClient("conclusions")
        report.status(
            f"Synthesising conclusions from {len(state.validations.validations)} interviews with {client.model}…",
            model=client.model,
        )
        prompt = (
            f"{business_context(state.request)}\n\n"
            f"Assumptions:\n{state.assumptions.model_dump_json(indent=2)}\n\n"
            f"Verdicts per interview:\n{state.validations.model_dump_json(indent=2)}"
        )
        state.conclusions = client.parse(SYSTEM, prompt, ConclusionsOutput)
        report.status(f"Conclusions ready for {len(state.conclusions.assumptions)} assumptions")
        return state
