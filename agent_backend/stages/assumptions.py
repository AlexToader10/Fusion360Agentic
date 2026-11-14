from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineState

SYSTEM = """You are a lean-startup analyst. From the business described by the user, list the 6 to 8 riskiest \
assumptions it depends on: the ones that must be validated with real customers before building anything.
Cover the problem, the customer, the solution, the willingness to pay and the channel.
Each assumption is one falsifiable statement. Use the ids A1, A2, ..."""


class AssumptionsStage(Stage):
    name = "assumptions"

    def run(self, state: PipelineState) -> PipelineState:
        client = OpenRouterClient("assumptions")
        state.assumptions = client.parse(SYSTEM, business_context(state.request), AssumptionsOutput)
        return state
