from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context, count_rule, reindex, trim_count
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineState

SYSTEM = """You are a lean-startup analyst. From the business described by the user, list the {count} riskiest \
assumptions it depends on: the ones that must be validated with real customers before building anything.
Cover the problem, the customer, the solution, the willingness to pay and the channel.
Each assumption is one falsifiable statement. Use the ids A1, A2, ...
{count_rule}"""

DEFAULT_COUNT = "6 to 8"


class AssumptionsStage(Stage):
    name = "assumptions"

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report = report or Reporter.noop(self.name)
        client = OpenRouterClient("assumptions")
        count, rule = count_rule(state.request.assumption_count, DEFAULT_COUNT, "assumption", "assumptions")
        report.status(f"Asking {client.model} for {count} risky assumptions…", model=client.model)
        parsed = client.parse(
            SYSTEM.format(count=count, count_rule=rule), business_context(state.request), AssumptionsOutput
        )
        assumptions = trim_count(parsed.assumptions, state.request.assumption_count, report, "assumptions")
        reindex(assumptions, "A")
        state.assumptions = AssumptionsOutput(assumptions=assumptions)
        report.status(f"Found {len(assumptions)} assumptions", count=len(assumptions))
        return state
