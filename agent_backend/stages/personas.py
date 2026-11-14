from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context, count_rule, reindex, trim_count
from api_common.assets.common import PipelineState
from api_common.assets.personas import PersonasOutput

SYSTEM = """You design personas for customer-discovery interviews. From the target audience, create {count} \
distinct, realistic people who form a representative sample of it: cover every segment the audience contains, in \
proportion to its weight, and vary their situation, habits and attitude from enthusiasts to skeptics.
Spread the assumptions so that each one can be spoken to from experience by as many personas as possible, \
at least 3 when the sample is large enough.
Use the ids P1, P2, ... and list in assumption_ids the assumptions each persona can speak to.
{count_rule}"""

DEFAULT_COUNT = "8 to 10"


class PersonasStage(Stage):
    name = "personas"

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report = report or Reporter.noop(self.name)
        client = OpenRouterClient("personas")
        count, rule = count_rule(state.request.persona_count, DEFAULT_COUNT, "persona", "personas")
        system = SYSTEM.format(count=count, count_rule=rule)
        prompt = f"{business_context(state.request)}\n\nAssumptions:\n{state.assumptions.model_dump_json(indent=2)}"
        report.status(f"Designing {count} personas with {client.model}…", model=client.model)
        parsed = client.parse(system, prompt, PersonasOutput)
        personas = trim_count(parsed.personas, state.request.persona_count, report, "personas")
        reindex(personas, "P")
        state.personas = PersonasOutput(personas=personas)
        report.status(f"Created {len(personas)} personas", count=len(personas))
        return state
