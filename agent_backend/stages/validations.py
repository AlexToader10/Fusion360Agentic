from concurrent.futures import ThreadPoolExecutor

from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage
from api_common.assets.common import PipelineState
from api_common.assets.interviews import Interview
from api_common.assets.validations import InterviewValidation, ValidationsOutput

SYSTEM = """You analyse one customer interview against a list of business assumptions.
Give every assumption a verdict for this interviewee:
- validated: what they said is evidence the assumption holds for them
- invalidated: what they said is evidence it does not hold for them
- inconclusive: the interview does not give enough evidence either way
Judge only on what was said in the answers and cite it as evidence."""


class ValidationsStage(Stage):
    name = "validations"

    def run(self, state: PipelineState) -> PipelineState:
        with ThreadPoolExecutor() as pool:
            validations = pool.map(lambda interview: self._validate(interview, state), state.interviews.interviews)
            state.validations = ValidationsOutput(validations=list(validations))
        return state

    @staticmethod
    def _validate(interview: Interview, state: PipelineState) -> InterviewValidation:
        prompt = (
            f"Assumptions:\n{state.assumptions.model_dump_json(indent=2)}\n\n"
            f"Interview with persona {interview.persona_id}:\n"
            f"{interview.model_dump_json(indent=2, include={'persona_id', 'exchanges'})}"
        )
        return OpenRouterClient("validation").parse(SYSTEM, prompt, InterviewValidation)
