from concurrent.futures import ThreadPoolExecutor

from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from agent_backend.stages.deep_research import DeepResearchHarness
from api_common.assets.common import PipelineState
from api_common.assets.interviews import Exchange, Interview, InterviewsOutput, QuestionPick
from api_common.assets.personas import Persona
from api_common.assets.questions import QuestionsOutput

INTERVIEWER_SYSTEM = """You run a customer-discovery interview to test business assumptions.
Pick the remaining question that is most worth asking this persona next, given the conversation so far.
Return null to end the interview when no remaining question is relevant to this persona."""


class InterviewsStage(Stage):
    name = "interviews"

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report = report or Reporter.noop(self.name)
        personas = state.personas.personas
        report.status(f"Interviewing {len(personas)} personas in parallel…", personas=len(personas))
        with ThreadPoolExecutor() as pool:
            interviews = pool.map(
                lambda persona: self._interview(persona, state, report.child(persona_id=persona.id, persona=persona.name)),
                personas,
            )
            state.interviews = InterviewsOutput(interviews=list(interviews))
        answered = sum(len(interview.exchanges) for interview in state.interviews.interviews)
        report.status(f"Finished {len(state.interviews.interviews)} interviews with {answered} answers", answers=answered)
        return state

    def _interview(self, persona: Persona, state: PipelineState, report: Reporter) -> Interview:
        harness = DeepResearchHarness(persona, business_context(state.request), report=report)
        interviewer = OpenRouterClient("interviewer")
        remaining = {question.id: question for question in state.questions.questions}
        exchanges: list[Exchange] = []

        while remaining:
            transcript = "\n\n".join(f"{e.question}\nAnswer: {e.answer}" for e in exchanges)
            prompt = (
                f"Persona:\n{persona.model_dump_json(indent=2)}\n\n"
                f"Assumptions:\n{state.assumptions.model_dump_json(indent=2)}\n\n"
                f"Conversation so far:\n{transcript or 'Nothing asked yet.'}\n\n"
                f"Remaining questions:\n{QuestionsOutput(questions=list(remaining.values())).model_dump_json(indent=2)}"
            )
            report.status(f"Choosing the next question for {persona.name}…", remaining=len(remaining))
            question = remaining.pop(interviewer.parse(INTERVIEWER_SYSTEM, prompt, QuestionPick).question_id, None)
            if question is None:
                report.status(f"{persona.name} has nothing more to add", remaining=len(remaining))
                break
            report.status(f"Asking {persona.name}: {question.id}", question_id=question.id)
            answer = harness.answer(question.text, exchanges)
            exchanges.append(Exchange(question_id=question.id, question=question.text, answer=answer))
            report.status(f"{persona.name} answered {question.id}", question_id=question.id)

        return Interview(persona_id=persona.id, research=harness.research, exchanges=exchanges)
