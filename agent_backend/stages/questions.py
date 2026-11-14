from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context
from api_common.assets.common import PipelineState
from api_common.assets.questions import JudgeFeedback, QuestionsOutput

GENERATOR_SYSTEM = """You write customer-interview questions that validate or invalidate business assumptions.
Questions are open-ended and ask about past behaviour and concrete experiences, never about hypothetical futures.
They never mention the product, pitch the idea or hint at the answer the business hopes for.
Write 6 to 10 questions with the ids Q1, Q2, ..., each listing in assumption_ids the assumptions it tests.
Every assumption must be tested by at least one question."""

JUDGE_SYSTEM = """You review customer-interview questions for bias. A question is leading when it suggests the \
desired answer, presupposes the problem or the solution, pitches the product, or invites a yes/no that confirms \
the assumption.
Return one suggestion per question that needs adapting, naming its id and how to fix it.
Return no suggestions when every question is neutral."""

MAX_JUDGE_ROUNDS = 3


class QuestionsStage(Stage):
    name = "questions"

    def run(self, state: PipelineState) -> PipelineState:
        generator = OpenRouterClient("questions_generator")
        judge = OpenRouterClient("questions_judge")
        context = f"{business_context(state.request)}\n\nAssumptions:\n{state.assumptions.model_dump_json(indent=2)}"

        questions = generator.parse(GENERATOR_SYSTEM, context, QuestionsOutput)
        rounds = 0
        while rounds < MAX_JUDGE_ROUNDS and (suggestions := self._judge(judge, questions)):
            edit_prompt = (
                f"{context}\n\nYour questions:\n{questions.model_dump_json(indent=2)}\n\n"
                "The judge asks you to adapt them:\n" + "\n".join(f"- {s}" for s in suggestions)
            )
            questions = generator.parse(GENERATOR_SYSTEM, edit_prompt, QuestionsOutput)
            rounds += 1

        state.questions = questions
        return state

    @staticmethod
    def _judge(judge: OpenRouterClient, questions: QuestionsOutput) -> list[str]:
        return judge.parse(JUDGE_SYSTEM, questions.model_dump_json(indent=2), JudgeFeedback).suggestions
