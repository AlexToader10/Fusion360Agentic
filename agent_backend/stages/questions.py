from agent_backend.events import Reporter
from agent_backend.openrouter_client import OpenRouterClient
from agent_backend.stages.base import Stage, business_context, count_rule, reindex, trim_count
from api_common.assets.common import PipelineState
from api_common.assets.questions import JudgeFeedback, QuestionsOutput

GENERATOR_SYSTEM = """You write customer-interview questions that validate or invalidate business assumptions.
Questions are open-ended and ask about past behaviour and concrete experiences, never about hypothetical futures.
They never mention the product, pitch the idea or hint at the answer the business hopes for.
Write {count} questions with the ids Q1, Q2, ..., each listing in assumption_ids the assumptions it tests.
Every assumption must be tested by at least one question.
{count_rule}"""

JUDGE_SYSTEM = """You review customer-interview questions for bias. A question is leading when it suggests the \
desired answer, presupposes the problem or the solution, pitches the product, or invites a yes/no that confirms \
the assumption.
Return one suggestion per question that needs adapting, naming its id and how to fix it.
Return no suggestions when every question is neutral."""

DEFAULT_COUNT = "6 to 10"
MAX_JUDGE_ROUNDS = 3


class QuestionsStage(Stage):
    name = "questions"

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report = report or Reporter.noop(self.name)
        generator = OpenRouterClient("questions_generator")
        judge = OpenRouterClient("questions_judge")
        count, rule = count_rule(state.request.question_count, DEFAULT_COUNT, "question", "questions")
        generator_system = GENERATOR_SYSTEM.format(count=count, count_rule=rule)
        context = f"{business_context(state.request)}\n\nAssumptions:\n{state.assumptions.model_dump_json(indent=2)}"

        report.status(f"Drafting {count} interview questions with {generator.model}…", model=generator.model)
        questions = self._normalise(
            generator.parse(generator_system, context, QuestionsOutput), state.request.question_count, report
        )
        rounds = 0
        while rounds < MAX_JUDGE_ROUNDS and (suggestions := self._judge(judge, questions)):
            report.status(
                f"Bias judge round {rounds + 1}/{MAX_JUDGE_ROUNDS}: {len(suggestions)} question(s) to fix",
                round=rounds + 1,
                suggestions=len(suggestions),
            )
            edit_prompt = (
                f"{context}\n\nYour questions:\n{questions.model_dump_json(indent=2)}\n\n"
                "The judge asks you to adapt them:\n" + "\n".join(f"- {s}" for s in suggestions)
            )
            questions = self._normalise(
                generator.parse(generator_system, edit_prompt, QuestionsOutput),
                state.request.question_count,
                report,
            )
            rounds += 1

        report.status(
            f"Kept {len(questions.questions)} questions after {rounds} judge round(s)",
            count=len(questions.questions),
            rounds=rounds,
        )
        state.questions = questions
        return state

    @staticmethod
    def _normalise(questions: QuestionsOutput, count: int | None, report: Reporter) -> QuestionsOutput:
        kept = trim_count(questions.questions, count, report, "questions")
        reindex(kept, "Q")
        return QuestionsOutput(questions=kept)

    @staticmethod
    def _judge(judge: OpenRouterClient, questions: QuestionsOutput) -> list[str]:
        return judge.parse(JUDGE_SYSTEM, questions.model_dump_json(indent=2), JudgeFeedback).suggestions
