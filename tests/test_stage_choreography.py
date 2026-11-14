"""Offline choreography tests: prompt assembly, call order and progress reporting of each stage.

Model calls go through FakeOpenRouterClient or small routing fakes that answer by persona id, so thread pools cannot
reorder the responses. Nothing here touches the network.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from agent_backend import openrouter_client
from agent_backend.events import Reporter
from agent_backend.openrouter_client import BASE_URL, OpenRouterClient, model_for
from agent_backend.stages import assumptions as assumptions_stage
from agent_backend.stages import conclusions as conclusions_stage
from agent_backend.stages import deep_research
from agent_backend.stages import interviews as interviews_stage
from agent_backend.stages import personas as personas_stage
from agent_backend.stages import questions as questions_stage
from agent_backend.stages import validations as validations_stage
from agent_backend.stages.base import business_context
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import PipelineRequest, PipelineState
from api_common.assets.conclusions import ConclusionsOutput
from api_common.assets.interviews import Exchange, QuestionPick
from api_common.assets.personas import Persona, PersonasOutput
from api_common.assets.questions import JudgeFeedback, QuestionsOutput
from tests.helpers import FakeOpenRouterClient, make_assumption, make_persona, make_question


def _reporter(stage: str) -> tuple[list[dict[str, Any]], Reporter]:
    events: list[dict[str, Any]] = []
    return events, Reporter(stage, events.append)


def _use_fake(monkeypatch: pytest.MonkeyPatch, module: Any, fake: FakeOpenRouterClient) -> FakeOpenRouterClient:
    """Route `OpenRouterClient(task)` in one stage module to the fake. Stages read `.model` for status text."""

    def factory(task: str) -> Any:
        view = fake(task)
        view.model = f"offline/{task}"
        return view

    monkeypatch.setattr(module, "OpenRouterClient", factory)
    return fake


def _with_assumptions(request: PipelineRequest, assumptions: AssumptionsOutput, **request_updates: Any) -> PipelineState:
    return PipelineState(request=request.model_copy(update=request_updates), assumptions=assumptions)


# Business context


def test_business_context_keeps_only_filled_fields_in_order() -> None:
    partial = PipelineRequest(business_idea="Idea text", target_audience="", additional_info="Extra notes", business_plan="Plan text")
    assert business_context(partial) == "Business idea:\nIdea text\n\nAdditional info:\nExtra notes\n\nBusiness plan:\nPlan text"

    full = PipelineRequest(business_idea="I", target_audience="A", additional_info="X", business_plan="P")
    context = business_context(full)
    positions = [context.index(title) for title in ("Business idea:", "Target audience:", "Additional info:", "Business plan:")]
    assert positions == sorted(positions)


# Assumptions and personas


def test_assumptions_with_count_seven_asks_for_seven_and_sends_context(
    monkeypatch: pytest.MonkeyPatch, sample_request: PipelineRequest, sample_assumptions: AssumptionsOutput
) -> None:
    request = sample_request.model_copy(update={"assumption_count": 7})
    fake = _use_fake(monkeypatch, assumptions_stage, FakeOpenRouterClient(sample_assumptions))
    events, report = _reporter("assumptions")

    result = assumptions_stage.AssumptionsStage().run(PipelineState(request=request), report)

    call = fake.calls[0]
    assert call.task == "assumptions"
    assert call.output_model is AssumptionsOutput
    assert "the 7 riskiest assumptions" in call.system
    assert call.prompt == business_context(request)
    assert result.assumptions == sample_assumptions
    assert "7 risky assumptions" in events[0]["message"]
    assert events[-1]["data"]["count"] == len(sample_assumptions.assumptions)


def test_assumptions_without_count_lets_the_model_pick_six_to_eight(
    monkeypatch: pytest.MonkeyPatch, sample_request: PipelineRequest, sample_assumptions: AssumptionsOutput
) -> None:
    request = sample_request.model_copy(update={"assumption_count": None})
    fake = _use_fake(monkeypatch, assumptions_stage, FakeOpenRouterClient(sample_assumptions))
    events, report = _reporter("assumptions")

    assumptions_stage.AssumptionsStage().run(PipelineState(request=request), report)

    assert "6 to 8 riskiest assumptions" in fake.calls[0].system
    assert "6 to 8 risky" in events[0]["message"]


def test_personas_with_count_four_asks_for_four_and_includes_assumptions(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_personas: PersonasOutput,
) -> None:
    fake = _use_fake(monkeypatch, personas_stage, FakeOpenRouterClient(sample_personas))
    state = _with_assumptions(sample_request, sample_assumptions, persona_count=4)
    events, report = _reporter("personas")

    result = personas_stage.PersonasStage().run(state, report)

    call = fake.calls[0]
    assert call.task == "personas"
    assert call.output_model is PersonasOutput
    assert "create 4 distinct" in call.system
    assert business_context(state.request) in call.prompt
    assert sample_assumptions.model_dump_json(indent=2) in call.prompt
    assert result.personas == sample_personas
    assert events[-1]["data"]["count"] == len(sample_personas.personas)


def test_personas_without_count_use_eight_to_ten(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_personas: PersonasOutput,
) -> None:
    fake = _use_fake(monkeypatch, personas_stage, FakeOpenRouterClient(sample_personas))
    state = _with_assumptions(sample_request, sample_assumptions, persona_count=None)

    personas_stage.PersonasStage().run(state, Reporter.noop("personas"))

    assert "create 8 to 10 distinct" in fake.calls[0].system


# Questions: generator and bias judge loop


def test_questions_without_judge_suggestions_call_each_model_once(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_questions: QuestionsOutput,
) -> None:
    fake = _use_fake(
        monkeypatch,
        questions_stage,
        FakeOpenRouterClient(sample_questions, JudgeFeedback(suggestions=[])),
    )
    state = _with_assumptions(sample_request, sample_assumptions)

    result = questions_stage.QuestionsStage().run(state, Reporter.noop("questions"))

    assert fake.tasks == ["questions_generator", "questions_judge"]
    assert result.questions == sample_questions


def test_judge_suggestion_triggers_a_rewrite_that_carries_the_feedback(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_questions: QuestionsOutput,
) -> None:
    feedback = "Q2 presupposes that someone plans the meals; ask who does it instead."
    revised = QuestionsOutput(questions=[make_question("Q1", ["A1"], text="Describe the last dinner you cooked.")])
    fake = _use_fake(
        monkeypatch,
        questions_stage,
        FakeOpenRouterClient(
            sample_questions,
            JudgeFeedback(suggestions=[feedback]),
            revised,
            JudgeFeedback(suggestions=[]),
        ),
    )
    state = _with_assumptions(sample_request, sample_assumptions)
    events, report = _reporter("questions")

    result = questions_stage.QuestionsStage().run(state, report)

    assert fake.tasks == ["questions_generator", "questions_judge", "questions_generator", "questions_judge"]
    rewrite = fake.calls[2]
    assert f"- {feedback}" in rewrite.prompt
    assert sample_questions.model_dump_json(indent=2) in rewrite.prompt
    assert result.questions == revised
    assert [e["data"]["round"] for e in events if "round" in e.get("data", {})] == [1]


def test_judge_loop_stops_at_max_rounds_even_if_judge_always_objects(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_questions: QuestionsOutput,
) -> None:
    objection = JudgeFeedback(suggestions=["Q1 leads the respondent toward a yes."])
    rounds = questions_stage.MAX_JUDGE_ROUNDS
    responses: list[Any] = [sample_questions]
    for _ in range(rounds):
        responses += [objection, sample_questions]
    fake = _use_fake(monkeypatch, questions_stage, FakeOpenRouterClient(*responses))
    state = _with_assumptions(sample_request, sample_assumptions)
    events, report = _reporter("questions")

    questions_stage.QuestionsStage().run(state, report)

    assert fake.tasks.count("questions_generator") == 1 + rounds
    assert fake.tasks.count("questions_judge") == rounds
    assert [e["data"]["round"] for e in events if "round" in e.get("data", {})] == list(range(1, rounds + 1))
    assert events[-1]["data"]["rounds"] == rounds


def test_question_count_appears_in_generator_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    sample_request: PipelineRequest,
    sample_assumptions: AssumptionsOutput,
    sample_questions: QuestionsOutput,
) -> None:
    fake = _use_fake(
        monkeypatch,
        questions_stage,
        FakeOpenRouterClient(sample_questions, JudgeFeedback(suggestions=[])),
    )
    questions_stage.QuestionsStage().run(
        _with_assumptions(sample_request, sample_assumptions, question_count=5), Reporter.noop("questions")
    )
    assert "Write 5 questions" in fake.calls[0].system

    fallback = _use_fake(
        monkeypatch,
        questions_stage,
        FakeOpenRouterClient(sample_questions, JudgeFeedback(suggestions=[])),
    )
    questions_stage.QuestionsStage().run(
        _with_assumptions(sample_request, sample_assumptions, question_count=None), Reporter.noop("questions")
    )
    assert "6 to 10 questions" in fallback.calls[0].system


def test_judge_sends_questions_json_and_returns_its_suggestions(sample_questions: QuestionsOutput) -> None:
    feedback = JudgeFeedback(suggestions=["Q4 names a price, which presupposes the answer."])
    fake = FakeOpenRouterClient(feedback)

    suggestions = questions_stage.QuestionsStage._judge(fake("questions_judge"), sample_questions)

    assert suggestions == feedback.suggestions
    call = fake.calls[0]
    assert call.task == "questions_judge"
    assert call.system == questions_stage.JUDGE_SYSTEM
    assert call.prompt == sample_questions.model_dump_json(indent=2)
    assert call.output_model is JudgeFeedback


# Interviews


class _Interviewer:
    """Answers the interviewer's QuestionPick calls by persona id, read from the persona JSON in the prompt.

    A persona's script is a list of question ids. None, or an exhausted script, ends that interview.
    """

    def __init__(self, scripts: dict[str, list[str | None]]) -> None:
        self._scripts = {persona_id: deque(picks) for persona_id, picks in scripts.items()}
        self._lock = threading.Lock()

    def parse(self, system: str, prompt: str, output_model: type[QuestionPick]) -> QuestionPick:
        assert output_model is QuestionPick
        persona_id = re.search(r'"id": "(P\d+)"', prompt)
        assert persona_id is not None
        with self._lock:
            script = self._scripts[persona_id.group(1)]
            pick = script.popleft() if script else None
        return QuestionPick(question_id=pick)


def _install_interviewer(monkeypatch: pytest.MonkeyPatch, scripts: dict[str, list[str | None]]) -> None:
    interviewer = _Interviewer(scripts)
    monkeypatch.setattr(interviews_stage, "OpenRouterClient", lambda task: interviewer)


def _install_harnesses(monkeypatch: pytest.MonkeyPatch, delays: dict[str, float] | None = None) -> list[Any]:
    """Replace DeepResearchHarness in the interviews module. Each harness reports through the report it receives."""
    made: list[Any] = []

    class FakeHarness:
        def __init__(self, persona: Persona, business: str, report: Reporter | None = None) -> None:
            time.sleep((delays or {}).get(persona.id, 0.0))
            self.persona = persona
            self.business = business
            self.research = f"Research notes about {persona.name}."
            self.answered: list[tuple[str, list[Exchange]]] = []
            self.report = report
            made.append(self)
            self.report.status(f"Harness ready for {persona.name}")

        def answer(self, question: str, exchanges: list[Exchange]) -> str:
            self.answered.append((question, list(exchanges)))
            return f"{self.persona.id} on {question}"

    monkeypatch.setattr(interviews_stage, "DeepResearchHarness", FakeHarness)
    return made


def test_each_persona_gets_a_harness_one_exchange_and_persona_tagged_events(
    monkeypatch: pytest.MonkeyPatch, full_state: PipelineState
) -> None:
    full_state.personas = PersonasOutput(personas=full_state.personas.personas[:2])
    made = _install_harnesses(monkeypatch)
    _install_interviewer(monkeypatch, {"P1": ["Q1"], "P2": ["Q1"]})
    events, report = _reporter("interviews")

    result = interviews_stage.InterviewsStage().run(full_state, report)

    assert sorted(harness.persona.id for harness in made) == ["P1", "P2"]
    by_persona = {interview.persona_id: interview for interview in result.interviews.interviews}
    assert set(by_persona) == {"P1", "P2"}
    names = {persona.id: persona.name for persona in full_state.personas.personas}
    for persona_id, interview in by_persona.items():
        assert interview.research == f"Research notes about {names[persona_id]}."
        assert [(e.question_id, e.answer) for e in interview.exchanges] == [("Q1", f"{persona_id} on {full_state.questions.questions[0].text}")]
    assert {e["data"]["persona_id"] for e in events if "persona_id" in e.get("data", {})} == {"P1", "P2"}


@pytest.mark.parametrize(
    ("picks", "expected_ids"),
    [(["Q99"], []), (["Q1", "Q1"], ["Q1"])],
    ids=["unknown-id-ends-interview", "repeated-id-ends-interview"],
)
def test_unknown_or_repeated_pick_ends_the_interview_without_duplicating_exchanges(
    monkeypatch: pytest.MonkeyPatch,
    full_state: PipelineState,
    picks: list[str],
    expected_ids: list[str],
) -> None:
    full_state.personas = PersonasOutput(personas=full_state.personas.personas[:1])
    made = _install_harnesses(monkeypatch)
    _install_interviewer(monkeypatch, {"P1": picks})

    result = interviews_stage.InterviewsStage().run(full_state, Reporter.noop("interviews"))

    interview = result.interviews.interviews[0]
    assert [e.question_id for e in interview.exchanges] == expected_ids
    assert len(made[0].answered) == len(expected_ids)


def test_interviews_keep_persona_order_despite_varied_delays(
    monkeypatch: pytest.MonkeyPatch, full_state: PipelineState
) -> None:
    _install_harnesses(monkeypatch, delays={"P1": 0.06, "P2": 0.0, "P3": 0.03})
    _install_interviewer(monkeypatch, {"P1": [], "P2": [], "P3": []})

    result = interviews_stage.InterviewsStage().run(full_state, Reporter.noop("interviews"))

    assert [interview.persona_id for interview in result.interviews.interviews] == ["P1", "P2", "P3"]


# Validations


class _Router:
    """Answers each validation call with the verdicts for the persona named in the prompt."""

    def __init__(self, answers: dict[str, Any], delays: dict[str, float]) -> None:
        self.answers = answers
        self.delays = delays
        self.prompts: dict[str, str] = {}
        self.tasks: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, task: str) -> _Router:
        self.tasks.append(task)
        return self

    def parse(self, system: str, prompt: str, output_model: type[Any]) -> Any:
        persona_id = re.search(r"Interview with persona (P\d+):", prompt)
        assert persona_id is not None
        time.sleep(self.delays.get(persona_id.group(1), 0.0))
        with self._lock:
            self.prompts[persona_id.group(1)] = prompt
        return self.answers[persona_id.group(1)]


def test_validations_prompt_omits_research_and_keeps_interview_order(
    monkeypatch: pytest.MonkeyPatch, full_state: PipelineState, sample_validations: Any
) -> None:
    answers = {validation.persona_id: validation for validation in sample_validations.validations}
    router = _Router(answers, delays={"P1": 0.05})
    monkeypatch.setattr(validations_stage, "OpenRouterClient", router)
    events, report = _reporter("validations")

    result = validations_stage.ValidationsStage().run(full_state, report)

    assert router.tasks == ["validation", "validation"]
    assert [v.persona_id for v in result.validations.validations] == ["P1", "P2"]
    assert result.validations.validations == sample_validations.validations
    for interview in full_state.interviews.interviews:
        prompt = router.prompts[interview.persona_id]
        assert f"Interview with persona {interview.persona_id}:" in prompt
        assert interview.research not in prompt
        assert '"research"' not in prompt
        for exchange in interview.exchanges:
            assert exchange.answer in prompt
    assert {e["data"]["persona_id"] for e in events if "persona_id" in e.get("data", {})} == {"P1", "P2"}


# Conclusions


def test_conclusions_prompt_carries_context_assumptions_and_validations(
    monkeypatch: pytest.MonkeyPatch, full_state: PipelineState, sample_conclusions: ConclusionsOutput
) -> None:
    fake = _use_fake(monkeypatch, conclusions_stage, FakeOpenRouterClient(sample_conclusions))
    events, report = _reporter("conclusions")

    result = conclusions_stage.ConclusionsStage().run(full_state, report)

    call = fake.calls[0]
    assert call.task == "conclusions"
    assert call.output_model is ConclusionsOutput
    assert business_context(full_state.request) in call.prompt
    assert full_state.assumptions.model_dump_json(indent=2) in call.prompt
    assert full_state.validations.model_dump_json(indent=2) in call.prompt
    assert result.conclusions == sample_conclusions
    assert events[0]["data"]["model"] == "offline/conclusions"
    assert events[-1]["message"] == f"Conclusions ready for {len(sample_conclusions.assumptions)} assumptions"


# Deep research harness


def _install_text_client(monkeypatch: pytest.MonkeyPatch, research: str, deltas: list[str]) -> list[dict[str, Any]]:
    """Fake text client that can stream: calls with on_token get each delta. Returns the recorded calls."""
    calls: list[dict[str, Any]] = []

    class FakeTextClient:
        def __init__(self, task: str) -> None:
            self.task = task

        def text(
            self,
            system: str,
            prompt: str,
            web_search: bool = False,
            on_token: Any = None,
        ) -> str:
            calls.append(
                {
                    "task": self.task,
                    "system": system,
                    "prompt": prompt,
                    "web_search": web_search,
                    "streamed": on_token is not None,
                }
            )
            if web_search:
                return research
            for delta in deltas:
                if on_token is not None:
                    on_token(delta)
            return "".join(deltas)

    monkeypatch.setattr(deep_research, "OpenRouterClient", FakeTextClient)
    return calls


def test_harness_researches_with_web_search_on_construction(
    monkeypatch: pytest.MonkeyPatch, sample_personas: PersonasOutput
) -> None:
    calls = _install_text_client(monkeypatch, research="Notes on the segment.", deltas=[])
    persona = sample_personas.personas[0]
    events, report = _reporter("interviews")

    harness = deep_research.DeepResearchHarness(persona, "Business text", report=report)

    assert harness.research == "Notes on the segment."
    assert len(calls) == 1
    assert calls[0]["task"] == "persona"
    assert calls[0]["web_search"] is True
    assert "Business text" in calls[0]["prompt"]
    assert persona.background in calls[0]["prompt"]
    assert '"assumption_ids"' not in calls[0]["prompt"]
    assert [e["data"]["persona_id"] for e in events] == [persona.id, persona.id]


def test_harness_answer_passes_prior_exchanges_and_streams_tokens(
    monkeypatch: pytest.MonkeyPatch, sample_personas: PersonasOutput, sample_interviews: Any
) -> None:
    calls = _install_text_client(monkeypatch, research="Notes on the segment.", deltas=["I would ", "wait."])
    persona = sample_personas.personas[0]
    events, report = _reporter("interviews")
    harness = deep_research.DeepResearchHarness(persona, "Business text", report=report)
    prior = sample_interviews.interviews[0].exchanges[:2]

    answer = harness.answer("Would you order a kit?", prior)

    call = calls[-1]
    transcript = "\n\n".join(f"Interviewer: {e.question}\nYou: {e.answer}" for e in prior)
    assert call["prompt"] == f"{transcript}\n\nInterviewer: Would you order a kit?"
    assert call["streamed"] is True
    assert call["web_search"] is False
    assert "You are Laura Meyer, 41, Project manager," in call["system"]
    assert "Notes on the segment." in call["system"]
    assert answer == "I would wait."
    tokens = [e for e in events if e["kind"] == "token"]
    assert [e["message"] for e in tokens] == ["I would ", "wait."]
    assert all(e["data"]["persona_id"] == persona.id for e in tokens)


# OpenRouter client


def _patch_openai(monkeypatch: pytest.MonkeyPatch, result: Any) -> SimpleNamespace:
    """Replace the OpenAI class. Its create and parse calls record their kwargs and return `result`."""
    recorded = SimpleNamespace(inits=[], calls=[])

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            recorded.inits.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._call, parse=self._call))

        def _call(self, **kwargs: Any) -> Any:
            recorded.calls.append(kwargs)
            return result

    monkeypatch.setattr(openrouter_client, "OpenAI", FakeOpenAI)
    return recorded


def _chunk(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_model_for_prefers_task_override_then_general_model_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "general/model")
    monkeypatch.setenv("OPENROUTER_MODEL_QUESTIONS_JUDGE", "judge/model")
    monkeypatch.delenv("OPENROUTER_MODEL_PERSONA", raising=False)
    assert model_for("questions_judge") == "judge/model"
    assert model_for("persona") == "general/model"

    monkeypatch.delenv("OPENROUTER_MODEL")
    assert model_for("questions_judge") == "judge/model"
    with pytest.raises(KeyError):
        model_for("persona")


def test_text_with_on_token_concatenates_deltas_and_skips_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_chunk("Hel"), SimpleNamespace(choices=[]), _chunk(None), _chunk("lo"), _chunk(" world")]
    recorded = _patch_openai(monkeypatch, chunks)
    client = OpenRouterClient("persona")
    tokens: list[str] = []

    text = client.text("sys", "user", web_search=True, on_token=tokens.append)

    assert text == "Hello world"
    assert tokens == ["Hel", "lo", " world"]
    assert recorded.calls[0]["stream"] is True
    assert recorded.calls[0]["tools"] == [openrouter_client.WEB_SEARCH_TOOL]


def test_text_without_callback_and_parse_return_the_model_output(
    monkeypatch: pytest.MonkeyPatch, sample_assumptions: AssumptionsOutput
) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Plain answer", parsed=sample_assumptions))]
    )
    recorded = _patch_openai(monkeypatch, completion)

    client = OpenRouterClient("assumptions")
    assert recorded.inits == [{"base_url": BASE_URL, "api_key": "offline-test-key"}]
    assert client.model == "offline/test-model"

    assert client.text("sys", "user") == "Plain answer"
    assert "stream" not in recorded.calls[0]
    assert "tools" not in recorded.calls[0]
    assert recorded.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]

    assert client.parse("sys", "user", AssumptionsOutput) == sample_assumptions
    assert recorded.calls[1]["response_format"] is AssumptionsOutput


# Explicit counts: an oversized model output is trimmed and reindexed


def test_personas_trim_an_oversized_cast_to_the_requested_count_and_reindex(
    monkeypatch: pytest.MonkeyPatch, sample_request: PipelineRequest, sample_assumptions: AssumptionsOutput
) -> None:
    returned = PersonasOutput(
        personas=[make_persona(f"P{n}", name=f"Persona {n}") for n in (7, 8, 9, 10, 11)]
    )
    _use_fake(monkeypatch, personas_stage, FakeOpenRouterClient(returned))
    state = _with_assumptions(sample_request, sample_assumptions, persona_count=2)
    events, report = _reporter("personas")

    result = personas_stage.PersonasStage().run(state, report)

    assert [p.id for p in result.personas.personas] == ["P1", "P2"]
    assert [p.name for p in result.personas.personas] == ["Persona 7", "Persona 8"]
    trims = [e for e in events if e.get("data", {}).get("returned") == 5]
    assert len(trims) == 1
    assert trims[0]["data"]["kept"] == 2
    assert "keeping the requested 2" in trims[0]["message"]


def test_questions_trim_an_oversized_generator_output_to_the_requested_count(
    monkeypatch: pytest.MonkeyPatch, sample_request: PipelineRequest, sample_assumptions: AssumptionsOutput
) -> None:
    returned = QuestionsOutput(questions=[make_question(f"Q{n}", text=f"Question {n}") for n in range(4, 10)])
    fake = _use_fake(
        monkeypatch,
        questions_stage,
        FakeOpenRouterClient(returned, JudgeFeedback(suggestions=[])),
    )
    state = _with_assumptions(sample_request, sample_assumptions, question_count=3)
    events, report = _reporter("questions")

    result = questions_stage.QuestionsStage().run(state, report)

    assert [q.id for q in result.questions.questions] == ["Q1", "Q2", "Q3"]
    assert [q.text for q in result.questions.questions] == ["Question 4", "Question 5", "Question 6"]
    assert fake.tasks == ["questions_generator", "questions_judge"]
    trims = [e for e in events if e.get("data", {}).get("returned") == 6]
    assert len(trims) == 1
    assert trims[0]["data"]["kept"] == 3
    assert "keeping the requested 3" in trims[0]["message"]


def test_assumptions_trim_an_oversized_output_to_the_requested_count(
    monkeypatch: pytest.MonkeyPatch, sample_request: PipelineRequest
) -> None:
    returned = AssumptionsOutput(assumptions=[make_assumption(f"A{n}") for n in range(7, 12)])
    _use_fake(monkeypatch, assumptions_stage, FakeOpenRouterClient(returned))
    state = PipelineState(request=sample_request.model_copy(update={"assumption_count": 2}))
    events, report = _reporter("assumptions")

    result = assumptions_stage.AssumptionsStage().run(state, report)

    assert [a.id for a in result.assumptions.assumptions] == ["A1", "A2"]
    trims = [e for e in events if e.get("data", {}).get("returned") == 5]
    assert len(trims) == 1
    assert trims[0]["data"]["kept"] == 2
    assert "keeping the requested 2" in trims[0]["message"]
