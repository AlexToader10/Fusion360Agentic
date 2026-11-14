"""Every LLM stage, with OpenRouterClient replaced by a scripted fake: what each stage sends, and how it reacts to
what the model returns (judge loop, interviewer picks, parallel interviews and validations)."""

import json
import re
import threading
import time

import pytest

from agent_backend.stages import STAGES
from agent_backend.stages.assumptions import AssumptionsStage
from agent_backend.stages.base import business_context
from agent_backend.stages.conclusions import ConclusionsStage
from agent_backend.stages.deep_research import DeepResearchHarness
from agent_backend.stages.interviews import InterviewsStage
from agent_backend.stages.personas import DEFAULT_COUNT, PersonasStage
from agent_backend.stages.questions import MAX_JUDGE_ROUNDS, QuestionsStage
from agent_backend.stages.validations import ValidationsStage
from api_common.assets.common import STAGE_OUTPUTS, PipelineState
from api_common.assets.interviews import QuestionPick
from api_common.assets.questions import JudgeFeedback
from api_common.assets.validations import AssumptionVerdict, InterviewValidation
from tests import factories


def returns(value):
    return lambda system, prompt, extra: value


def persona_id_in(prompt: str) -> str:
    return json.loads(prompt.split("Persona:\n", 1)[1].split("\n\n", 1)[0])["id"]


# ---- registry and shared context --------------------------------------------------------------------------------


def test_stage_list_matches_the_state_fields_in_order():
    assert [stage.name for stage in STAGES] == list(STAGE_OUTPUTS)
    assert all(stage.name in PipelineState.model_fields for stage in STAGES)


def test_business_context_keeps_the_given_parts_in_order_and_drops_empty_ones():
    request = factories.request("a bakery", additional_info="", business_plan="Plan text")
    assert business_context(request) == (
        "Business idea:\na bakery\n\nTarget audience:\naudience of a bakery\n\nBusiness plan:\nPlan text"
    )


def test_business_context_from_a_plan_only_request():
    request = factories.request("", target_audience="", business_plan="only a plan")
    assert business_context(request) == "Business plan:\nonly a plan"


# ---- single-call stages -----------------------------------------------------------------------------------------


def test_assumptions_stage(scripted):
    output = factories.assumptions(7)
    scripted.reset(assumptions=returns(output))
    state = factories.state_through(None, "drone delivery", additional_info="we own 3 drones")

    result = AssumptionsStage().run(state)

    assert result.assumptions == output
    [call] = scripted.calls
    assert call["output_model"].__name__ == "AssumptionsOutput"
    assert "drone delivery" in call["prompt"] and "we own 3 drones" in call["prompt"]
    assert "riskiest" in call["system"]


@pytest.mark.parametrize(("persona_count", "expected"), [(None, DEFAULT_COUNT), (1, "1"), (30, "30")])
def test_personas_stage_asks_for_the_requested_sample_size(scripted, persona_count, expected):
    scripted.reset(personas=returns(factories.personas(3)))
    state = factories.state_through("assumptions", persona_count=persona_count)

    PersonasStage().run(state)

    [call] = scripted.calls
    assert f"create {expected} distinct" in call["system"]
    assert '"A3"' in call["prompt"]  # the assumptions travel in the prompt


def test_conclusions_stage_sees_assumptions_and_every_verdict(scripted):
    scripted.reset(conclusions=returns(factories.conclusions()))
    state = factories.state_through("validations")

    ConclusionsStage().run(state)

    prompt = scripted.calls[0]["prompt"]
    assert state.assumptions.model_dump_json(indent=2) in prompt
    assert state.validations.model_dump_json(indent=2) in prompt
    assert state.conclusions == factories.conclusions()


# ---- questions: generator / judge loop --------------------------------------------------------------------------


def script_questions(scripted, judge_rounds: list[list[str]]):
    """Generator returns a new version each call; the judge returns the given suggestion lists one after the other
    (then 'no suggestions' forever)."""
    versions = iter(range(100))
    verdicts = iter(judge_rounds)
    scripted.reset(
        questions_generator=lambda s, p, e: factories.questions(tag=f"v{next(versions)}"),
        questions_judge=lambda s, p, e: JudgeFeedback(suggestions=next(verdicts, [])),
    )


def test_questions_accepted_first_time(scripted):
    script_questions(scripted, [[]])
    state = QuestionsStage().run(factories.state_through("personas"))

    assert len(scripted.calls_for("questions_generator")) == 1
    assert len(scripted.calls_for("questions_judge")) == 1
    assert state.questions == factories.questions(tag="v0")


def test_questions_are_regenerated_with_the_judge_suggestions_until_neutral(scripted):
    script_questions(scripted, [["Q1 presupposes the problem"], ["Q2 pitches the product", "Q3 is yes/no"], []])
    state = QuestionsStage().run(factories.state_through("personas"))

    generator_calls = scripted.calls_for("questions_generator")
    judge_calls = scripted.calls_for("questions_judge")
    assert (len(generator_calls), len(judge_calls)) == (3, 3)
    assert state.questions == factories.questions(tag="v2")

    first_edit, second_edit = generator_calls[1]["prompt"], generator_calls[2]["prompt"]
    assert "- Q1 presupposes the problem" in first_edit and "v0" in first_edit  # the judged version is shown back
    assert "- Q2 pitches the product\n- Q3 is yes/no" in second_edit and "v1" in second_edit
    assert "Q1 presupposes" not in second_edit  # old suggestions do not pile up
    # the judge reviews the exact version the generator produced
    assert [json.loads(call["prompt"])["questions"][0]["text"][-2:] for call in judge_calls] == ["v0", "v1", "v2"]


def test_a_judge_that_never_agrees_stops_after_the_round_limit(scripted):
    script_questions(scripted, [["still leading"]] * 50)
    state = QuestionsStage().run(factories.state_through("personas"))

    assert len(scripted.calls_for("questions_judge")) == MAX_JUDGE_ROUNDS
    assert len(scripted.calls_for("questions_generator")) == MAX_JUDGE_ROUNDS + 1
    assert state.questions == factories.questions(tag=f"v{MAX_JUDGE_ROUNDS}")  # the last rewrite is kept unjudged


# ---- interviews -------------------------------------------------------------------------------------------------


def script_interviews(scripted, plans: dict[str, list[str | None]], delay: float = 0.0):
    """plans: persona id -> the question ids the interviewer picks, in order. Answers are deterministic."""
    cursors = {persona: iter(picks) for persona, picks in plans.items()}
    lock = threading.Lock()

    def interviewer(system, prompt, extra):
        persona = persona_id_in(prompt)
        with lock:
            pick = next(cursors[persona], None)
        time.sleep(delay)
        return QuestionPick(question_id=pick)

    def persona(system, prompt, extra):
        if extra["web_search"]:
            return f"research about {re.search(r'"name": "([^"]+)"', prompt).group(1)}"
        name = re.search(r"You are (.+?), \d+", system).group(1)
        asked = prompt.rsplit("Interviewer: ", 1)[1]
        return f"{name} answers '{asked}'"

    scripted.reset(interviewer=interviewer, persona=persona)


def test_each_persona_is_interviewed_on_the_questions_the_interviewer_picks(scripted):
    script_interviews(scripted, {"P1": ["Q2", "Q1", None], "P2": ["Q3"]})
    state = InterviewsStage().run(factories.state_through("questions", tag=""))

    p1, p2 = state.interviews.interviews
    assert (p1.persona_id, p2.persona_id) == ("P1", "P2")
    assert [e.question_id for e in p1.exchanges] == ["Q2", "Q1"]
    assert [e.question_id for e in p2.exchanges] == ["Q3"]
    assert p1.exchanges[0].answer == "Person 1 answers 'Tell me about the last time 2'"
    assert p1.research == "research about Person 1"


def test_an_interview_ends_once_every_question_was_asked_without_asking_the_interviewer_again(scripted):
    script_interviews(scripted, {"P1": ["Q1", "Q2", "Q3"], "P2": ["Q3", "Q2", "Q1"]})
    state = InterviewsStage().run(factories.state_through("questions"))

    assert [len(i.exchanges) for i in state.interviews.interviews] == [3, 3]
    assert len(scripted.calls_for("interviewer")) == 6


def test_the_interviewer_only_ever_sees_questions_not_yet_asked(scripted):
    script_interviews(scripted, {"P1": ["Q2", "Q3", "Q1"], "P2": [None]})
    InterviewsStage().run(factories.state_through("questions"))

    offered = [
        [q["id"] for q in json.loads(call["prompt"].split("Remaining questions:\n", 1)[1])["questions"]]
        for call in scripted.calls_for("interviewer")
        if persona_id_in(call["prompt"]) == "P1"
    ]
    assert offered == [["Q1", "Q2", "Q3"], ["Q1", "Q3"], ["Q1"]]


def test_the_persona_hears_the_whole_conversation_so_far(scripted):
    script_interviews(scripted, {"P1": ["Q1", "Q2", "Q3"], "P2": [None]})
    InterviewsStage().run(factories.state_through("questions", tag=""))

    answers = [c for c in scripted.calls_for("persona") if not c["web_search"] and "Person 1" in c["system"]]
    last_prompt = answers[-1]["prompt"]
    assert last_prompt.count("Interviewer: ") == 3
    assert "You: Person 1 answers 'Tell me about the last time 1'" in last_prompt
    assert last_prompt.endswith("Interviewer: Tell me about the last time 3")
    assert answers[0]["prompt"] == "Interviewer: Tell me about the last time 1"  # no empty transcript prefix
    assert "research about Person 1" in answers[0]["system"]


@pytest.mark.parametrize("bad_pick", ["Q99", "q1", "", "Q1 "])
def test_a_question_id_that_is_not_remaining_ends_the_interview(scripted, bad_pick):
    """The interviewer model can hallucinate an id. The stage treats it like null: the interview ends there."""
    script_interviews(scripted, {"P1": ["Q1", bad_pick, "Q2"], "P2": [None]})
    state = InterviewsStage().run(factories.state_through("questions"))
    assert [e.question_id for e in state.interviews.interviews[0].exchanges] == ["Q1"]


def test_asking_the_same_question_twice_ends_the_interview(scripted):
    script_interviews(scripted, {"P1": ["Q1", "Q1", "Q2"], "P2": [None]})
    state = InterviewsStage().run(factories.state_through("questions"))
    assert [e.question_id for e in state.interviews.interviews[0].exchanges] == ["Q1"]


def test_interviews_run_in_parallel_and_keep_persona_order(scripted):
    count = 8
    state = factories.state_through("questions")
    state.personas = factories.personas(count)
    script_interviews(scripted, {f"P{i}": ["Q1", "Q2", None] for i in range(1, count + 1)}, delay=0.1)

    started = time.monotonic()
    result = InterviewsStage().run(state)
    elapsed = time.monotonic() - started

    # 8 personas x 3 interviewer calls x 0.1 s is 2.4 s sequentially
    assert elapsed < 1.5, f"interviews look sequential ({elapsed:.1f}s)"
    assert [i.persona_id for i in result.interviews.interviews] == [f"P{i}" for i in range(1, count + 1)]
    for index, interview in enumerate(result.interviews.interviews, start=1):
        assert all(e.answer.startswith(f"Person {index} ") for e in interview.exchanges), "answers got mixed up"


def test_a_failing_interview_fails_the_whole_stage(scripted):
    script_interviews(scripted, {"P1": ["Q1"], "P2": ["Q1"]})
    original = scripted.handlers["persona"]

    def persona(system, prompt, extra):
        if "Person 2" in system or "Person 2" in prompt:
            raise ConnectionError("OpenRouter unreachable")
        return original(system, prompt, extra)

    scripted.handlers["persona"] = persona
    with pytest.raises(ConnectionError, match="unreachable"):
        InterviewsStage().run(factories.state_through("questions"))


def test_deep_research_harness_researches_once_with_web_search_and_hides_ids(scripted):
    scripted.reset(persona=lambda s, p, e: "notes" if e["web_search"] else "answer")
    persona = factories.personas(1).personas[0]

    harness = DeepResearchHarness(persona, "Business idea:\nbread")
    harness.answer("q1", [])
    harness.answer("q2", [])

    research_calls = [c for c in scripted.calls if c["web_search"]]
    assert len(research_calls) == 1
    assert '"id"' not in research_calls[0]["prompt"] and "assumption_ids" not in research_calls[0]["prompt"]
    assert "Business idea:\nbread" in research_calls[0]["prompt"]


def test_persona_fields_with_braces_do_not_break_the_prompt_template(scripted):
    scripted.reset(persona=lambda s, p, e: "research with {curly} braces" if e["web_search"] else s)
    persona = factories.personas(1).personas[0].model_copy(update={"background": "writes {json} and ${price}"})

    system = DeepResearchHarness(persona, "biz").answer("q", [])

    assert "writes {json} and ${price}" in system
    assert "research with {curly} braces" in system


# ---- validations ------------------------------------------------------------------------------------------------


def test_each_interview_is_validated_separately_and_order_is_kept(scripted):
    state = factories.state_through("interviews")
    state.interviews = factories.interviews(persona_count=6)

    def validate(system, prompt, extra):
        persona = re.search(r"Interview with persona (P\d+)", prompt).group(1)
        time.sleep(0.05 * (7 - int(persona[1:])))  # later personas finish first
        return InterviewValidation(
            persona_id=persona,
            verdicts=[AssumptionVerdict(assumption_id="A1", verdict="validated", evidence=f"{persona} said so")],
        )

    scripted.reset(validation=validate)
    result = ValidationsStage().run(state)

    assert [v.persona_id for v in result.validations.validations] == [f"P{i}" for i in range(1, 7)]
    for call in scripted.calls_for("validation"):
        persona = re.search(r"Interview with persona (P\d+)", call["prompt"]).group(1)
        assert f"{persona} answer 1" in call["prompt"]
        assert "research on" not in call["prompt"]  # judged only on what was said, not on the research notes
        assert state.assumptions.model_dump_json(indent=2) in call["prompt"]


def test_stage_receives_and_returns_the_same_state_object(scripted):
    """Stages fill in their field on the state they receive; the pipeline relies on the returned state."""
    scripted.reset(conclusions=returns(factories.conclusions()))
    state = factories.state_through("validations")
    assert ConclusionsStage().run(state) is state
