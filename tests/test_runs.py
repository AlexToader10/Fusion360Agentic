"""RunManager: the run lifecycle (start, stop, edit, resume) as a state machine, then end to end on real process pools."""

import itertools
from uuid import uuid4

import pytest
from pydantic import ValidationError

import agent_backend.runs as runs
from agent_backend.runs import STAGE_NAMES, InvalidAction, RunManager, RunNotFound
from api_common.assets.common import RunStatus
from tests import factories
from tests.fakes import wait_until


class FakePipeline:
    """Records what RunManager asks of the pipeline; the test plays the pipeline's callbacks itself."""

    def __init__(self, stages, on_stage_done, on_error, **_):
        self.stages = stages
        self.on_stage_done = on_stage_done
        self.on_error = on_error
        self.submits: list[tuple] = []
        self.stops: list = []

    def submit(self, job_id, state, start=0):
        self.submits.append((job_id, state.model_copy(deep=True), start))

    def stop(self, job_id):
        self.stops.append(job_id)

    def shutdown(self):
        pass

    def finish(self, job_id, stage: str, tag="idea"):
        """Simulate the stage finishing: the worker returns a copy of the state with the stage filled in."""
        _, state, _ = self.submits[-1]
        state = state.model_copy(deep=True)
        setattr(state, stage, factories.output(stage, tag))
        self.on_stage_done(job_id, state)
        self.submits.append((job_id, state, STAGE_NAMES.index(stage) + 1))


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(runs, "Pipeline", FakePipeline)
    return RunManager()


def run_with(manager: RunManager, last_stage: str | None, status: RunStatus):
    """Put a run directly in a given state, as if the pipeline had brought it there."""
    run = manager.start(factories.request())
    run.state = factories.state_through(last_stage)
    run.status = status
    return run


# ---- start / get ------------------------------------------------------------------------------------------------


def test_start_registers_a_running_run_and_submits_it_from_the_first_stage(manager):
    run = manager.start(factories.request("bakery", persona_count=4))

    assert run.status == RunStatus.RUNNING and run.error is None
    assert manager.get(run.id) is run
    [(job, state, start)] = manager.pipeline.submits
    assert (job, start) == (run.id, 0)
    assert state.request.persona_count == 4
    assert all(getattr(state, stage) is None for stage in STAGE_NAMES)


def test_every_run_gets_its_own_id(manager):
    ids = {manager.start(factories.request(f"idea{i}")).id for i in range(50)}
    assert len(ids) == 50 and set(manager.runs) == ids


def test_get_unknown_run_raises(manager):
    unknown = uuid4()
    with pytest.raises(RunNotFound, match=str(unknown)):
        manager.get(unknown)


def test_run_completes_only_when_the_last_stage_reports(manager):
    run = manager.start(factories.request())
    for stage in STAGE_NAMES[:-1]:
        manager.pipeline.finish(run.id, stage)
        assert run.status == RunStatus.RUNNING
    manager.pipeline.finish(run.id, STAGE_NAMES[-1])
    assert run.status == RunStatus.COMPLETED
    assert run.state.conclusions is not None


def test_a_pipeline_error_marks_the_run_failed_with_the_message(manager):
    run = manager.start(factories.request())
    manager.pipeline.on_error(run.id, ValueError("rate limited"))
    assert (run.status, run.error) == (RunStatus.FAILED, "rate limited")


# ---- the transition table ---------------------------------------------------------------------------------------

ALLOWED = {
    # (action, status) -> allowed?
    ("stop", RunStatus.RUNNING): True,
    ("stop", RunStatus.STOPPED): False,
    ("stop", RunStatus.COMPLETED): False,
    ("stop", RunStatus.FAILED): False,
    ("edit", RunStatus.RUNNING): False,
    ("edit", RunStatus.STOPPED): True,
    ("edit", RunStatus.COMPLETED): True,
    ("edit", RunStatus.FAILED): True,
    ("resume", RunStatus.RUNNING): False,
    ("resume", RunStatus.STOPPED): True,
    ("resume", RunStatus.COMPLETED): True,
    ("resume", RunStatus.FAILED): True,
}


@pytest.mark.parametrize(("action", "status"), list(ALLOWED))
def test_transition_table(manager, action, status):
    run = run_with(manager, "personas", status)
    before = run.model_copy(deep=True)
    call = {
        "stop": lambda: manager.stop(run.id),
        "edit": lambda: manager.edit(run.id, "assumptions", factories.assumptions(2, tag="new")),
        "resume": lambda: manager.resume(run.id),
    }[action]

    if ALLOWED[(action, status)]:
        call()
        assert run.status in {RunStatus.STOPPED, RunStatus.RUNNING}
    else:
        with pytest.raises(InvalidAction):
            call()
        assert run == before, "a refused action must leave the run untouched"
        assert manager.pipeline.stops == [] and len(manager.pipeline.submits) == 1


@pytest.mark.parametrize("action", ["stop", "edit", "resume"])
def test_every_action_on_an_unknown_run_raises_not_found(manager, action):
    args = {"stop": (), "edit": ("assumptions", {}), "resume": ()}[action]
    with pytest.raises(RunNotFound):
        getattr(manager, action)(uuid4(), *args)


def test_stop_holds_the_run_in_the_pipeline(manager):
    run = manager.start(factories.request())
    manager.stop(run.id)
    assert run.status == RunStatus.STOPPED
    assert manager.pipeline.stops == [run.id]


# ---- edit -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("edited", STAGE_NAMES)
def test_editing_a_stage_clears_exactly_the_stages_after_it(manager, edited):
    run = run_with(manager, STAGE_NAMES[-1], RunStatus.COMPLETED)
    new_output = factories.output(edited, tag="edited")

    manager.edit(run.id, edited, new_output)

    position = STAGE_NAMES.index(edited)
    for stage in STAGE_NAMES[:position]:
        assert getattr(run.state, stage) == factories.output(stage, "idea"), f"{stage} must be kept"
    assert getattr(run.state, edited) == new_output
    for stage in STAGE_NAMES[position + 1 :]:
        assert getattr(run.state, stage) is None, f"{stage} must be cleared"
    assert run.status == RunStatus.STOPPED  # a completed run that was edited is no longer complete


def test_edit_accepts_a_plain_dict_and_validates_it_into_the_stage_model(manager):
    """The edit agent sends JSON objects, not models; PipelineState(validate_assignment=True) converts them."""
    run = run_with(manager, "questions", RunStatus.STOPPED)
    raw = factories.personas(4, tag="raw").model_dump()

    manager.edit(run.id, "personas", raw)

    assert type(run.state.personas).__name__ == "PersonasOutput"
    assert [persona.id for persona in run.state.personas.personas] == ["P1", "P2", "P3", "P4"]
    assert run.state.questions is None


def test_an_invalid_edit_is_rejected_and_leaves_every_stage_untouched(manager):
    run = run_with(manager, "interviews", RunStatus.STOPPED)
    before = run.model_copy(deep=True)
    bad = factories.personas().model_dump()
    bad["personas"][0]["age"] = "not a number"

    with pytest.raises(ValidationError):
        manager.edit(run.id, "personas", bad)

    assert run == before


def test_a_stage_output_from_another_stage_is_rejected(manager):
    run = run_with(manager, "questions", RunStatus.STOPPED)
    with pytest.raises(ValidationError):
        manager.edit(run.id, "personas", factories.questions())


@pytest.mark.parametrize("stage", ["questions", "conclusions", "nonsense", "request", ""])
def test_only_finished_stages_can_be_edited(manager, stage):
    run = run_with(manager, "personas", RunStatus.STOPPED)
    with pytest.raises(InvalidAction, match="no output to edit"):
        manager.edit(run.id, stage, {})


def test_edit_does_not_resubmit_anything(manager):
    run = run_with(manager, "questions", RunStatus.STOPPED)
    manager.edit(run.id, "assumptions", factories.assumptions(5))
    assert len(manager.pipeline.submits) == 1  # only the start


# ---- resume -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("last_stage", [None, *STAGE_NAMES[:-1]])
def test_resume_continues_from_the_first_stage_without_output(manager, last_stage):
    run = run_with(manager, last_stage, RunStatus.STOPPED)
    manager.resume(run.id)

    expected_start = 0 if last_stage is None else STAGE_NAMES.index(last_stage) + 1
    job, state, start = manager.pipeline.submits[-1]
    assert (job, start) == (run.id, expected_start)
    assert state == run.state
    assert run.status == RunStatus.RUNNING


def test_resume_of_a_finished_run_marks_it_completed_without_running_anything(manager):
    run = run_with(manager, STAGE_NAMES[-1], RunStatus.STOPPED)
    manager.resume(run.id)
    assert run.status == RunStatus.COMPLETED
    assert len(manager.pipeline.submits) == 1


def test_resume_after_a_failure_clears_the_error(manager):
    run = manager.start(factories.request())
    manager.pipeline.finish(run.id, "assumptions")
    manager.pipeline.on_error(run.id, RuntimeError("timeout"))

    manager.resume(run.id)

    assert (run.status, run.error) == (RunStatus.RUNNING, None)
    assert manager.pipeline.submits[-1][2] == STAGE_NAMES.index("personas")


def test_a_long_random_session_keeps_the_run_consistent(manager):
    """Drive one run through a long scripted sequence of actions and stage completions, checking invariants each step:
    finished stages always form a prefix of the pipeline, and the status agrees with what is finished."""
    run = manager.start(factories.request())
    script = itertools.cycle(
        ["finish", "finish", "stop", "edit:assumptions", "resume", "finish", "finish", "finish", "stop",
         "edit:personas", "resume", "finish", "finish", "finish", "finish", "finish", "edit:questions", "resume", "fail",
         "resume", "finish", "finish", "finish"]
    )  # fmt: skip
    for step in itertools.islice(script, 200):
        if step == "finish" and run.status == RunStatus.RUNNING:
            pending = [stage for stage in STAGE_NAMES if getattr(run.state, stage) is None]
            manager.pipeline.finish(run.id, pending[0])
        elif step == "fail" and run.status == RunStatus.RUNNING:
            manager.pipeline.on_error(run.id, RuntimeError("flaky"))
        elif step == "stop" and run.status == RunStatus.RUNNING:
            manager.stop(run.id)
        elif step.startswith("edit:") and run.status != RunStatus.RUNNING:
            stage = step.removeprefix("edit:")
            if getattr(run.state, stage) is not None:
                manager.edit(run.id, stage, factories.output(stage, "edited"))
        elif step == "resume" and run.status != RunStatus.RUNNING:
            manager.resume(run.id)

        done = [getattr(run.state, stage) is not None for stage in STAGE_NAMES]
        assert done == sorted(done, reverse=True), f"finished stages are not a prefix after {step}: {done}"
        if run.status == RunStatus.COMPLETED:
            assert all(done)
        if run.status == RunStatus.FAILED:
            assert run.error == "flaky"
        else:
            assert run.error is None


# ---- end to end on the real Pipeline ----------------------------------------------------------------------------


def test_live_run_completes(live_manager):
    run = live_manager.start(factories.request("live"))
    wait_until(lambda: run.status == RunStatus.COMPLETED, message="completion")
    assert run.state.conclusions.summary == "summary live#0"


def test_live_stop_edit_resume_reruns_only_the_cleared_stages(live_manager, control):
    control.gate("interviews", "flow")
    run = live_manager.start(factories.request("flow"))
    wait_until(lambda: control.started("interviews", "flow"), message="interviews to start")

    live_manager.stop(run.id)
    control.open("interviews", "flow")  # the held interviews finish in the worker; their output must be dropped
    wait_until(lambda: control.runs_of("interviews", "flow") == 1, message="the held stage to finish")
    assert run.status == RunStatus.STOPPED and run.state.interviews is None

    edited = factories.personas(5, tag="edited")
    live_manager.edit(run.id, "personas", edited)
    assert run.state.questions is None

    live_manager.resume(run.id)
    wait_until(lambda: run.status == RunStatus.COMPLETED, message="the resumed run to complete")

    assert run.state.personas == edited  # the edit survived the rest of the pipeline
    assert run.state.questions.questions[0].text.endswith("flow#1")  # questions ran a second time
    assert [control.runs_of(stage, "flow") for stage in STAGE_NAMES] == [1, 1, 2, 2, 1, 1]


def test_live_failure_then_resume(live_manager, control):
    control.fail("validations", "flaky", "provider returned 502")
    run = live_manager.start(factories.request("flaky"))
    wait_until(lambda: run.status == RunStatus.FAILED, message="the failure")
    assert run.error == "provider returned 502"
    assert run.state.interviews is not None and run.state.validations is None

    control.heal("validations", "flaky")
    live_manager.resume(run.id)
    wait_until(lambda: run.status == RunStatus.COMPLETED, message="completion after resume")
    assert run.error is None
    assert control.runs_of("interviews", "flaky") == 1


def test_live_parallel_runs_do_not_interfere(live_manager, control):
    control.gate("questions", "slow")
    slow = live_manager.start(factories.request("slow"))
    fast = [live_manager.start(factories.request(f"fast{i}")) for i in range(4)]

    wait_until(lambda: all(run.status == RunStatus.COMPLETED for run in fast), timeout=60, message="the fast runs")
    assert slow.status == RunStatus.RUNNING and slow.state.questions is None

    control.open("questions", "slow")
    wait_until(lambda: slow.status == RunStatus.COMPLETED, message="the slow run")
    for run in [slow, *fast]:
        tag = run.state.request.business_idea
        assert run.state.conclusions.summary == f"summary {tag}#0"
