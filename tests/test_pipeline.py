"""The Pipeline engine with real process pools: chaining, stop, stale results, failures and many concurrent jobs."""

import threading
from uuid import UUID, uuid4

import pytest

from api_common.assets.common import PipelineState
from tests import factories
from tests.factories import STAGE_NAMES
from tests.fakes import wait_until


class Recorder:
    """Collects the Pipeline callbacks, which arrive on the executors' management threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.done: list[tuple[UUID, PipelineState]] = []
        self.errors: list[tuple[UUID, Exception]] = []

    def on_stage_done(self, job_id, state):
        with self.lock:
            self.done.append((job_id, state))

    def on_error(self, job_id, error):
        with self.lock:
            self.errors.append((job_id, error))

    def done_for(self, job_id) -> list[PipelineState]:
        with self.lock:
            return [state for job, state in self.done if job == job_id]

    def errors_for(self, job_id) -> list[Exception]:
        with self.lock:
            return [error for job, error in self.errors if job == job_id]


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def pipeline(make_pipeline, fake_stages, recorder):
    return make_pipeline(fake_stages, recorder.on_stage_done, recorder.on_error)


def finished_stages(state: PipelineState) -> list[str]:
    return [name for name in STAGE_NAMES if getattr(state, name) is not None]


def test_job_runs_every_stage_in_order_and_hands_each_output_forward(pipeline, recorder, control):
    job = uuid4()
    pipeline.submit(job, factories.state_through(None, "solo"))

    wait_until(lambda: len(recorder.done_for(job)) == len(STAGE_NAMES), message="all stages")

    states = recorder.done_for(job)
    # after the n-th callback exactly the first n stages have an output: nothing skipped, nothing reordered
    assert [finished_stages(state) for state in states] == [STAGE_NAMES[: n + 1] for n in range(len(STAGE_NAMES))]
    # each stage ran exactly once, and the final state holds what each of those single executions produced
    assert [stage for _, stage, _ in control.log()] == STAGE_NAMES
    final = states[-1]
    assert final.conclusions.summary == "summary solo#0"
    assert final.request.business_idea == "solo"
    assert recorder.errors == []
    assert pipeline.futures == {}  # a finished job leaves nothing behind


def test_submit_with_a_start_index_skips_the_earlier_stages(pipeline, recorder, control):
    job = uuid4()
    pipeline.submit(job, factories.state_through("questions", "mid"), start=STAGE_NAMES.index("interviews"))

    wait_until(lambda: len(recorder.done_for(job)) == 3, message="the three last stages")
    assert [stage for _, stage, _ in control.log()] == ["interviews", "validations", "conclusions"]
    assert recorder.done_for(job)[-1].assumptions.assumptions[0].statement == "statement 1 mid"  # untouched input


def test_stages_run_in_separate_worker_processes(pipeline, recorder, control):
    import os

    job = uuid4()
    pipeline.submit(job, factories.state_through(None, "pids"))
    wait_until(lambda: len(recorder.done_for(job)) == len(STAGE_NAMES), message="all stages")

    pids = [pid for _, _, pid in control.log()]
    assert os.getpid() not in pids
    assert len(set(pids)) == len(STAGE_NAMES)  # one pool per stage, so no two stages share a worker


def test_stopping_a_job_drops_the_output_of_the_running_stage(pipeline, recorder, control):
    job = uuid4()
    control.gate("personas", "held")
    pipeline.submit(job, factories.state_through(None, "held"))
    wait_until(lambda: control.started("personas", "held"), message="personas to start")

    pipeline.stop(job)
    control.open("personas", "held")
    wait_until(lambda: control.runs_of("personas", "held") == 1, message="the worker to finish anyway")

    # the worker finished, but its result never reaches the callback and the next stage never starts
    with pytest.raises(AssertionError):
        wait_until(lambda: control.started("questions", "held"), timeout=1)
    assert [finished_stages(state) for state in recorder.done_for(job)] == [["assumptions"]]
    assert recorder.errors == []
    assert job not in pipeline.futures


def test_stopping_a_job_that_is_queued_behind_busy_workers_cancels_it_before_it_starts(
    make_pipeline, fake_stages, recorder, control
):
    pipeline = make_pipeline(fake_stages, recorder.on_stage_done, recorder.on_error, workers_per_stage=1)
    blocker, queued = uuid4(), uuid4()
    control.gate("assumptions", "blocker")
    pipeline.submit(blocker, factories.state_through(None, "blocker"))
    wait_until(lambda: control.started("assumptions", "blocker"), message="blocker to occupy the only worker")
    pipeline.submit(queued, factories.state_through(None, "queued"))

    pipeline.stop(queued)
    control.open("assumptions", "blocker")
    wait_until(lambda: len(recorder.done_for(blocker)) == len(STAGE_NAMES), message="blocker to finish")

    # ProcessPoolExecutor may already have handed the queued call to the worker's queue, in which case cancel() fails
    # and it still runs; either way its result must be dropped and the job must not move to the next stage.
    with pytest.raises(AssertionError):
        wait_until(lambda: control.started("personas", "queued"), timeout=1)
    assert recorder.done_for(queued) == [] and recorder.errors_for(queued) == []


def test_a_stale_result_from_a_stopped_stage_cannot_overwrite_a_resubmitted_job(pipeline, recorder, control):
    """Stop while a stage runs, resubmit the same job from that stage, then let the old execution finish last.

    The old execution's callback must be ignored: only the new execution's output may move the job forward."""
    job = uuid4()
    control.gate("personas", "race")
    pipeline.submit(job, factories.state_through(None, "race"))
    wait_until(lambda: control.started("personas", "race"), message="first personas run")

    pipeline.stop(job)
    edited = recorder.done_for(job)[0].model_copy(deep=True)
    edited.assumptions.assumptions[0].statement = "edited while held"
    pipeline.submit(job, edited, start=STAGE_NAMES.index("personas"))
    # both executions are now blocked on the same gate; opening it lets them race to finish
    control.open("personas", "race")

    wait_until(lambda: len(recorder.done_for(job)) == len(STAGE_NAMES), message="the resubmitted job to finish")
    wait_until(lambda: control.runs_of("personas", "race") == 2, message="the stale execution to finish too")

    states = recorder.done_for(job)
    assert [finished_stages(state) for state in states] == [STAGE_NAMES[: n + 1] for n in range(len(STAGE_NAMES))]
    assert all(state.assumptions.assumptions[0].statement == "edited while held" for state in states[1:])
    assert [stage for _, stage, _ in control.log()].count("questions") == 1  # the stale result started nothing
    assert recorder.errors == []


def test_a_failing_stage_reports_the_error_and_the_job_goes_no_further(pipeline, recorder, control):
    job = uuid4()
    control.fail("questions", "bad", "the judge exploded")
    pipeline.submit(job, factories.state_through(None, "bad"))

    wait_until(lambda: recorder.errors_for(job), message="the error callback")
    [error] = recorder.errors_for(job)
    assert isinstance(error, RuntimeError) and str(error) == "the judge exploded"
    assert [finished_stages(state) for state in recorder.done_for(job)] == [["assumptions"], ["assumptions", "personas"]]
    assert not control.started("interviews", "bad")
    assert job not in pipeline.futures


def test_a_failed_job_can_be_resubmitted_from_the_failed_stage(pipeline, recorder, control):
    job = uuid4()
    control.fail("validations", "retry")
    pipeline.submit(job, factories.state_through(None, "retry"))
    wait_until(lambda: recorder.errors_for(job), message="the failure")

    control.heal("validations", "retry")
    last_good = recorder.done_for(job)[-1]
    pipeline.submit(job, last_good, start=STAGE_NAMES.index("validations"))

    wait_until(lambda: recorder.done_for(job)[-1].conclusions is not None, message="the retried job to finish")
    assert len(recorder.errors_for(job)) == 1
    assert control.runs_of("interviews", "retry") == 1  # earlier stages were not redone


def test_many_concurrent_jobs_finish_independently_with_their_own_data(make_pipeline, fake_stages, recorder, control):
    pipeline = make_pipeline(fake_stages, recorder.on_stage_done, recorder.on_error, workers_per_stage=3)
    jobs = {uuid4(): f"job{i}" for i in range(16)}
    for i, (job, tag) in enumerate(jobs.items()):
        if i % 4 == 1:
            control.fail(STAGE_NAMES[i % len(STAGE_NAMES)], tag, f"failure of {tag}")
    for job, tag in jobs.items():
        pipeline.submit(job, factories.state_through(None, tag))

    failing = {job for i, job in enumerate(jobs) if i % 4 == 1}
    wait_until(
        lambda: all(recorder.errors_for(job) for job in failing)
        and all(len(recorder.done_for(job)) == len(STAGE_NAMES) for job in jobs.keys() - failing),
        timeout=60,
        message="every job to finish or fail",
    )

    for i, (job, tag) in enumerate(jobs.items()):
        states = recorder.done_for(job)
        assert all(state.request.business_idea == tag for state in states), "a job received another job's state"
        if job in failing:
            failed_stage = STAGE_NAMES[i % len(STAGE_NAMES)]
            assert [str(e) for e in recorder.errors_for(job)] == [f"failure of {tag}"]
            assert len(states) == STAGE_NAMES.index(failed_stage)
        else:
            assert recorder.errors_for(job) == []
            assert states[-1].conclusions.summary == f"summary {tag}#0"
    assert pipeline.futures == {}


def test_stopping_an_unknown_or_finished_job_is_a_no_op(pipeline, recorder):
    pipeline.stop(uuid4())
    job = uuid4()
    pipeline.submit(job, factories.state_through("validations", "end"), start=len(STAGE_NAMES) - 1)
    wait_until(lambda: recorder.done_for(job), message="the last stage")
    pipeline.stop(job)
    assert len(recorder.done_for(job)) == 1


def test_a_crashed_worker_fails_the_job_instead_of_hanging_it(pipeline, recorder, control):
    job = uuid4()
    control.crash("personas", "crash")
    pipeline.submit(job, factories.state_through(None, "crash"))

    wait_until(lambda: recorder.errors_for(job), message="the crash to be reported")
    from concurrent.futures.process import BrokenProcessPool

    assert isinstance(recorder.errors_for(job)[0], BrokenProcessPool)


def test_the_pipeline_recovers_after_a_worker_crash(pipeline, recorder, control):
    crashed, later = uuid4(), uuid4()
    control.crash("assumptions", "crash")
    pipeline.submit(crashed, factories.state_through(None, "crash"))
    wait_until(lambda: recorder.errors_for(crashed), message="the crash")

    pipeline.submit(later, factories.state_through(None, "healthy"))
    wait_until(lambda: len(recorder.done_for(later)) == len(STAGE_NAMES), timeout=10, message="a healthy job")
