"""Offline state-machine tests for the thread-based Pipeline, RunManager, EventLog and Reporter.

No network and no models: fake stages and a fake pipeline stand in for the real ones. Callbacks are waited on with a
condition variable or threading.Event, not sleeps.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from uuid import UUID, uuid4

import pytest

from agent_backend.events import EventLog, Reporter
from agent_backend.pipeline import Pipeline
from agent_backend.runs import InvalidAction, RunManager, RunNotFound
from api_common.assets.common import PipelineRequest, PipelineState, RunStatus
from api_common.assets.personas import PersonasOutput

WAIT = 5.0


def _request() -> PipelineRequest:
    return PipelineRequest(business_idea="idea", target_audience="audience")


class _FakeStage:
    """Duck-typed stage. Records each call, appends its name to the business idea, and can report, block or raise."""

    def __init__(
        self,
        name: str,
        *,
        block: bool = False,
        error: Exception | None = None,
        script: Callable[[Reporter], None] | None = None,
    ) -> None:
        self.name = name
        self.calls: list[PipelineState] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._block = block
        self._error = error
        self._script = script

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        self.calls.append(state)
        self.started.set()
        if self._block:
            self.release.wait(WAIT)
        if self._script is not None and report is not None:
            self._script(report)
        if self._error is not None:
            raise self._error
        request = state.request.model_copy(update={"business_idea": f"{state.request.business_idea} > {self.name}"})
        return PipelineState(request=request)


class _Recorder:
    """Collects Pipeline callbacks and lets a test wait until a condition holds."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.done: list[tuple[UUID, PipelineState]] = []
        self.errors: list[tuple[UUID, Exception]] = []
        self.events: list[tuple[UUID, dict]] = []

    def on_stage_done(self, job_id: UUID, state: PipelineState) -> None:
        self._append(self.done, (job_id, state))

    def on_error(self, job_id: UUID, error: Exception) -> None:
        self._append(self.errors, (job_id, error))

    def on_event(self, job_id: UUID, event: dict) -> None:
        self._append(self.events, (job_id, event))

    def _append(self, target: list, item: object) -> None:
        with self._cond:
            target.append(item)
            self._cond.notify_all()

    def wait_until(self, predicate: Callable[[_Recorder], bool]) -> None:
        with self._cond:
            ok = self._cond.wait_for(lambda: predicate(self), WAIT)
        assert ok, "timed out waiting for pipeline callbacks"

    def kinds(self, job_id: UUID) -> list[str]:
        with self._cond:
            return [event["kind"] for job, event in self.events if job == job_id]


class _FakePipeline:
    """Stands in for Pipeline inside RunManager. Records calls and runs nothing."""

    def __init__(self) -> None:
        self.submitted: list[tuple[UUID, PipelineState, int]] = []
        self.stopped: list[UUID] = []

    def submit(self, job_id: UUID, state: PipelineState, start: int = 0) -> None:
        self.submitted.append((job_id, state, start))

    def stop(self, job_id: UUID) -> None:
        self.stopped.append(job_id)

    def shutdown(self) -> None:
        pass


def _manager() -> RunManager:
    """A RunManager whose real thread pool is shut down and replaced by a fake, so no threads leak."""
    manager = RunManager()
    manager.pipeline.shutdown()
    manager.pipeline = _FakePipeline()
    return manager


def _kinds(manager: RunManager, run_id: UUID) -> list[str]:
    return [event["kind"] for event in manager.events(run_id)]


@pytest.fixture
def build_pipeline() -> Iterator[Callable[[list[_FakeStage]], tuple[Pipeline, _Recorder]]]:
    pipelines: list[Pipeline] = []

    def build(stages: list[_FakeStage]) -> tuple[Pipeline, _Recorder]:
        rec = _Recorder()
        pipeline = Pipeline(stages, rec.on_stage_done, rec.on_error, rec.on_event)
        pipelines.append(pipeline)
        return pipeline, rec

    yield build
    for pipeline in pipelines:
        pipeline.shutdown()


def test_pipeline_runs_stages_in_order_once_each(build_pipeline):
    """Each stage runs exactly once, receives the previous stage's output, and on_stage_done sees states in order."""
    stages = [_FakeStage("a"), _FakeStage("b"), _FakeStage("c")]
    pipeline, rec = build_pipeline(stages)
    job = uuid4()

    pipeline.submit(job, PipelineState(request=_request()))
    rec.wait_until(lambda r: len(r.done) == 3)

    assert [len(stage.calls) for stage in stages] == [1, 1, 1]
    assert stages[1].calls[0].request.business_idea == "idea > a"
    assert stages[2].calls[0].request.business_idea == "idea > a > b"
    assert all(job_id == job for job_id, _ in rec.done)
    assert [state.request.business_idea for _, state in rec.done] == [
        "idea > a",
        "idea > a > b",
        "idea > a > b > c",
    ]
    assert rec.errors == []


def test_pipeline_emits_stage_and_report_events_in_order(build_pipeline):
    """stage_started, status, token and stage_completed arrive in that order, with the stage name and report data."""

    def script(report: Reporter) -> None:
        report.status("loading model", step=1)
        report.token("Hel")

    pipeline, rec = build_pipeline([_FakeStage("fake", script=script)])
    job = uuid4()

    pipeline.submit(job, PipelineState(request=_request()))
    rec.wait_until(lambda r: "stage_completed" in r.kinds(job))

    events = [event for job_id, event in rec.events if job_id == job]
    assert [event["kind"] for event in events] == ["stage_started", "status", "token", "stage_completed"]
    assert all(event["stage"] == "fake" for event in events)
    assert events[1] == {"kind": "status", "stage": "fake", "message": "loading model", "data": {"step": 1}}
    assert events[2] == {"kind": "token", "stage": "fake", "message": "Hel"}


def test_failing_stage_calls_on_error_and_never_starts_next_stage(build_pipeline):
    """A raising stage sends its exception to on_error, emits stage_failed, and the next stage never starts."""
    error = RuntimeError("model timed out")
    failing = _FakeStage("failing", error=error)
    following = _FakeStage("next")
    pipeline, rec = build_pipeline([failing, following])
    job = uuid4()

    pipeline.submit(job, PipelineState(request=_request()))
    rec.wait_until(lambda r: bool(r.errors))

    assert len(rec.errors) == 1
    error_job, delivered = rec.errors[0]
    assert error_job == job and delivered is error
    assert rec.done == []
    assert following.calls == [] and not following.started.is_set()
    assert rec.kinds(job) == ["stage_started", "stage_failed"]
    failed = [event for _, event in rec.events if event["kind"] == "stage_failed"]
    assert failed == [{"kind": "stage_failed", "stage": "failing", "message": "model timed out"}]


def test_stop_drops_late_result_and_shutdown_does_not_hang(build_pipeline):
    """Stopping a running job drops its late result: no callbacks and no next stage. Shutdown and a second stop return."""
    slow = _FakeStage("slow", block=True)
    following = _FakeStage("next")
    pipeline, rec = build_pipeline([slow, following])
    job = uuid4()

    pipeline.submit(job, PipelineState(request=_request()))
    assert slow.started.wait(WAIT)
    finished = threading.Event()
    pipeline.futures[job].add_done_callback(lambda _future: finished.set())
    pipeline.stop(job)

    shutdown_thread = threading.Thread(target=pipeline.shutdown, daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(WAIT)
    assert not shutdown_thread.is_alive(), "Pipeline.shutdown hung while a stage was running"

    slow.release.set()
    assert finished.wait(WAIT)
    assert rec.done == [] and rec.errors == []
    assert following.calls == [] and not following.started.is_set()
    assert rec.kinds(job) == ["stage_started"]
    assert job not in pipeline.futures

    pipeline.stop(job)
    pipeline.stop(uuid4())


class _ChattyStage:
    """Reports before and after a blocking event, so a test can stop the job while it is still running."""

    name = "chatty"

    def __init__(self) -> None:
        self.reported = threading.Event()
        self.release = threading.Event()

    def run(self, state: PipelineState, report: Reporter | None = None) -> PipelineState:
        report.status(f"{state.request.business_idea}: before")
        report.token("pre")
        self.reported.set()
        self.release.wait(WAIT)
        report.status(f"{state.request.business_idea}: after")
        report.token("post")
        return PipelineState(request=state.request)


def test_stopped_stage_drops_late_events_and_done_callback_and_new_job_gets_its_own(build_pipeline):
    """After stop, the blocked stage's late status/token and its completion are dropped. A new job still gets its events."""
    chatty = _ChattyStage()
    following = _FakeStage("next")
    pipeline, rec = build_pipeline([chatty, following])
    first = uuid4()

    pipeline.submit(first, PipelineState(request=PipelineRequest(business_idea="job-one", target_audience="a")))
    assert chatty.reported.wait(WAIT)
    future = pipeline.futures[first]
    finished = threading.Event()
    future.add_done_callback(lambda _future: finished.set())
    pipeline.stop(first)

    chatty.release.set()
    assert finished.wait(WAIT)
    assert rec.kinds(first) == ["stage_started", "status", "token"]
    assert [event["message"] for job_id, event in rec.events if job_id == first] == [
        "Starting chatty…",
        "job-one: before",
        "pre",
    ]
    assert rec.done == [] and rec.errors == []
    assert following.calls == [] and not following.started.is_set()

    second = uuid4()
    pipeline.submit(second, PipelineState(request=PipelineRequest(business_idea="job-two", target_audience="a")))
    rec.wait_until(lambda r: len([job for job, _ in r.done if job == second]) == 2)

    assert rec.kinds(second) == [
        "stage_started",
        "status",
        "token",
        "status",
        "token",
        "stage_completed",
        "stage_started",
        "stage_completed",
    ]
    assert [event["message"] for job_id, event in rec.events if job_id == second][1:5] == [
        "job-two: before",
        "pre",
        "job-two: after",
        "post",
    ]
    assert [state.request.business_idea for job_id, state in rec.done if job_id == second] == [
        "job-two",
        "job-two > next",
    ]
    assert [call.request.business_idea for call in following.calls] == ["job-two"]
    assert rec.kinds(first) == ["stage_started", "status", "token"]


def test_run_manager_start_creates_running_run_and_submits_stage_zero(sample_request):
    """start returns a RUNNING run holding the request, submits at stage 0 and emits run_started."""
    manager = _manager()

    run = manager.start(sample_request)

    assert run.status == RunStatus.RUNNING
    assert run.state.request == sample_request
    assert run.state.assumptions is None
    assert manager.get(run.id) is run
    assert len(manager.pipeline.submitted) == 1
    job, state, start = manager.pipeline.submitted[0]
    assert job == run.id and state is run.state and start == 0
    assert _kinds(manager, run.id) == ["run_started"]
    with pytest.raises(RunNotFound):
        manager.events(uuid4())
    with pytest.raises(RunNotFound):
        manager.last_event_seq(uuid4())


def test_run_manager_stop_only_works_while_running(sample_request, full_state):
    """stop moves a RUNNING run to STOPPED and emits run_stopped. Stopping a stopped or completed run raises."""
    manager = _manager()
    run = manager.start(sample_request)

    assert manager.stop(run.id) is run
    assert run.status == RunStatus.STOPPED
    assert manager.pipeline.stopped == [run.id]
    assert _kinds(manager, run.id)[-1] == "run_stopped"
    with pytest.raises(InvalidAction):
        manager.stop(run.id)

    completed = manager.start(sample_request)
    manager._stage_done(completed.id, full_state)
    assert completed.status == RunStatus.COMPLETED
    with pytest.raises(InvalidAction):
        manager.stop(completed.id)
    assert manager.pipeline.stopped == [run.id]


def test_run_manager_edit_refuses_running_unknown_and_empty_stages(sample_request, sample_assumptions, sample_personas):
    """edit refuses a RUNNING run, an unknown stage name and a stage that has no output yet. The state is unchanged."""
    manager = _manager()
    run = manager.start(sample_request)
    with pytest.raises(InvalidAction, match="Stop the run"):
        manager.edit(run.id, "assumptions", sample_assumptions)

    partial = PipelineState(request=sample_request, assumptions=sample_assumptions, personas=sample_personas)
    manager._stage_done(run.id, partial)
    manager.stop(run.id)
    for stage in ("questions", "request", "not_a_stage"):
        with pytest.raises(InvalidAction):
            manager.edit(run.id, stage, sample_assumptions)

    assert run.state.questions is None
    assert run.state.assumptions == sample_assumptions
    assert "stage_edited" not in _kinds(manager, run.id)


def test_run_manager_edit_replaces_output_and_clears_later_stages(sample_request, full_state, sample_personas):
    """edit swaps in the new output (model or dict), clears every later stage, sets STOPPED and emits stage_edited."""
    manager = _manager()
    run = manager.start(sample_request)
    manager._stage_done(run.id, full_state)
    assert run.status == RunStatus.COMPLETED

    smaller = PersonasOutput(personas=sample_personas.personas[:1])
    assert manager.edit(run.id, "personas", smaller) is run

    assert run.state.personas == smaller
    assert run.state.assumptions == full_state.assumptions
    for later in ("questions", "interviews", "validations", "conclusions"):
        assert getattr(run.state, later) is None
    assert run.status == RunStatus.STOPPED
    edited = manager.events(run.id)[-1]
    assert edited["kind"] == "stage_edited"
    assert edited["data"] == {"stage": "personas", "cleared": ["questions", "interviews", "validations", "conclusions"]}

    manager.edit(run.id, "assumptions", full_state.assumptions.model_dump())
    assert run.state.assumptions == full_state.assumptions
    assert run.state.personas is None
    assert manager.events(run.id)[-1]["data"]["cleared"] == ["personas", "questions", "interviews", "validations", "conclusions"]


def test_run_manager_resume_continues_from_first_missing_stage_and_guards(
    sample_request, sample_assumptions, sample_personas, full_state
):
    """resume refuses a RUNNING run, restarts at the first missing stage and clears the old error. A complete run stays COMPLETED."""
    manager = _manager()
    run = manager.start(sample_request)
    with pytest.raises(InvalidAction, match="already running"):
        manager.resume(run.id)

    partial = PipelineState(request=sample_request, assumptions=sample_assumptions, personas=sample_personas)
    manager._stage_done(run.id, partial)
    manager._failed(run.id, RuntimeError("model timed out"))
    assert run.status == RunStatus.FAILED and run.error == "model timed out"

    assert manager.resume(run.id) is run
    assert run.status == RunStatus.RUNNING and run.error is None
    job, state, start = manager.pipeline.submitted[-1]
    assert job == run.id and state is run.state and start == 2
    assert _kinds(manager, run.id) == ["run_started", "run_failed", "run_resumed"]
    assert manager.events(run.id)[-1]["message"] == "Resuming at questions"

    done = manager.start(sample_request)
    manager._stage_done(done.id, full_state)
    submitted_before = len(manager.pipeline.submitted)
    manager.resume(done.id)
    assert done.status == RunStatus.COMPLETED
    assert len(manager.pipeline.submitted) == submitted_before
    assert manager.events(done.id)[-1]["kind"] == "run_completed"


def test_run_manager_completes_only_when_last_stage_is_filled_and_records_failures(sample_request, full_state):
    """_stage_done marks COMPLETED only when conclusions is filled. _failed records FAILED, the error text and run_failed."""
    manager = _manager()
    run = manager.start(sample_request)
    almost = full_state.model_copy(update={"conclusions": None})

    manager._stage_done(run.id, almost)
    assert run.status == RunStatus.RUNNING
    assert run.state is almost
    assert "run_completed" not in _kinds(manager, run.id)

    manager._stage_done(run.id, full_state)
    assert run.status == RunStatus.COMPLETED
    assert _kinds(manager, run.id)[-1] == "run_completed"

    failing = manager.start(sample_request)
    manager._failed(failing.id, ValueError("bad output"))
    assert failing.status == RunStatus.FAILED
    assert failing.error == "bad output"
    last = manager.events(failing.id)[-1]
    assert last["kind"] == "run_failed" and last["message"] == "bad output"


def test_event_log_stamps_monotonic_seq_filters_after_and_drops_oldest():
    """seq grows by one per add, after(seq) returns only newer events, and only the newest max_events are kept."""
    log = EventLog(max_events=3)

    stamped = [log.add({"kind": "status", "n": n}) for n in range(5)]

    assert [event["seq"] for event in stamped] == [1, 2, 3, 4, 5]
    assert all("at" in event for event in stamped)
    assert log.last_seq == 5
    assert [event["seq"] for event in log.after(0)] == [3, 4, 5]
    assert [event["n"] for event in log.after(3)] == [3, 4]
    assert log.after(5) == []


def test_event_log_concurrent_adds_keep_unique_monotonic_seq():
    """Adds from several threads at once give every event a unique seq, in order, and keep each thread's own order."""
    log = EventLog()
    workers, per_worker = 8, 250
    barrier = threading.Barrier(workers)

    def work(worker: int) -> None:
        barrier.wait(WAIT)
        for i in range(per_worker):
            log.add({"kind": "status", "worker": worker, "i": i})

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads)

    events = log.after(0)
    assert [event["seq"] for event in events] == list(range(1, workers * per_worker + 1))
    assert log.last_seq == workers * per_worker
    for worker in range(workers):
        assert [event["i"] for event in events if event["worker"] == worker] == list(range(per_worker))


def test_reporter_status_and_token_payloads_and_child_merge():
    """status and token emit kind, stage, message and data. A child reporter merges its constant fields into every event."""
    sent: list[dict] = []
    reporter = Reporter("personas", sent.append)

    reporter.status("Drafting", step=2)
    reporter.status("Plain")
    reporter.token("Hel")
    assert sent[0] == {"kind": "status", "stage": "personas", "message": "Drafting", "data": {"step": 2}}
    assert sent[1] == {"kind": "status", "stage": "personas", "message": "Plain"}
    assert sent[2] == {"kind": "token", "stage": "personas", "message": "Hel"}

    child = reporter.child(persona_id="P1")
    child.status("Asking", turn=1)
    grandchild = child.child(question_id="Q3")
    grandchild.token("x")
    assert sent[3] == {"kind": "status", "stage": "personas", "message": "Asking", "data": {"persona_id": "P1", "turn": 1}}
    assert sent[4] == {
        "kind": "token",
        "stage": "personas",
        "message": "x",
        "data": {"persona_id": "P1", "question_id": "Q3"},
    }


def test_noop_reporter_accepts_every_call_silently():
    """Reporter.noop accepts status, token and child calls without raising and returns None."""
    reporter = Reporter.noop("assumptions")

    assert reporter.status("working", step=1) is None
    assert reporter.token("Hel") is None
    assert reporter.child(persona_id="P1").status("still working") is None
