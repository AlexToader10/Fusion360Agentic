from threading import RLock
from uuid import UUID, uuid4

from pydantic import BaseModel

from agent_backend.events import EventLog
from agent_backend.pipeline import Pipeline
from agent_backend.stages import STAGES
from api_common.assets.common import PipelineRequest, PipelineState, Run, RunStatus

STAGE_NAMES = [stage.name for stage in STAGES]


class RunNotFound(Exception):
    pass


class InvalidAction(Exception):
    pass


class RunManager:
    """Owns the runs and the pipeline. Its methods are what the HTTP paths and the edit agent both call."""

    def __init__(self):
        self.runs: dict[UUID, Run] = {}
        self.event_logs: dict[UUID, EventLog] = {}
        self._events_lock = RLock()
        self.pipeline = Pipeline(STAGES, self._stage_done, self._failed, self._event)

    def start(self, request: PipelineRequest) -> Run:
        run = Run(id=uuid4(), state=PipelineState(request=request))
        self.runs[run.id] = run
        self.event_logs[run.id] = EventLog()
        self._event(run.id, {"kind": "run_started", "message": "Validation run started"})
        self.pipeline.submit(run.id, run.state)
        return run

    def get(self, run_id: UUID) -> Run:
        if run_id not in self.runs:
            raise RunNotFound(f"Run {run_id} not found")
        return self.runs[run_id]

    def events(self, run_id: UUID, after: int = 0) -> list[dict]:
        self.get(run_id)
        with self._events_lock:
            log = self.event_logs[run_id]
        return log.after(after)

    def last_event_seq(self, run_id: UUID) -> int:
        self.get(run_id)
        with self._events_lock:
            return self.event_logs[run_id].last_seq

    def stop(self, run_id: UUID) -> Run:
        run = self.get(run_id)
        if run.status != RunStatus.RUNNING:
            raise InvalidAction(f"Run is {run.status}, not running")
        self.pipeline.stop(run_id)
        run.status = RunStatus.STOPPED
        self._event(run_id, {"kind": "run_stopped", "message": "Run stopped. Edit any finished stage, then resume."})
        return run

    def edit(self, run_id: UUID, stage: str, output: BaseModel | dict) -> Run:
        """Replace the output of a finished stage. Later stages are cleared and run again on resume."""
        run = self.get(run_id)
        if run.status == RunStatus.RUNNING:
            raise InvalidAction("Stop the run before editing it")
        if stage not in STAGE_NAMES or getattr(run.state, stage) is None:
            raise InvalidAction(f"Stage '{stage}' has no output to edit")
        setattr(run.state, stage, output)
        cleared = STAGE_NAMES[STAGE_NAMES.index(stage) + 1 :]
        for later in cleared:
            setattr(run.state, later, None)
        run.status = RunStatus.STOPPED
        self._event(
            run_id,
            {
                "kind": "stage_edited",
                "stage": stage,
                "message": f"Edited {stage}; cleared {len(cleared)} later stage(s)",
                "data": {"stage": stage, "cleared": cleared},
            },
        )
        return run

    def resume(self, run_id: UUID) -> Run:
        """Continue from the first stage without an output."""
        run = self.get(run_id)
        if run.status == RunStatus.RUNNING:
            raise InvalidAction("Run is already running")
        pending = [index for index, name in enumerate(STAGE_NAMES) if getattr(run.state, name) is None]
        if not pending:
            run.status = RunStatus.COMPLETED
            self._event(run_id, {"kind": "run_completed", "message": "Run was already complete"})
            return run
        run.status, run.error = RunStatus.RUNNING, None
        self._event(run_id, {"kind": "run_resumed", "message": f"Resuming at {STAGE_NAMES[pending[0]]}"})
        self.pipeline.submit(run_id, run.state, start=pending[0])
        return run

    def _event(self, run_id: UUID, event: dict) -> None:
        with self._events_lock:
            log = self.event_logs.setdefault(run_id, EventLog())
        log.add(event)

    def _stage_done(self, run_id: UUID, state: PipelineState) -> None:
        run = self.runs[run_id]
        run.state = state
        if getattr(state, STAGE_NAMES[-1]) is not None:
            run.status = RunStatus.COMPLETED
            self._event(run_id, {"kind": "run_completed", "message": "Validation run completed"})

    def _failed(self, run_id: UUID, error: Exception) -> None:
        run = self.runs[run_id]
        run.status, run.error = RunStatus.FAILED, str(error)
        self._event(run_id, {"kind": "run_failed", "message": str(error)})
