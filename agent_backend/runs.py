from uuid import UUID, uuid4

from pydantic import BaseModel

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
        self.pipeline = Pipeline(STAGES, self._stage_done, self._failed)

    def start(self, request: PipelineRequest) -> Run:
        run = Run(id=uuid4(), state=PipelineState(request=request))
        self.runs[run.id] = run
        self.pipeline.submit(run.id, run.state)
        return run

    def get(self, run_id: UUID) -> Run:
        if run_id not in self.runs:
            raise RunNotFound(f"Run {run_id} not found")
        return self.runs[run_id]

    def stop(self, run_id: UUID) -> Run:
        run = self.get(run_id)
        if run.status != RunStatus.RUNNING:
            raise InvalidAction(f"Run is {run.status}, not running")
        self.pipeline.stop(run_id)
        run.status = RunStatus.STOPPED
        return run

    def edit(self, run_id: UUID, stage: str, output: BaseModel | dict) -> Run:
        """Replace the output of a finished stage. Later stages are cleared and run again on resume."""
        run = self.get(run_id)
        if run.status == RunStatus.RUNNING:
            raise InvalidAction("Stop the run before editing it")
        if stage not in STAGE_NAMES or getattr(run.state, stage) is None:
            raise InvalidAction(f"Stage '{stage}' has no output to edit")
        setattr(run.state, stage, output)
        for later in STAGE_NAMES[STAGE_NAMES.index(stage) + 1 :]:
            setattr(run.state, later, None)
        run.status = RunStatus.STOPPED
        return run

    def resume(self, run_id: UUID) -> Run:
        """Continue from the first stage without an output."""
        run = self.get(run_id)
        if run.status == RunStatus.RUNNING:
            raise InvalidAction("Run is already running")
        pending = [index for index, name in enumerate(STAGE_NAMES) if getattr(run.state, name) is None]
        if not pending:
            run.status = RunStatus.COMPLETED
            return run
        run.status, run.error = RunStatus.RUNNING, None
        self.pipeline.submit(run_id, run.state, start=pending[0])
        return run

    def _stage_done(self, run_id: UUID, state: PipelineState) -> None:
        run = self.runs[run_id]
        run.state = state
        if getattr(state, STAGE_NAMES[-1]) is not None:
            run.status = RunStatus.COMPLETED

    def _failed(self, run_id: UUID, error: Exception) -> None:
        run = self.runs[run_id]
        run.status, run.error = RunStatus.FAILED, str(error)
