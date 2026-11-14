from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from threading import RLock
from uuid import UUID

from agent_backend.events import Reporter
from agent_backend.stages.base import Stage
from api_common.assets.common import PipelineState


class Pipeline:
    """A chain of stages. Each stage runs in its own thread pool and its output is the input of the next stage.

    Stages are I/O-bound: almost all of their time is spent waiting for OpenRouter. Threads keep the reporter
    callbacks in-process, so the WebSocket can stream status and token events while the stage is still running.
    """

    def __init__(
        self,
        stages: list[Stage],
        on_stage_done: Callable[[UUID, PipelineState], None],
        on_error: Callable[[UUID, Exception], None],
        on_event: Callable[[UUID, dict], None] | None = None,
        workers_per_stage: int = 2,
    ):
        self.stages = stages
        self.on_stage_done = on_stage_done
        self.on_error = on_error
        self.on_event = on_event or (lambda _job_id, _event: None)
        self.pools = [ThreadPoolExecutor(max_workers=workers_per_stage) for _ in stages]
        self.futures: dict[UUID, Future] = {}  # the stage each job is currently in
        self.tokens: dict[UUID, object] = {}  # invalidates a stopped stage's late reporter events
        self.lock = RLock()

    def submit(self, job_id: UUID, state: PipelineState, start: int = 0) -> None:
        stage = self.stages[start]
        token = object()
        with self.lock:
            self.on_event(
                job_id,
                {"kind": "stage_started", "stage": stage.name, "message": f"Starting {stage.name}…"},
            )
            self.tokens[job_id] = token
            reporter = Reporter(
                stage.name,
                lambda event: self._stage_event(job_id, token, event),
            )
            future = self.pools[start].submit(stage.run, state, reporter)
            self.futures[job_id] = future
        future.add_done_callback(partial(self._stage_done, job_id, start, token))

    def _stage_event(self, job_id: UUID, token: object, event: dict) -> None:
        if self.tokens.get(job_id) is token:
            self.on_event(job_id, event)

    def stop(self, job_id: UUID) -> None:
        """Hold a job: the output of the stage it is in gets dropped and nothing is sent to the next stage."""
        with self.lock:
            future = self.futures.pop(job_id, None)
            self.tokens.pop(job_id, None)
        if future:
            future.cancel()

    def shutdown(self) -> None:
        for pool in self.pools:
            pool.shutdown(wait=False, cancel_futures=True)

    def _stage_done(self, job_id: UUID, index: int, token: object, future: Future) -> None:
        with self.lock:
            if self.futures.get(job_id) is not future or self.tokens.get(job_id) is not token:
                return  # the job was stopped while this stage was running
            del self.futures[job_id]
            self.tokens.pop(job_id, None)
            try:
                state = future.result()
            except Exception as error:
                self.on_event(
                    job_id,
                    {
                        "kind": "stage_failed",
                        "stage": self.stages[index].name,
                        "message": str(error),
                    },
                )
                self.on_error(job_id, error)
                return
            self.on_stage_done(job_id, state)
            self.on_event(
                job_id,
                {
                    "kind": "stage_completed",
                    "stage": self.stages[index].name,
                    "message": f"Finished {self.stages[index].name}",
                },
            )
            if index + 1 < len(self.stages):
                self.submit(job_id, state, index + 1)
