from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager
from queue import Queue
from threading import RLock, Thread
from uuid import UUID, uuid4

from agent_backend.events import Reporter
from agent_backend.stages.base import Stage
from api_common.assets.common import PipelineState


def run_stage(stage: Stage, state: PipelineState, messages: Queue, job_id: UUID, token: UUID) -> PipelineState:
    """Runs in the stage's process. What the stage reports travels to the main process through the queue."""
    return stage.run(state, Reporter(stage.name, lambda event: messages.put(("event", job_id, token, event))))


class Pipeline:
    """A chain of stages. Each stage runs in its own process pool and its output is the input of the next stage.

    The stage processes report through one queue, and the end of a stage goes through the same queue. One thread
    reads it and makes the callbacks, so the events of a stage always arrive before its completion.
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
        self.pools = [ProcessPoolExecutor(max_workers=workers_per_stage) for _ in stages]
        self.manager = Manager()
        self.messages = self.manager.Queue()
        self.futures: dict[UUID, Future] = {}  # the stage each job is currently in
        self.tokens: dict[UUID, UUID] = {}  # tells that stage run apart from an earlier, stopped one
        self.lock = RLock()
        Thread(target=self._dispatch, daemon=True).start()

    def submit(self, job_id: UUID, state: PipelineState, start: int = 0) -> None:
        stage = self.stages[start]
        token = uuid4()
        with self.lock:
            self.on_event(job_id, {"kind": "stage_started", "stage": stage.name, "message": f"Starting {stage.name}…"})
            self.tokens[job_id] = token
            future = self.pools[start].submit(run_stage, stage, state, self.messages, job_id, token)
            self.futures[job_id] = future
        future.add_done_callback(lambda _: self._send_done(job_id, token, start))

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
        with self.lock:
            self.futures.clear()
            self.tokens.clear()
        self.manager.shutdown()  # ends the dispatch thread

    def _send_done(self, job_id: UUID, token: UUID, index: int) -> None:
        with self.lock:
            if self.tokens.get(job_id) == token:  # otherwise the job was stopped while this stage was running
                self.messages.put(("done", job_id, token, index))

    def _dispatch(self) -> None:
        while True:
            try:
                kind, job_id, token, payload = self.messages.get()
            except (EOFError, OSError):  # the manager is shut down
                return
            if kind == "done":
                self._stage_done(job_id, token, payload)
            elif self.tokens.get(job_id) == token:
                self.on_event(job_id, payload)

    def _stage_done(self, job_id: UUID, token: UUID, index: int) -> None:
        stage = self.stages[index]
        with self.lock:
            if self.tokens.get(job_id) != token:  # stopped after the stage finished
                return
            del self.tokens[job_id]
            try:
                state = self.futures.pop(job_id).result()
            except Exception as error:
                self.on_event(job_id, {"kind": "stage_failed", "stage": stage.name, "message": str(error)})
                self.on_error(job_id, error)
                return
            self.on_stage_done(job_id, state)
            self.on_event(job_id, {"kind": "stage_completed", "stage": stage.name, "message": f"Finished {stage.name}"})
            if index + 1 < len(self.stages):
                self.submit(job_id, state, index + 1)
