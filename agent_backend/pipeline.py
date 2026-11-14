import multiprocessing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from threading import RLock
from uuid import UUID

from agent_backend.stages.base import Stage
from api_common.assets.common import PipelineState

# Before Python 3.14 the Linux default is "fork", which copies the locks other threads (uvicorn, httpx) hold into the
# worker and can deadlock it. "forkserver" is the 3.14 default and safe with threads.
MP_CONTEXT = multiprocessing.get_context("forkserver")


class Pipeline:
    """A chain of stages. Each stage runs in its own process pool and its output is the input of the next stage."""

    def __init__(
        self,
        stages: list[Stage],
        on_stage_done: Callable[[UUID, PipelineState], None],
        on_error: Callable[[UUID, Exception], None],
        workers_per_stage: int = 2,
    ):
        self.stages = stages
        self.on_stage_done = on_stage_done
        self.on_error = on_error
        self.workers_per_stage = workers_per_stage
        self.pools = [self._new_pool() for _ in stages]
        self.futures: dict[UUID, Future] = {}  # the stage each job is currently in
        self.lock = RLock()

    def submit(self, job_id: UUID, state: PipelineState, start: int = 0) -> None:
        with self.lock:
            try:
                future = self.pools[start].submit(self.stages[start].run, state)
            except BrokenProcessPool:
                # a worker of this stage died earlier, which breaks its pool for good: replace the pool and retry
                self.pools[start].shutdown(wait=False, cancel_futures=True)
                self.pools[start] = self._new_pool()
                future = self.pools[start].submit(self.stages[start].run, state)
            self.futures[job_id] = future
        future.add_done_callback(partial(self._stage_done, job_id, start))

    def stop(self, job_id: UUID) -> None:
        """Hold a job: the output of the stage it is in gets dropped and nothing is sent to the next stage."""
        with self.lock:
            future = self.futures.pop(job_id, None)
        if future:
            future.cancel()

    def _new_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=self.workers_per_stage, mp_context=MP_CONTEXT)

    def shutdown(self) -> None:
        for pool in self.pools:
            pool.shutdown(wait=False, cancel_futures=True)

    def _stage_done(self, job_id: UUID, index: int, future: Future) -> None:
        with self.lock:
            if self.futures.get(job_id) is not future:  # the job was stopped while this stage was running
                return
            del self.futures[job_id]
            try:
                state = future.result()
            except Exception as error:
                self.on_error(job_id, error)
                return
            self.on_stage_done(job_id, state)
            if index + 1 < len(self.stages):
                self.submit(job_id, state, index + 1)
