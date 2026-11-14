"""Test doubles.

FileControlledStage runs inside the real ProcessPoolExecutor workers, so it is defined at module level (picklable) and
is steered from the test process through marker files in a shared directory, one set per job tag
(the job tag is the run's business_idea):

    <stage>-<tag>.gate     the stage blocks until <stage>-<tag>.go (or release-all) exists
    <stage>-<tag>.fail     the stage raises RuntimeError with the file's text
    <stage>-<tag>.crash    the worker process dies (os._exit), which breaks the pool

The stage always writes <stage>-<tag>.started and appends "<tag> <stage> <pid>" to log.txt when it finishes.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agent_backend.stages.base import Stage
from api_common.assets.common import STAGE_OUTPUTS, PipelineState
from tests import factories

GATE_TIMEOUT = 30


class FileControlledStage(Stage):
    def __init__(self, name: str, workdir: str):
        self.name = name
        self.workdir = workdir

    def run(self, state: PipelineState) -> PipelineState:
        tag = state.request.business_idea
        root = Path(self.workdir)
        prefix = root / f"{self.name}-{tag}"
        Path(f"{prefix}.started").touch()

        stages = list(STAGE_OUTPUTS)
        for earlier in stages[: stages.index(self.name)]:  # every earlier stage must have handed over its output
            if getattr(state, earlier) is None:
                raise RuntimeError(f"{self.name} received no {earlier}")
        if getattr(state, self.name) is not None:
            raise RuntimeError(f"{self.name} received a state that already has its output")

        if Path(f"{prefix}.gate").exists():
            deadline = time.monotonic() + GATE_TIMEOUT
            while not (Path(f"{prefix}.go").exists() or (root / "release-all").exists()):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.name} gate for {tag} never opened")
                time.sleep(0.01)
        if Path(f"{prefix}.crash").exists():
            os._exit(13)
        if Path(f"{prefix}.fail").exists():
            raise RuntimeError(Path(f"{prefix}.fail").read_text())

        # each run of a stage leaves a distinct fingerprint, so a test can tell which execution produced an output
        attempt = len(list(root.glob(f"{self.name}-{tag}.done*")))
        Path(f"{prefix}.done{attempt}").touch()
        setattr(state, self.name, factories.output(self.name, tag=f"{tag}#{attempt}"))
        with open(root / "log.txt", "a") as log:
            log.write(f"{tag} {self.name} {os.getpid()}\n")
        return state


class StageControl:
    """Test-side handle on the marker files of FileControlledStage."""

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def _marker(self, stage: str, tag: str, kind: str) -> Path:
        return self.workdir / f"{stage}-{tag}.{kind}"

    def gate(self, stage: str, tag: str) -> None:
        self._marker(stage, tag, "gate").touch()

    def open(self, stage: str, tag: str) -> None:
        self._marker(stage, tag, "go").touch()

    def fail(self, stage: str, tag: str, message: str = "boom") -> None:
        self._marker(stage, tag, "fail").write_text(message)

    def heal(self, stage: str, tag: str) -> None:
        self._marker(stage, tag, "fail").unlink(missing_ok=True)

    def crash(self, stage: str, tag: str) -> None:
        self._marker(stage, tag, "crash").touch()

    def started(self, stage: str, tag: str) -> bool:
        return self._marker(stage, tag, "started").exists()

    def runs_of(self, stage: str, tag: str) -> int:
        return len(list(self.workdir.glob(f"{stage}-{tag}.done*")))

    def release_all(self) -> None:
        (self.workdir / "release-all").touch()

    def log(self) -> list[tuple[str, str, int]]:
        path = self.workdir / "log.txt"
        if not path.exists():
            return []
        return [(tag, stage, int(pid)) for tag, stage, pid in (line.split() for line in path.read_text().splitlines())]


def wait_until(predicate: Callable[[], bool], timeout: float = 20, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        time.sleep(0.01)


class ScriptedClient:
    """Stands in for OpenRouterClient. Each task gets a handler (system, prompt, extra) -> result; calls are recorded.

    Thread-safe, because the interviews and validations stages call it from a thread pool."""

    handlers: dict[str, Callable] = {}
    calls: list[dict] = []
    lock = threading.Lock()

    def __init__(self, task: str):
        self.task = task
        self.model = f"model-for-{task}"

    @classmethod
    def reset(cls, **handlers: Callable) -> type["ScriptedClient"]:
        cls.handlers = handlers
        cls.calls = []
        return cls

    @classmethod
    def calls_for(cls, task: str) -> list[dict]:
        with cls.lock:
            return [call for call in cls.calls if call["task"] == task]

    def _call(self, kind: str, system: str, prompt: str, **extra):
        with self.lock:
            self.calls.append({"task": self.task, "kind": kind, "system": system, "prompt": prompt, **extra})
        if self.task not in self.handlers:
            raise AssertionError(f"unexpected call for task {self.task}")
        return self.handlers[self.task](system, prompt, extra)

    def text(self, system: str, prompt: str, web_search: bool = False) -> str:
        return self._call("text", system, prompt, web_search=web_search)

    def parse(self, system: str, prompt: str, output_model):
        result = self._call("parse", system, prompt, output_model=output_model)
        assert isinstance(result, output_model), f"handler for {self.task} returned {type(result).__name__}"
        return result
