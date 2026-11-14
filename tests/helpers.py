"""Small utilities shared by the test modules. Import them as `from tests.helpers import ...`.

Only `require_live()` needs pytest, and it imports it lazily, so the rest runs without it.
"""

from __future__ import annotations

import importlib
import os
import re
import threading
from collections import Counter, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

from api_common.assets.assumptions import Assumption
from api_common.assets.conclusions import AssumptionConclusion
from api_common.assets.interviews import Exchange, Interview
from api_common.assets.personas import Persona
from api_common.assets.questions import Question
from api_common.assets.validations import AssumptionVerdict, InterviewValidation, Verdict

if TYPE_CHECKING:
    import pytest

M = TypeVar("M", bound=BaseModel)


def _build(model: type[M], **fields: Any) -> M:
    return model.model_validate(fields)


# Factories. Each takes its id positionally and any model field as a keyword override.


def make_assumption(id: str = "A1", **overrides: Any) -> Assumption:
    return _build(
        Assumption,
        **{
            "id": id,
            "statement": f"Assumption {id} holds for the target customers.",
            "category": "problem",
            "rationale": f"The business case depends on {id}.",
            **overrides,
        },
    )


def make_persona(id: str = "P1", assumption_ids: Sequence[str] = ("A1",), **overrides: Any) -> Persona:
    return _build(
        Persona,
        **{
            "id": id,
            "name": f"Persona {id}",
            "age": 35,
            "occupation": "Office manager",
            "background": "Lives in a suburb with two children and a long commute.",
            "goals": "Get dinner on the table without spending the evening planning it.",
            "pain_points": "Plans fall apart by midweek and takeaway costs add up.",
            "assumption_ids": list(assumption_ids),
            **overrides,
        },
    )


def make_question(id: str = "Q1", assumption_ids: Sequence[str] = ("A1",), **overrides: Any) -> Question:
    return _build(
        Question,
        **{
            "id": id,
            "text": "Walk me through the last time this came up for you.",
            "assumption_ids": list(assumption_ids),
            **overrides,
        },
    )


def make_exchange(question_id: str = "Q1", **overrides: Any) -> Exchange:
    return _build(
        Exchange,
        **{
            "question_id": question_id,
            "question": "Walk me through the last time this came up for you.",
            "answer": "Last week I sorted it out myself, mostly by improvising.",
            **overrides,
        },
    )


def make_interview(
    persona_id: str = "P1",
    exchanges: Sequence[Exchange] = (),
    research: str = "No specific research notes.",
    **overrides: Any,
) -> Interview:
    return _build(
        Interview,
        **{
            "persona_id": persona_id,
            "research": research,
            "exchanges": list(exchanges),
            **overrides,
        },
    )


def make_validation(persona_id: str = "P1", verdicts: Mapping[str, Verdict] | None = None) -> InterviewValidation:
    """One verdict per assumption id. Without `verdicts`, the single assumption A1 is validated."""
    chosen = dict(verdicts) if verdicts is not None else {"A1": "validated"}
    return _build(
        InterviewValidation,
        persona_id=persona_id,
        verdicts=[
            AssumptionVerdict(
                assumption_id=assumption_id,
                verdict=verdict,
                evidence=f"{persona_id} said something relevant to {assumption_id}.",
            )
            for assumption_id, verdict in chosen.items()
        ],
    )


def make_conclusion(
    assumption_id: str = "A1",
    verdict: Verdict = "validated",
    reasoning: str | None = None,
) -> AssumptionConclusion:
    return _build(
        AssumptionConclusion,
        assumption_id=assumption_id,
        verdict=verdict,
        reasoning=reasoning or f"Overall verdict for {assumption_id} across the interviews.",
    )


# Id assertions


def assert_unique_ids(items: Iterable[Any], attr: str = "id") -> None:
    ids = [getattr(item, attr) for item in items]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    assert not duplicates, f"duplicate {attr} values: {duplicates}"


def assert_sequential_ids(items: Iterable[Any], prefix: str, attr: str = "id") -> None:
    """Items must carry ids prefix1, prefix2, ... in order, with no gaps."""
    items = list(items)
    expected = [f"{prefix}{number}" for number in range(1, len(items) + 1)]
    actual = [getattr(item, attr) for item in items]
    assert actual == expected, f"expected {attr} sequence {expected}, got {actual}"


# Text metrics


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+")


def sentence_count(text: str) -> int:
    """Splits on . ! ? followed by whitespace. Abbreviations such as 'e.g.' count as breaks."""
    return len([part for part in _SENTENCE_BREAK.split(text.strip()) if part])


def token_set(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def jaccard(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """Jaccard similarity of two collections. Two empty collections score 1.0. Pass token_set(text) for text."""
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def contains_url(text: str) -> bool:
    return _URL.search(text) is not None


def extract_ids(text: str, prefixes: str = "QAP") -> dict[str, list[str]]:
    """Ids such as Q1, A2 or P3 found in free text, in order of appearance and without duplicates.

    The result has one key per prefix letter, e.g. {"Q": ["Q1", "Q3"], "A": ["A2"], "P": []}.
    """
    pattern = re.compile(rf"\b([{re.escape(prefixes)}])(\d+)\b")
    found: dict[str, list[str]] = {letter: [] for letter in prefixes}
    for letter, number in pattern.findall(text):
        ident = f"{letter}{number}"
        if ident not in found[letter]:
            found[letter].append(ident)
    return found


# Fake OpenRouter client


@dataclass(frozen=True)
class FakeCall:
    kind: Literal["text", "parse"]
    task: str
    system: str
    prompt: str
    web_search: bool = False
    output_model: type[BaseModel] | None = None


class FakeOpenRouterClient:
    """Stands in for OpenRouterClient in offline stage tests.

    Queue responses in the order the code will call them, text and parse together. Each queued item is either
    a value to return or an Exception instance, which is raised. A call with an empty queue raises RuntimeError
    so that unexpected calls fail loudly. Every call is recorded in `calls`.

    A stage builds its client as `OpenRouterClient(task)`, so the fake is callable with a task name and returns a
    view that records that task. Use install_fake_client() to patch a stage module with it.
    """

    def __init__(self, *responses: Any, task: str = "fake") -> None:
        self.task = task
        self.responses: deque[Any] = deque(responses)
        self.calls: list[FakeCall] = []
        self._lock = threading.Lock()

    def queue(self, *responses: Any) -> FakeOpenRouterClient:
        with self._lock:
            self.responses.extend(responses)
        return self

    def __call__(self, task: str) -> _TaskView:
        return _TaskView(self, task)

    @property
    def tasks(self) -> list[str]:
        return [call.task for call in self.calls]

    def text(self, system: str, prompt: str, web_search: bool = False) -> str:
        return self._answer(FakeCall("text", self.task, system, prompt, web_search=web_search))

    def parse(self, system: str, prompt: str, output_model: type[Any]) -> Any:
        return self._answer(FakeCall("parse", self.task, system, prompt, output_model=output_model))

    def _answer(self, call: FakeCall) -> Any:
        with self._lock:
            self.calls.append(call)
            if not self.responses:
                raise RuntimeError(f"FakeOpenRouterClient has no queued response for {call.kind} call from task {call.task!r}")
            item = self.responses.popleft()
        if isinstance(item, BaseException):
            raise item
        if call.kind == "text":
            if not isinstance(item, str):
                raise TypeError(f"text call from task {call.task!r} needs a str response, got {type(item).__name__}")
            return item
        model = call.output_model
        assert model is not None
        if isinstance(item, model):
            return item
        if isinstance(item, BaseModel):
            item = item.model_dump()
        return model.model_validate(item)


class _TaskView:
    """What `FakeOpenRouterClient(task)` returns: the same fake, recording calls under one task name."""

    def __init__(self, fake: FakeOpenRouterClient, task: str) -> None:
        self._fake = fake
        self._task = task

    def text(self, system: str, prompt: str, web_search: bool = False) -> str:
        return self._fake._answer(FakeCall("text", self._task, system, prompt, web_search=web_search))

    def parse(self, system: str, prompt: str, output_model: type[Any]) -> Any:
        return self._fake._answer(FakeCall("parse", self._task, system, prompt, output_model=output_model))


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    fake: FakeOpenRouterClient,
) -> FakeOpenRouterClient:
    """Replace the OpenRouterClient name in one module, e.g. "agent_backend.stages.assumptions".

    Stages that build a second client elsewhere, such as DeepResearchHarness in deep_research.py, need their
    own install_fake_client call for that module.
    """
    module = importlib.import_module(module_path)
    monkeypatch.setattr(module, "OpenRouterClient", fake)
    return fake


# Live gating


def live_skip_reason() -> str | None:
    """Why live tests cannot run now, or None when they can."""
    if os.getenv("RUN_LIVE_TESTS") != "1":
        return "live test skipped: set RUN_LIVE_TESTS=1 to call OpenRouter (costs credits, results are stochastic)"
    if not os.getenv("OPENROUTER_API_KEY"):
        return "live test skipped: OPENROUTER_API_KEY is not set (put it in .env or the environment)"
    return None


def require_live() -> None:
    """Call at the start of a test that must not run offline. Skips with a clear message when it cannot run."""
    import pytest

    reason = live_skip_reason()
    if reason is not None:
        pytest.skip(reason)
