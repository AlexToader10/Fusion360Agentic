"""Offline contract tests for the HTTP API, the WebSocket event stream and the Python client.

The edit agent imports Hermes (`run_agent`, `tools.registry`), which is not installed in the test environment.
Lightweight stand-ins are put in sys.modules before agent_backend.main is imported. No model is ever called: the
pipeline is replaced by a fake that records submissions, and the chat endpoint gets a fake edit agent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable, Iterator
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient


def _install_hermes_stubs() -> None:
    """Stand-ins for the Hermes modules, used only where the real ones cannot be imported."""
    if "run_agent" not in sys.modules and importlib.util.find_spec("run_agent") is None:
        run_agent = types.ModuleType("run_agent")

        class AIAgent:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def run_conversation(self, message: str, conversation_history: object = None) -> dict:
                raise RuntimeError("fake AIAgent: these tests replace EditAgent.chat and never call a model")

        run_agent.AIAgent = AIAgent
        sys.modules["run_agent"] = run_agent

    if "tools.registry" not in sys.modules and importlib.util.find_spec("tools") is None:
        tools = types.ModuleType("tools")
        tools.__path__ = []  # a package, so that `tools.registry` can be imported from it
        registry_module = types.ModuleType("tools.registry")

        class _Registry:
            def register(self, **kwargs: object) -> None:
                return None

        registry_module.registry = _Registry()
        tools.registry = registry_module
        sys.modules["tools"] = tools
        sys.modules["tools.registry"] = registry_module


_install_hermes_stubs()

from agent_backend import main as app_main  # noqa: E402
from agent_backend.runs import STAGE_NAMES, InvalidAction, RunManager  # noqa: E402
from api_common.assets.assumptions import AssumptionsOutput  # noqa: E402
from api_common.assets.common import (  # noqa: E402
    STAGE_OUTPUTS,
    PipelineRequest,
    PipelineState,
    Run,
    RunStatus,
)
from api_common.assets.personas import PersonasOutput  # noqa: E402
from api_common.assets.questions import QuestionsOutput  # noqa: E402
from api_common.client.client import BackendClient, BackendError  # noqa: E402


class _FakePipeline:
    """Records what the manager submits and stops. It never runs a stage."""

    def __init__(self) -> None:
        self.submitted: list[tuple[UUID, int]] = []
        self.stopped: list[UUID] = []
        self.shut_down = False

    def submit(self, run_id: UUID, state: PipelineState, start: int = 0) -> None:
        self.submitted.append((run_id, start))

    def stop(self, run_id: UUID) -> None:
        self.stopped.append(run_id)

    def shutdown(self) -> None:
        self.shut_down = True


class _FakeEditAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[UUID, str]] = []

    def chat(self, run_id: UUID, message: str) -> str:
        self.calls.append((run_id, message))
        return self.reply


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> RunManager:
    fresh = RunManager()
    fresh.pipeline = _FakePipeline()  # type: ignore[assignment]
    monkeypatch.setattr(app_main, "manager", fresh)
    return fresh


@pytest.fixture
def client(manager: RunManager) -> Iterator[TestClient]:
    with TestClient(app_main.app) as test_client:
        yield test_client


def _pipeline(manager: RunManager) -> _FakePipeline:
    assert isinstance(manager.pipeline, _FakePipeline)
    return manager.pipeline


def _start(client: TestClient, request: PipelineRequest) -> UUID:
    return UUID(client.post("/runs", json=request.model_dump(mode="json")).json()["id"])


def _held_run(client: TestClient, manager: RunManager, state: PipelineState) -> UUID:
    """A run carrying `state`, stopped the way a user stops it."""
    run_id = _start(client, state.request)
    manager.get(run_id).state = state.model_copy(deep=True)
    assert client.post(f"/runs/{run_id}/stop").status_code == 200
    return run_id


def _finished_run(manager: RunManager, state: PipelineState) -> UUID:
    """A run with three events (run_started, status, run_completed) whose last stage is filled."""
    run = manager.start(state.request)
    manager._event(run.id, {"kind": "status", "stage": "assumptions", "message": "Drafting assumptions"})
    manager._stage_done(run.id, state.model_copy(deep=True))
    return run.id


def test_app_imports_with_hermes_stubbed_and_registers_every_route():
    assert isinstance(app_main.app, FastAPI)
    http_routes = {
        (method, route.path)
        for route in app_main.app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    socket_routes = {route.path for route in app_main.app.routes if isinstance(route, APIWebSocketRoute)}

    assert {
        ("POST", "/runs"),
        ("GET", "/runs/{run_id}"),
        ("POST", "/runs/{run_id}/stop"),
        ("POST", "/runs/{run_id}/resume"),
        ("POST", "/runs/{run_id}/chat"),
    } <= http_routes
    assert {("PUT", f"/runs/{{run_id}}/{stage}") for stage in STAGE_OUTPUTS} <= http_routes
    assert set(STAGE_OUTPUTS) == set(STAGE_NAMES)
    assert "/runs/{run_id}/events" in socket_routes
    assert "run_agent" in sys.modules and "tools.registry" in sys.modules


def test_post_runs_starts_a_running_run_and_submits_the_first_stage(client, manager, sample_request):
    response = client.post("/runs", json=sample_request.model_dump(mode="json"))

    assert response.status_code == 200
    body = response.json()
    run_id = UUID(body["id"])
    assert body["status"] == "running"
    assert body["state"]["request"]["persona_count"] == 3
    assert _pipeline(manager).submitted == [(run_id, 0)]
    assert manager.events(run_id)[0]["kind"] == "run_started"


@pytest.mark.parametrize("persona_count", [0, 31])
def test_post_runs_rejects_persona_count_outside_1_to_30(client, manager, sample_request, persona_count):
    payload = sample_request.model_dump(mode="json") | {"persona_count": persona_count}

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    assert manager.runs == {}
    assert _pipeline(manager).submitted == []


def test_get_run_returns_the_run_and_unknown_id_is_404(client, sample_request):
    run_id = _start(client, sample_request)

    found = client.get(f"/runs/{run_id}")
    assert found.status_code == 200
    assert found.json()["id"] == str(run_id)

    missing = uuid4()
    not_found = client.get(f"/runs/{missing}")
    assert not_found.status_code == 404
    assert str(missing) in not_found.json()["detail"]


def test_stop_holds_a_running_run_and_a_second_stop_conflicts(client, manager, sample_request):
    run_id = _start(client, sample_request)

    first = client.post(f"/runs/{run_id}/stop")
    assert first.status_code == 200
    assert first.json()["status"] == "stopped"
    assert _pipeline(manager).stopped == [run_id]

    second = client.post(f"/runs/{run_id}/stop")
    assert second.status_code == 409
    assert "not running" in second.json()["detail"]
    assert _pipeline(manager).stopped == [run_id]

    unknown = client.post(f"/runs/{uuid4()}/stop")
    assert unknown.status_code == 404
    assert "detail" in unknown.json()


def test_resume_continues_at_the_first_missing_stage_and_completed_run_stays_completed(
    client, manager, sample_request, sample_assumptions, sample_personas, full_state
):
    partial = PipelineState(request=sample_request, assumptions=sample_assumptions, personas=sample_personas)
    run_id = _held_run(client, manager, partial)

    resumed = client.post(f"/runs/{run_id}/resume")

    assert resumed.status_code == 200
    assert resumed.json()["status"] == "running"
    assert _pipeline(manager).submitted[-1] == (run_id, STAGE_NAMES.index("questions"))

    done_id = _held_run(client, manager, full_state)
    submitted_before = len(_pipeline(manager).submitted)

    done = client.post(f"/runs/{done_id}/resume")

    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert len(_pipeline(manager).submitted) == submitted_before

    assert client.post(f"/runs/{uuid4()}/resume").status_code == 404


def test_put_replaces_the_stage_output_clears_later_stages_and_keeps_the_run_stopped(client, manager, full_state):
    run_id = _held_run(client, manager, full_state)
    one_persona = PersonasOutput(personas=[full_state.personas.personas[0]])

    response = client.put(f"/runs/{run_id}/personas", json=one_persona.model_dump(mode="json"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stopped"
    assert len(body["state"]["personas"]["personas"]) == 1
    assert body["state"]["assumptions"] == full_state.assumptions.model_dump(mode="json")
    for later in STAGE_NAMES[STAGE_NAMES.index("questions") :]:
        assert body["state"][later] is None
    edited = [event for event in manager.events(run_id) if event["kind"] == "stage_edited"]
    assert edited[-1]["data"]["cleared"] == ["questions", "interviews", "validations", "conclusions"]


def test_put_rejects_running_runs_missing_outputs_unknown_stages_and_bad_bodies(
    client, manager, sample_request, sample_assumptions, sample_questions
):
    running = _start(client, sample_request)
    running_put = client.put(f"/runs/{running}/assumptions", json=sample_assumptions.model_dump(mode="json"))
    assert running_put.status_code == 409
    assert "Stop the run" in running_put.json()["detail"]

    held = _held_run(client, manager, PipelineState(request=sample_request, assumptions=sample_assumptions))
    no_output = client.put(f"/runs/{held}/questions", json=sample_questions.model_dump(mode="json"))
    assert no_output.status_code == 409
    assert "no output" in no_output.json()["detail"]

    with pytest.raises(InvalidAction):
        manager.edit(held, "bogus", {})
    # The stage name is part of the route, so an unknown name never reaches the handler and gets a 404.
    assert client.put(f"/runs/{held}/bogus", json={}).status_code == 404

    bad_body = client.put(f"/runs/{held}/assumptions", json={"assumptions": [{"id": "A1"}]})
    assert bad_body.status_code == 422
    assert manager.get(held).state.assumptions == sample_assumptions


def test_chat_forwards_the_message_to_the_edit_agent_and_returns_reply_with_run(
    client, manager, sample_request, monkeypatch
):
    agent = _FakeEditAgent("Kept only Laura and noted the change.")
    monkeypatch.setattr(app_main, "edit_agent", agent)
    run_id = _start(client, sample_request)

    response = client.post(f"/runs/{run_id}/chat", json={"message": "Keep only Laura"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == agent.reply
    assert body["run"]["id"] == str(run_id)
    assert agent.calls == [(run_id, "Keep only Laura")]

    unknown = client.post(f"/runs/{uuid4()}/chat", json={"message": "hello"})
    assert unknown.status_code == 404
    assert len(agent.calls) == 1


def test_events_socket_sends_history_in_order_then_run_state_and_closes(client, manager, full_state):
    run_id = _finished_run(manager, full_state)

    with client.websocket_connect(f"/runs/{run_id}/events") as ws:
        messages = [ws.receive_json() for _ in range(4)]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert closed.value.code == 1000
    assert [message["kind"] for message in messages] == ["run_started", "status", "run_completed", "run_state"]
    seqs = [message["seq"] for message in messages[:-1]]
    assert seqs == [1, 2, 3]
    assert all(later > earlier for earlier, later in zip(seqs, seqs[1:]))
    assert manager.last_event_seq(run_id) == 3

    snapshot = messages[-1]
    assert snapshot["last_seq"] == 3
    assert snapshot["run"]["status"] == "completed"
    assert snapshot["run"]["state"]["conclusions"] is not None


def test_events_socket_with_after_skips_the_events_already_seen(client, manager, full_state):
    run_id = _finished_run(manager, full_state)

    with client.websocket_connect(f"/runs/{run_id}/events?after=2") as ws:
        messages = [ws.receive_json() for _ in range(2)]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert [message["kind"] for message in messages] == ["run_completed", "run_state"]
    assert messages[0]["seq"] == 3


def test_events_socket_for_unknown_run_sends_error_then_closes_with_4404(client, manager):
    with client.websocket_connect(f"/runs/{uuid4()}/events") as ws:
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert error["kind"] == "error"
    assert "not found" in error["message"]
    assert closed.value.code == 4404


def _backend(handler: Callable[[httpx.Request], httpx.Response]) -> BackendClient:
    backend = BackendClient("http://backend.test")
    backend.http.close()
    backend.http = httpx.Client(base_url="http://backend.test", transport=httpx.MockTransport(handler))
    return backend


def _run_json(run_id: UUID, status: RunStatus, request: PipelineRequest) -> dict:
    return Run(id=run_id, status=status, state=PipelineState(request=request)).model_dump(mode="json")


def test_backend_client_sends_each_call_as_the_api_expects_and_builds_event_urls(sample_request, sample_assumptions):
    run_id = uuid4()
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/chat"):
            run = _run_json(run_id, RunStatus.STOPPED, sample_request)
            return httpx.Response(200, json={"reply": "Done.", "run": run})
        status = RunStatus.STOPPED if request.url.path.endswith(("/stop", "/assumptions")) else RunStatus.RUNNING
        return httpx.Response(200, json=_run_json(run_id, status, sample_request))

    backend = _backend(handler)
    started = backend.start_run(sample_request)
    fetched = backend.get_run(run_id)
    stopped = backend.stop_run(run_id)
    resumed = backend.resume_run(run_id)
    edited = backend.edit_stage(run_id, "assumptions", sample_assumptions)
    chat = backend.chat(run_id, "Make A1 sharper")
    backend.http.close()

    assert [(request.method, request.url.path) for request in sent] == [
        ("POST", "/runs"),
        ("GET", f"/runs/{run_id}"),
        ("POST", f"/runs/{run_id}/stop"),
        ("POST", f"/runs/{run_id}/resume"),
        ("PUT", f"/runs/{run_id}/assumptions"),
        ("POST", f"/runs/{run_id}/chat"),
    ]
    assert json.loads(sent[0].content) == sample_request.model_dump(mode="json")
    assert sent[1].content == b""
    assert json.loads(sent[4].content) == sample_assumptions.model_dump(mode="json")
    assert json.loads(sent[5].content) == {"message": "Make A1 sharper"}

    assert started.status == RunStatus.RUNNING
    assert fetched.id == run_id
    assert stopped.status == RunStatus.STOPPED
    assert resumed.status == RunStatus.RUNNING
    assert edited.status == RunStatus.STOPPED
    assert chat.reply == "Done."
    assert chat.run.id == run_id

    secure = BackendClient("https://backend.example.com")
    plain = BackendClient("http://localhost:8000/")
    assert secure.events_url(run_id) == f"wss://backend.example.com/runs/{run_id}/events?after=0"
    assert plain.events_url(run_id, after=7) == f"ws://localhost:8000/runs/{run_id}/events?after=7"
    secure.http.close()
    plain.http.close()


def test_backend_client_raises_backend_error_carrying_the_detail_text():
    run_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404, json={"detail": f"Run {run_id} not found"})
        return httpx.Response(409, json={"detail": "Run is stopped, not running"})

    backend = _backend(handler)
    with pytest.raises(BackendError) as missing:
        backend.get_run(run_id)
    assert str(missing.value) == f"Run {run_id} not found"

    with pytest.raises(BackendError) as conflict:
        backend.stop_run(run_id)
    assert str(conflict.value) == "Run is stopped, not running"
    backend.http.close()
