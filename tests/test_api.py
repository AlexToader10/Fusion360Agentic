"""The FastAPI app and the BackendClient against it: routes, status codes, body validation, and full lifecycles that go
client -> HTTP -> RunManager -> real process pools -> back."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import agent_backend.main as main
from agent_backend.edit_agent.agent import EditAgent
from api_common.assets.common import STAGE_OUTPUTS, Run, RunStatus
from api_common.client.client import BackendClient, BackendError
from tests import factories
from tests.fakes import wait_until


@pytest.fixture
def app(monkeypatch, live_manager):
    """Works with the real Hermes or the stub; only the chat tests need the stub (they request the hermes fixture)."""
    monkeypatch.setattr(main, "manager", live_manager)
    monkeypatch.setattr(main, "edit_agent", EditAgent(live_manager))
    return main.app


@pytest.fixture
def http(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def backend(http):
    """The real BackendClient, wired to the app in-process (TestClient is an httpx.Client)."""
    client = BackendClient()
    client.http = http
    return client


def wait_for_status(backend, run_id, status, timeout=20) -> Run:
    wait_until(lambda: backend.get_run(run_id).status == status, timeout=timeout, message=f"status {status}")
    return backend.get_run(run_id)


# ---- routes and status codes ------------------------------------------------------------------------------------


def test_one_edit_route_per_stage_with_its_own_body_schema(app):
    schema = app.openapi()
    for stage, model in STAGE_OUTPUTS.items():
        operation = schema["paths"][f"/runs/{{run_id}}/{stage}"]["put"]
        body_ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert body_ref.endswith(f"/{model.__name__}"), f"{stage} edits with the wrong model"


def test_start_returns_the_new_running_run(http):
    response = http.post("/runs", json={"business_idea": "api", "target_audience": "devs", "persona_count": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["state"]["request"]["persona_count"] == 3
    assert body["state"]["assumptions"] is None


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"business_idea": "x"},
        {"business_idea": "x", "target_audience": "y", "persona_count": 0},
        {"business_idea": "x", "target_audience": "y", "persona_count": 31},
        {"business_idea": "x", "target_audience": "y", "persona_count": "many"},
    ],
)
def test_start_rejects_an_invalid_request(http, live_manager, body):
    assert http.post("/runs", json=body).status_code == 422
    assert live_manager.runs == {}  # nothing was started


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", ""), ("POST", "/stop"), ("POST", "/resume"), ("PUT", "/assumptions"), ("POST", "/chat")],
)
def test_unknown_run_is_404_on_every_route(http, method, path):
    body = {"assumptions": []} if method == "PUT" else {"message": "hi"} if path == "/chat" else None
    response = http.request(method, f"/runs/{uuid4()}{path}", json=body)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_a_malformed_run_id_is_422(http):
    assert http.get("/runs/not-a-uuid").status_code == 422


def test_editing_a_stage_that_does_not_exist_is_not_routed(http, backend):
    run = backend.start_run(factories.request("routes"))
    assert http.put(f"/runs/{run.id}/request", json={}).status_code in {404, 405}
    assert http.put(f"/runs/{run.id}/stop", json={}).status_code == 405  # /stop is POST only


def test_edit_body_is_validated_against_the_stage_model(http, backend, control):
    control.gate("questions", "validate")
    run = backend.start_run(factories.request("validate"))
    wait_until(lambda: control.started("questions", "validate"))
    backend.stop_run(run.id)

    wrong_model = factories.questions().model_dump()
    assert http.put(f"/runs/{run.id}/personas", json=wrong_model).status_code == 422
    bad_type = factories.personas().model_dump()
    bad_type["personas"][0]["age"] = "old"
    assert http.put(f"/runs/{run.id}/personas", json=bad_type).status_code == 422
    assert backend.get_run(run.id).state.personas == factories.personas(tag="validate#0")  # unchanged


def test_refused_actions_are_409_with_the_reason(backend, control):
    control.gate("assumptions", "busy")
    run = backend.start_run(factories.request("busy"))

    with pytest.raises(BackendError, match="Stop the run before editing it"):
        backend.edit_stage(run.id, "assumptions", factories.assumptions())
    with pytest.raises(BackendError, match="already running"):
        backend.resume_run(run.id)

    backend.stop_run(run.id)
    with pytest.raises(BackendError, match="not running"):
        backend.stop_run(run.id)
    with pytest.raises(BackendError, match="has no output to edit"):
        backend.edit_stage(run.id, "assumptions", factories.assumptions())


# ---- full lifecycles through the client -------------------------------------------------------------------------


def test_client_drives_a_run_to_completion(backend):
    run = backend.start_run(factories.request("happy"))
    done = wait_for_status(backend, run.id, RunStatus.COMPLETED)

    assert done.id == run.id
    assert all(getattr(done.state, stage) is not None for stage in STAGE_OUTPUTS)
    assert done.state == factories.state_through("conclusions", "happy").model_copy(
        update={stage: factories.output(stage, "happy#0") for stage in STAGE_OUTPUTS}
    )


def test_client_hold_edit_resume_cycle_round_trips_every_model_over_json(backend, control):
    control.gate("validations", "cycle")
    run = backend.start_run(factories.request("cycle"))
    wait_until(lambda: control.started("validations", "cycle"))

    held = backend.stop_run(run.id)
    assert held.status == RunStatus.STOPPED

    # edit two stages in a row: the second edit is on an earlier stage and clears what the first one kept
    edited_interviews = factories.interviews(persona_count=1, tag="by hand")
    after_first = backend.edit_stage(run.id, "interviews", edited_interviews)
    assert after_first.state.interviews == edited_interviews
    edited_questions = factories.questions(n=5, tag="by hand")
    after_second = backend.edit_stage(run.id, "questions", edited_questions)
    assert after_second.state.questions == edited_questions and after_second.state.interviews is None

    control.open("validations", "cycle")  # lets the held execution finish (its output is dropped) and the new one run
    resumed = backend.resume_run(run.id)
    assert resumed.status == RunStatus.RUNNING
    done = wait_for_status(backend, run.id, RunStatus.COMPLETED)
    assert done.state.questions == edited_questions
    assert done.state.interviews.interviews[0].research.endswith("cycle#1")
    # held execution (dropped, may finish last) + the one after resume
    wait_until(lambda: control.runs_of("validations", "cycle") == 2, message="both validations executions")


def test_a_failed_run_reports_its_error_and_can_be_resumed_over_http(backend, control):
    control.fail("conclusions", "fails", "context length exceeded")
    run = backend.start_run(factories.request("fails"))
    failed = wait_for_status(backend, run.id, RunStatus.FAILED)
    assert failed.error == "context length exceeded"

    control.heal("conclusions", "fails")
    backend.resume_run(run.id)
    done = wait_for_status(backend, run.id, RunStatus.COMPLETED)
    assert done.error is None


# ---- chat with the edit agent -----------------------------------------------------------------------------------


def test_chat_lets_the_agent_edit_the_held_run_through_its_tools(backend, control, hermes):
    control.gate("interviews", "chat")
    run = backend.start_run(factories.request("chat"))
    wait_until(lambda: control.started("interviews", "chat"))
    backend.stop_run(run.id)

    def agent_script(agent, message, history):
        current = Run.model_validate_json(hermes.registry.call("get_run", {"run_id": str(run.id)}))
        personas = current.state.personas.model_dump()
        personas["personas"][0]["age"] = 70
        hermes.registry.call("edit_stage", {"run_id": str(run.id), "stage": "personas", "output": personas})
        return "Made P1 70. Questions and later stages will run again."

    hermes.agent.script = agent_script
    response = backend.chat(run.id, "make the first persona a retiree")

    assert response.reply.startswith("Made P1 70")
    assert response.run.state.personas.personas[0].age == 70  # the reply carries the run after the edit
    assert response.run.state.questions is None and response.run.status == RunStatus.STOPPED


def test_chat_on_an_unknown_run_never_reaches_the_agent(backend, hermes):
    hermes.agent.script = lambda *_: pytest.fail("the agent must not run")
    with pytest.raises(BackendError, match="not found"):
        backend.chat(uuid4(), "hello")
    assert hermes.agent.instances == []


def test_a_server_error_reaches_the_caller_as_a_backend_error(app, backend, control, hermes):
    def broken(agent, message, history):
        raise RuntimeError("OpenRouter is down")

    hermes.agent.script = broken
    backend.http = TestClient(app, raise_server_exceptions=False)
    control.gate("assumptions", "down")
    run = backend.start_run(factories.request("down"))
    backend.stop_run(run.id)

    with pytest.raises(BackendError):
        backend.chat(run.id, "anything")
