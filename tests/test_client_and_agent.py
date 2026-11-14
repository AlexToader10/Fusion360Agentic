"""BackendClient on the wire (httpx MockTransport), the Hermes edit agent's tools, and the OpenRouter wrapper."""

import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from agent_backend import openrouter_client
from agent_backend.edit_agent.agent import SYSTEM, TOOLSET, EditAgent
from agent_backend.runs import STAGE_NAMES, RunManager
from api_common.assets.assumptions import AssumptionsOutput
from api_common.assets.common import Run, RunStatus
from api_common.client.client import BackendClient, BackendError
from tests import factories

# ---- BackendClient on the wire ----------------------------------------------------------------------------------


class Wire:
    """A fake backend that records every request and answers with the next scripted response."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def client(self) -> BackendClient:
        client = BackendClient("http://backend.test")
        client.http = httpx.Client(base_url="http://backend.test", transport=httpx.MockTransport(self))
        return client


def run_json(**overrides) -> dict:
    run = Run(id=uuid4(), state=factories.full_state()).model_copy(update=overrides)
    return run.model_dump(mode="json")


@pytest.mark.parametrize(
    ("call", "method", "path_suffix", "body"),
    [
        (lambda c, rid: c.get_run(rid), "GET", "", None),
        (lambda c, rid: c.stop_run(rid), "POST", "/stop", None),
        (lambda c, rid: c.resume_run(rid), "POST", "/resume", None),
        (
            lambda c, rid: c.edit_stage(rid, "assumptions", factories.assumptions(1)),
            "PUT",
            "/assumptions",
            factories.assumptions(1).model_dump(mode="json"),
        ),
    ],
)
def test_client_sends_the_right_request_and_parses_the_run(call, method, path_suffix, body):
    payload = run_json(status=RunStatus.STOPPED)
    wire = Wire(httpx.Response(200, json=payload))
    run_id = uuid4()

    run = call(wire.client(), run_id)

    [request] = wire.requests
    assert (request.method, request.url.path) == (method, f"/runs/{run_id}{path_suffix}")
    assert (json.loads(request.content) if request.content else None) == body
    assert run == Run.model_validate(payload)


def test_start_run_serialises_the_request():
    wire = Wire(httpx.Response(200, json=run_json()))
    wire.client().start_run(factories.request("x", persona_count=7, business_plan="€ plan ✓"))
    sent = json.loads(wire.requests[0].content)
    assert sent == {
        "business_idea": "x",
        "target_audience": "audience of x",
        "additional_info": "",
        "business_plan": "€ plan ✓",
        "persona_count": 7,
    }


def test_chat_round_trip():
    payload = {"reply": "done", "run": run_json()}
    wire = Wire(httpx.Response(200, json=payload))
    response = wire.client().chat(uuid4(), "change it")
    assert json.loads(wire.requests[0].content) == {"message": "change it"}
    assert response.reply == "done" and response.run.state == factories.full_state()


@pytest.mark.parametrize("status", [404, 409])
def test_error_detail_becomes_a_backend_error(status):
    wire = Wire(httpx.Response(status, json={"detail": "Run is stopped, not running"}))
    with pytest.raises(BackendError, match="^Run is stopped, not running$"):
        wire.client().stop_run(uuid4())


def test_a_validation_error_detail_is_reported_too():
    detail = [{"loc": ["body", "persona_count"], "msg": "Input should be less than or equal to 30"}]
    wire = Wire(httpx.Response(422, json={"detail": detail}))
    with pytest.raises(BackendError, match="less than or equal to 30"):
        wire.client().start_run(factories.request())


def test_an_error_body_without_detail_falls_back_to_the_text():
    wire = Wire(httpx.Response(500, json={"message": "oops"}))
    with pytest.raises(BackendError, match="oops"):
        wire.client().get_run(uuid4())


def test_a_plain_text_error_body_becomes_a_backend_error():
    wire = Wire(httpx.Response(502, text="Bad Gateway"))
    with pytest.raises(BackendError, match="Bad Gateway"):
        wire.client().get_run(uuid4())


def test_an_error_body_that_is_json_but_not_an_object_becomes_a_backend_error():
    wire = Wire(httpx.Response(500, json=["unexpected", "shape"]))
    with pytest.raises(BackendError, match="unexpected"):
        wire.client().get_run(uuid4())


def test_a_response_that_is_not_a_run_is_rejected():
    from pydantic import ValidationError

    wire = Wire(httpx.Response(200, json={"id": "not-a-uuid", "state": {}}))
    with pytest.raises(ValidationError):
        wire.client().get_run(uuid4())


# ---- edit agent -------------------------------------------------------------------------------------------------


@pytest.fixture
def stopped_run(monkeypatch):
    import agent_backend.runs as runs

    from tests.test_runs import FakePipeline

    monkeypatch.setattr(runs, "Pipeline", FakePipeline)
    manager = RunManager()
    run = manager.start(factories.request())
    run.state = factories.state_through("interviews")
    run.status = RunStatus.STOPPED
    return manager, run


def test_agent_registers_two_tools_with_strict_schemas(hermes, stopped_run):
    EditAgent(stopped_run[0])

    assert set(hermes.registry.tools) >= {"get_run", "edit_stage"}
    get_run = hermes.registry.tools["get_run"]
    edit = hermes.registry.tools["edit_stage"]
    assert get_run["toolset"] == edit["toolset"] == TOOLSET
    assert get_run["schema"]["parameters"]["required"] == ["run_id"]
    assert edit["schema"]["parameters"]["required"] == ["run_id", "stage", "output"]
    assert edit["schema"]["parameters"]["properties"]["stage"]["enum"] == STAGE_NAMES


def test_get_run_tool_returns_the_run_as_json(hermes, stopped_run):
    manager, run = stopped_run
    EditAgent(manager)
    result = hermes.registry.call("get_run", {"run_id": str(run.id)})
    assert Run.model_validate_json(result) == run


def test_edit_stage_tool_applies_a_json_object_and_clears_later_stages(hermes, stopped_run):
    manager, run = stopped_run
    EditAgent(manager)
    new = factories.assumptions(2, tag="agent").model_dump(mode="json")

    result = hermes.registry.call("edit_stage", {"run_id": str(run.id), "stage": "assumptions", "output": new})

    assert json.loads(result) == {"edited": "assumptions"}
    assert run.state.assumptions == AssumptionsOutput.model_validate(new)
    assert [getattr(run.state, s) is None for s in STAGE_NAMES] == [False, True, True, True, True, True]


@pytest.mark.parametrize(
    ("args", "error_fragment"),
    [
        ({"run_id": "not-a-uuid", "stage": "personas", "output": {}}, "badly formed"),
        ({"run_id": str(uuid4()), "stage": "personas", "output": {}}, "not found"),
        ({"stage": "personas", "output": {}}, "run_id"),
        (None, "has no output to edit"),  # conclusions has no output yet
        ("invalid", "validation error"),
    ],
)
def test_tool_errors_are_returned_to_the_model_not_raised(hermes, stopped_run, args, error_fragment):
    manager, run = stopped_run
    EditAgent(manager)
    before = run.model_copy(deep=True)
    if args is None:
        args = {"run_id": str(run.id), "stage": "conclusions", "output": factories.conclusions().model_dump()}
    elif args == "invalid":
        args = {"run_id": str(run.id), "stage": "personas", "output": {"personas": [{"id": "P1"}]}}

    result = json.loads(hermes.registry.call("edit_stage", args))

    assert error_fragment in result["error"]
    assert run == before


def test_tool_refuses_to_edit_a_running_run(hermes, stopped_run):
    manager, run = stopped_run
    run.status = RunStatus.RUNNING
    EditAgent(manager)
    args = {"run_id": str(run.id), "stage": "assumptions", "output": factories.assumptions().model_dump()}
    assert "Stop the run" in json.loads(hermes.registry.call("edit_stage", args))["error"]


def test_chat_configures_the_agent_and_keeps_one_history_per_run(hermes, stopped_run, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_EDIT_AGENT", "special/editor")
    manager, run = stopped_run
    other = manager.start(factories.request("other"))
    agent = EditAgent(manager)
    seen_histories = []

    def script(fake, message, history):
        seen_histories.append([m["content"] for m in history])
        return f"reply to {message}"

    hermes.agent.script = script
    assert agent.chat(run.id, "first") == "reply to first"
    agent.chat(other.id, "elsewhere")
    agent.chat(run.id, "second")

    assert seen_histories == [[], [], ["first", "reply to first"]]
    first = hermes.agent.instances[0].kwargs
    assert first["model"] == "special/editor"
    assert first["enabled_toolsets"] == [TOOLSET]
    assert first["ephemeral_system_prompt"] == f"{SYSTEM}\n\nThe run id is {run.id}."
    assert hermes.agent.instances[1].kwargs["ephemeral_system_prompt"].endswith(f"{other.id}.")
    assert all(", ".join(STAGE_NAMES) in i.kwargs["ephemeral_system_prompt"] for i in hermes.agent.instances)


# ---- OpenRouter wrapper -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "task", "expected"),
    [
        ({"OPENROUTER_MODEL": "general"}, "assumptions", "general"),
        ({"OPENROUTER_MODEL": "general", "OPENROUTER_MODEL_ASSUMPTIONS": "special"}, "assumptions", "special"),
        ({"OPENROUTER_MODEL": "general", "OPENROUTER_MODEL_ASSUMPTIONS": ""}, "assumptions", "general"),
        ({"OPENROUTER_MODEL": "general", "OPENROUTER_MODEL_QUESTIONS_JUDGE": "judge"}, "questions_judge", "judge"),
        ({"OPENROUTER_MODEL": "general", "OPENROUTER_MODEL_PERSONAS": "other"}, "persona", "general"),
    ],
)
def test_model_for(monkeypatch, env, task, expected):
    for key in [k for k in __import__("os").environ if k.startswith("OPENROUTER_MODEL")]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert openrouter_client.model_for(task) == expected


def test_model_for_without_any_model_configured_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL_CONCLUSIONS", raising=False)
    with pytest.raises(KeyError, match="OPENROUTER_MODEL"):
        openrouter_client.model_for("conclusions")


@pytest.fixture
def fake_openai(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="text out"))])

        def parse(self, **kwargs):
            calls.append(("parse", kwargs))
            parsed = kwargs["response_format"].model_validate(factories.assumptions(1).model_dump())
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    class FakeOpenAI:
        def __init__(self, base_url, api_key):
            calls.append(("init", {"base_url": base_url, "api_key": api_key}))
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openrouter_client, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_MODEL_VALIDATION", "val/model")
    return calls


def test_client_text_with_and_without_web_search(fake_openai):
    client = openrouter_client.OpenRouterClient("validation")
    assert client.text("sys", "hi") == "text out"
    client.text("sys", "hi", web_search=True)

    init, plain, searching = fake_openai
    assert init[1]["base_url"] == openrouter_client.BASE_URL
    assert plain[1] == {
        "model": "val/model",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
    }
    assert searching[1]["tools"] == [openrouter_client.WEB_SEARCH_TOOL]


def test_client_parse_returns_the_parsed_model(fake_openai):
    result = openrouter_client.OpenRouterClient("validation").parse("sys", "p", AssumptionsOutput)
    assert isinstance(result, AssumptionsOutput)
    assert fake_openai[-1][1]["response_format"] is AssumptionsOutput
