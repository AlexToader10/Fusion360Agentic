"""The edit agent on the real Hermes Agent, against a local fake of the OpenRouter endpoint.

Runs only when Hermes is installed (vendor/hermes-agent); otherwise the stubbed tests in test_client_and_agent.py and
test_api.py cover the agent. The fake model is scripted per test: each completion request gets the next reply."""

import json
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent_backend.edit_agent.agent as agent_module
import agent_backend.runs as runs
from api_common.assets.common import RunStatus
from tests import factories
from tests.conftest import FakeRegistry
from tests.test_runs import FakePipeline


@pytest.fixture(autouse=True)
def real_hermes():
    import tools.registry

    if isinstance(tools.registry.registry, FakeRegistry):
        pytest.skip("Hermes Agent is not installed (see the README setup)")
    # A Hermes conversation moves vendor/hermes-agent to the front of sys.path. Hermes ships its own `tests` package,
    # which would then shadow ours in the pipeline's worker processes for every later test.
    saved = sys.path[:]
    yield
    sys.path[:] = saved


class FakeModel:
    """OpenAI-compatible endpoint. script(body) -> assistant message dict, one entry per completion request."""

    def __init__(self):
        self.script: list[Callable[[dict], dict]] = []
        self.requests: list[dict] = []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # model metadata lookups
                self._json({"data": [{"id": "test/model", "context_length": 200000}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(body)
                message = model.script.pop(0)(body)
                finish = "tool_calls" if message.get("tool_calls") else "stop"
                if body.get("stream"):
                    self._stream(message, finish)
                else:
                    self._json(
                        {
                            "id": "c", "object": "chat.completion", "created": int(time.time()), "model": "test/model",
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                        }
                    )  # fmt: skip

            def _stream(self, message, finish):
                base = {"id": "c", "object": "chat.completion.chunk", "created": int(time.time()), "model": "test/model"}
                delta = {"role": "assistant", "content": message.get("content")}
                if message.get("tool_calls"):
                    delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(message["tool_calls"])]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (
                    dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                    dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]),
                ):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def call(tool: str, arguments: dict) -> Callable[[dict], dict]:
        """A reply that calls a tool, directly when Hermes lists it, otherwise through its tool_call bridge."""

        def reply(body):
            listed = {t["function"]["name"] for t in body.get("tools", [])}
            name, args = (tool, arguments) if tool in listed else ("tool_call", {"name": tool, "arguments": arguments})
            call = {"id": f"call_{tool}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            return {"role": "assistant", "content": None, "tool_calls": [call]}

        return reply

    @staticmethod
    def say(text: str) -> Callable[[dict], dict]:
        return lambda body: {"role": "assistant", "content": text}

    @staticmethod
    def tool_results(body) -> list[str]:
        return [m["content"] for m in body["messages"] if m.get("role") == "tool"]


@pytest.fixture
def model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(agent_module, "BASE_URL", model.url)
    yield model
    model.server.shutdown()


@pytest.fixture
def held(monkeypatch):
    monkeypatch.setattr(runs, "Pipeline", FakePipeline)
    manager = runs.RunManager()
    run = manager.start(factories.request())
    run.state = factories.state_through("interviews")
    run.status = RunStatus.STOPPED
    return manager, run


def test_the_agent_edits_a_stage_through_the_real_tool_loop(model, held):
    manager, run = held
    personas = factories.personas().model_dump()
    personas["personas"][0]["age"] = 70
    model.script = [
        model.call("get_run", {"run_id": str(run.id)}),
        model.call("edit_stage", {"run_id": str(run.id), "stage": "personas", "output": personas}),
        model.say("P1 is 70 now. Questions, interviews, validations and conclusions will run again."),
    ]

    reply = agent_module.EditAgent(manager).chat(run.id, "make the first persona 70")

    assert reply.startswith("P1 is 70 now")
    assert run.state.personas.personas[0].age == 70
    assert run.state.questions is None and run.state.interviews is None
    get_run_result, edit_result = model.tool_results(model.requests[-1])
    assert json.loads(get_run_result)["id"] == str(run.id)
    assert '"edited": "personas"' in edit_result
    system = next(m["content"] for m in model.requests[0]["messages"] if m["role"] == "system")
    assert f"The run id is {run.id}." in system


def test_a_rejected_edit_reaches_the_model_as_an_error_and_changes_nothing(model, held):
    manager, run = held
    before = run.model_copy(deep=True)
    model.script = [
        model.call("edit_stage", {"run_id": str(run.id), "stage": "personas", "output": {"personas": [{"id": "P1"}]}}),
        model.say("That edit was invalid."),
    ]

    agent_module.EditAgent(manager).chat(run.id, "break it")

    [result] = model.tool_results(model.requests[-1])
    assert "validation error" in result
    assert run == before


def test_the_conversation_carries_over_to_the_next_message_on_the_same_run(model, held):
    manager, run = held
    agent = agent_module.EditAgent(manager)
    model.script = [model.say("first answer"), model.say("second answer")]

    agent.chat(run.id, "first question")
    agent.chat(run.id, "second question")

    contents = [m.get("content") for m in model.requests[-1]["messages"]]
    assert "first question" in str(contents) and "first answer" in str(contents)
    assert str(contents).index("first question") < str(contents).index("second question")
