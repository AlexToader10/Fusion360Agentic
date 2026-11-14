import os
import sys
import tempfile
import types

import pytest

# The backend reads these at call time; tests must never reach OpenRouter.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_MODEL", "test/general-model")
# Keep the embedded Hermes agent's state out of the checkout while testing.
os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-home-"))


class FakeRegistry:
    """Stands in for hermes' tools.registry.registry."""

    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register(self, name, toolset, schema, handler, **_):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}

    def call(self, name: str, args: dict) -> str:
        return self.tools[name]["handler"](args)


class FakeAIAgent:
    """Stands in for hermes' run_agent.AIAgent. A test sets FakeAIAgent.script to drive a conversation:
    script(agent, message, history) -> final response text. The script can call tools through the registry."""

    script = None
    instances: list["FakeAIAgent"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeAIAgent.instances.append(self)

    def run_conversation(self, message, conversation_history=None):
        history = list(conversation_history or [])
        reply = type(self).script(self, message, history)
        messages = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        return {"final_response": reply, "messages": messages}


def _install_hermes_stubs() -> None:
    """Hermes Agent is installed from a local checkout (vendor/). Stub it when it is not there."""
    try:
        import run_agent  # noqa: F401
        import tools.registry  # noqa: F401
    except ImportError:
        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = FakeAIAgent
        tools = types.ModuleType("tools")
        registry_module = types.ModuleType("tools.registry")
        registry_module.registry = FakeRegistry()
        tools.registry = registry_module
        sys.modules.update({"run_agent": run_agent, "tools": tools, "tools.registry": registry_module})


_install_hermes_stubs()


@pytest.fixture
def control(tmp_path):
    from tests.fakes import StageControl

    control = StageControl(tmp_path)
    yield control
    control.release_all()  # never leave a worker blocked on a gate


@pytest.fixture
def fake_stages(control):
    from api_common.assets.common import STAGE_OUTPUTS
    from tests.fakes import FileControlledStage

    return [FileControlledStage(name, str(control.workdir)) for name in STAGE_OUTPUTS]


@pytest.fixture
def make_pipeline(control):
    """Builds real Pipelines (real process pools) and shuts every one of them down after the test."""
    from agent_backend.pipeline import Pipeline

    pipelines = []

    def make(*args, **kwargs):
        pipeline = Pipeline(*args, **kwargs)
        pipelines.append(pipeline)
        return pipeline

    yield make
    control.release_all()
    for pipeline in pipelines:
        for pool in pipeline.pools:
            pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def live_manager(monkeypatch, fake_stages, make_pipeline):
    """A RunManager with the real Pipeline and process pools, whose stages are FileControlledStage."""
    import agent_backend.runs as runs

    monkeypatch.setattr(runs, "STAGES", fake_stages)
    monkeypatch.setattr(runs, "Pipeline", make_pipeline)
    return runs.RunManager()


@pytest.fixture
def scripted(monkeypatch):
    """Replaces OpenRouterClient in every stage module with ScriptedClient. Returns the class to script it."""
    from agent_backend.stages import (
        assumptions,
        conclusions,
        deep_research,
        interviews,
        personas,
        questions,
        validations,
    )
    from tests.fakes import ScriptedClient

    for module in (assumptions, conclusions, deep_research, interviews, personas, questions, validations):
        monkeypatch.setattr(module, "OpenRouterClient", ScriptedClient)
    ScriptedClient.reset()
    return ScriptedClient


@pytest.fixture
def hermes():
    """The stubbed hermes registry and agent class, reset for the test."""
    import tools.registry

    if not isinstance(tools.registry.registry, FakeRegistry):
        pytest.skip("real Hermes Agent is installed; these tests drive the stub")
    FakeAIAgent.script = None
    FakeAIAgent.instances = []
    return types.SimpleNamespace(registry=tools.registry.registry, agent=FakeAIAgent)
