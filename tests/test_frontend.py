"""The Streamlit frontend, run headless with streamlit.testing AppTest against an in-memory fake of BackendClient."""

from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import api_common.client.client as client_module  # noqa: E402
from api_common.assets.common import STAGE_OUTPUTS, ChatResponse, PipelineState, Run, RunStatus  # noqa: E402
from api_common.client.client import BackendError  # noqa: E402
from tests import factories  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "frontend_demo" / "app.py")


class FakeBackend:
    """Replaces BackendClient. The app builds a new client on every script run, so the state lives on the class."""

    run: Run | None = None
    calls: list[tuple] = []
    refuse: dict[str, str] = {}  # method name -> BackendError message

    def __init__(self, base_url: str = "", timeout: float = 0):
        pass

    @classmethod
    def reset(cls, run: Run | None = None):
        cls.run, cls.calls, cls.refuse = run, [], {}

    def _call(self, name, *args):
        type(self).calls.append((name, *args))
        if name in self.refuse:
            raise BackendError(self.refuse[name])

    def start_run(self, request):
        self._call("start_run", request)
        type(self).run = Run(id=uuid4(), status=RunStatus.STOPPED, state=PipelineState(request=request))
        return self.run

    def get_run(self, run_id):
        self._call("get_run", run_id)
        return self.run

    def stop_run(self, run_id):
        self._call("stop_run", run_id)
        self.run.status = RunStatus.STOPPED
        return self.run

    def resume_run(self, run_id):
        self._call("resume_run", run_id)
        self.run.status = RunStatus.COMPLETED
        return self.run

    def edit_stage(self, run_id, stage, output):
        self._call("edit_stage", run_id, stage, output)
        setattr(self.run.state, stage, output)
        return self.run

    def chat(self, run_id, message):
        self._call("chat", run_id, message)
        return ChatResponse(reply=f"echo: {message}", run=self.run)

    @classmethod
    def called(cls, name) -> list[tuple]:
        return [call[1:] for call in cls.calls if call[0] == name]


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(client_module, "BackendClient", FakeBackend)
    FakeBackend.reset()
    return FakeBackend


def app_with_run(run: Run) -> AppTest:
    FakeBackend.run = run
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state.run_id = run.id
    at.session_state.chat = []
    return at.run()


def stopped(last_stage: str | None, tag: str = "idea") -> Run:
    return Run(id=uuid4(), status=RunStatus.STOPPED, state=factories.state_through(last_stage, tag))


def verdict_frame(at):
    return next(d.value for d in at.dataframe if list(d.value.columns)[0] == "Assumption")


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    return "\n".join(parts)


# ---- start form -------------------------------------------------------------------------------------------------


def test_start_form_requires_an_idea_and_an_audience_or_a_plan(backend):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("only an idea")
    at.button[0].click().run()

    assert "Upload a business plan" in at.error[0].value
    assert backend.called("start_run") == []


def test_start_form_starts_a_run_and_switches_to_the_run_view(backend):
    at = AppTest.from_file(APP, default_timeout=30).run()
    idea, audience, info = at.text_area
    idea.input("Subscription sourdough")
    audience.input("Busy parents")
    info.input("We already bake for 2 cafés")
    at.number_input[0].set_value(4)
    at.button[0].click().run()

    [(request,)] = backend.called("start_run")
    assert (request.business_idea, request.target_audience, request.persona_count) == (
        "Subscription sourdough",
        "Busy parents",
        4,
    )
    assert request.additional_info == "We already bake for 2 cafés" and request.business_plan == ""
    assert at.session_state.run_id == backend.run.id
    assert at.title[0].value == "Business idea validation"
    assert not at.exception


# ---- run view ---------------------------------------------------------------------------------------------------


def test_a_completed_run_renders_every_stage_without_errors(backend):
    at = app_with_run(Run(id=uuid4(), status=RunStatus.COMPLETED, state=factories.full_state()))

    assert not at.exception
    assert [s.value for s in at.subheader][:6] == [f"{n}. {s.capitalize()}" for n, s in enumerate(STAGE_OUTPUTS, 1)]
    text = all_text(at)
    assert "\\$3 a loaf" in text  # prices are escaped so Streamlit does not read them as LaTeX
    assert "Waiting for the previous stages." not in text
    assert [b.label for b in at.sidebar.button] == ["New run"]  # neither stop nor resume on a completed run
    assert len(at.dataframe) == 4  # the verdict matrix plus the three table editors


def test_the_verdict_matrix_has_one_row_per_assumption_and_one_column_per_persona(backend):
    at = app_with_run(Run(id=uuid4(), status=RunStatus.COMPLETED, state=factories.full_state()))
    frame = verdict_frame(at)
    assert list(frame.columns) == ["Assumption", "Person 1 idea", "Person 2 idea"]
    assert len(frame) == 3
    assert set(frame.iloc[:, 1:].to_numpy().ravel()) <= {"✅ validated", "❌ invalidated", "❔ inconclusive"}


def test_a_held_run_shows_resume_and_the_pending_stages(backend):
    at = app_with_run(stopped("personas"))

    assert [b.label for b in at.sidebar.button] == ["Resume", "New run"]
    assert all_text(at).count("Waiting for the previous stages.") == 4
    at.sidebar.button[0].click().run()
    assert backend.called("resume_run") == [(backend.run.id,)]


def test_a_refused_action_is_shown_after_the_rerun(backend):
    at = app_with_run(stopped("personas"))
    backend.refuse["resume_run"] = "Run is already running"
    at.sidebar.button[0].click().run()

    assert [e.value for e in at.error] == ["Run is already running"]
    at.run()
    assert not at.error  # the refusal is shown once


def test_a_failed_run_shows_its_error(backend):
    run = stopped("questions")
    run.status, run.error = RunStatus.FAILED, "provider returned 502"
    at = app_with_run(run)
    assert [e.value for e in at.error] == ["provider returned 502"]
    assert at.sidebar.button[0].label == "Resume"


def test_a_backend_failure_on_load_offers_a_new_run(backend):
    backend.refuse["get_run"] = "Run 123 not found"
    at = app_with_run(stopped(None))
    assert "Backend error: Run 123 not found" in at.error[0].value
    assert [b.label for b in at.button] == ["New run"]


# ---- editing ----------------------------------------------------------------------------------------------------


def json_editor(at: AppTest):
    return next(t for t in at.text_area if t.label == "Output as JSON")


def save_button(at: AppTest, stage: str):
    return next(b for b in at.button if b.label == "Save" and stage in b.form_id)


def test_an_edited_json_stage_is_validated_and_sent(backend):
    at = app_with_run(stopped("interviews"))
    edited = factories.interviews(persona_count=1, tag="by hand")
    json_editor(at).input(edited.model_dump_json(indent=2))
    save_button(at, "interviews").click().run()

    [(run_id, stage, output)] = backend.called("edit_stage")
    assert (run_id, stage, output) == (backend.run.id, "interviews", edited)
    assert not at.exception


@pytest.mark.parametrize("bad", ["{not json", '{"interviews": [{"persona_id": "P1"}]}', "[]"])
def test_invalid_json_is_reported_and_nothing_is_sent(backend, bad):
    at = app_with_run(stopped("interviews"))
    json_editor(at).input(bad)
    save_button(at, "interviews").click().run()

    assert at.error and "validation error" in at.error[0].value
    assert backend.called("edit_stage") == []


def test_saving_an_unchanged_table_stage_sends_the_same_output(backend):
    at = app_with_run(stopped("questions"))
    save_button(at, "assumptions").click().run()

    [(_, stage, output)] = backend.called("edit_stage")
    assert stage == "assumptions" and output == factories.assumptions(tag="idea")


# ---- chat -------------------------------------------------------------------------------------------------------


def test_chat_sends_the_message_and_keeps_the_conversation(backend):
    at = app_with_run(stopped("personas"))
    at.chat_input[0].set_value("make P1 older").run()
    at.chat_input[0].set_value("and P2 younger").run()

    assert [m for (m,) in [c[1:] for c in backend.called("chat")]] == ["make P1 older", "and P2 younger"]
    assert at.session_state.chat == [
        ("user", "make P1 older"),
        ("assistant", "echo: make P1 older"),
        ("user", "and P2 younger"),
        ("assistant", "echo: and P2 younger"),
    ]
    assert [m.name for m in at.chat_message][-4:] == ["user", "assistant", "user", "assistant"]
