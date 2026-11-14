"""The shared pydantic contract between backend, client and frontend."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from api_common.assets.common import STAGE_OUTPUTS, PipelineRequest, PipelineState, Run, RunStatus
from api_common.assets.validations import AssumptionVerdict
from tests import factories


def test_stage_outputs_are_the_state_fields_after_the_request_in_order():
    fields = list(PipelineState.model_fields)
    assert fields[0] == "request"
    assert fields[1:] == list(STAGE_OUTPUTS)
    for stage, model in STAGE_OUTPUTS.items():
        assert model in PipelineState.model_fields[stage].annotation.__args__


@pytest.mark.parametrize("last_stage", [None, *STAGE_OUTPUTS])
def test_a_run_at_any_point_survives_a_json_round_trip(last_stage):
    run = Run(id=uuid4(), status=RunStatus.STOPPED, error="x", state=factories.state_through(last_stage, "€$ ✓"))
    assert Run.model_validate_json(run.model_dump_json()) == run
    assert Run.model_validate(run.model_dump(mode="json")) == run


@pytest.mark.parametrize("count", [0, 31, -1])
def test_persona_count_bounds(count):
    with pytest.raises(ValidationError):
        PipelineRequest(business_idea="x", target_audience="y", persona_count=count)


def test_state_assignment_is_validated_and_coerced():
    state = factories.state_through(None)
    state.questions = factories.questions().model_dump()
    assert state.questions == factories.questions()
    with pytest.raises(ValidationError):
        state.questions = {"questions": [{"id": "Q1"}]}
    with pytest.raises(ValidationError):
        state.request = None


@pytest.mark.parametrize("verdict", ["valid", "VALIDATED", "", None])
def test_only_the_three_verdicts_are_accepted(verdict):
    with pytest.raises(ValidationError):
        AssumptionVerdict(assumption_id="A1", verdict=verdict, evidence="e")


def test_run_status_serialises_as_its_lowercase_value():
    assert Run(id=uuid4(), state=factories.state_through(None)).model_dump(mode="json")["status"] == "running"
    assert [s.value for s in RunStatus] == ["running", "stopped", "completed", "failed"]
