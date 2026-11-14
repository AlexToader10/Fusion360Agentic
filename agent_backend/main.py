from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_backend.edit_agent.agent import EditAgent
from agent_backend.runs import InvalidAction, RunManager, RunNotFound
from api_common.assets.common import STAGE_OUTPUTS, ChatRequest, ChatResponse, PipelineRequest, Run

manager = RunManager()
edit_agent = EditAgent(manager)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    manager.pipeline.shutdown()


app = FastAPI(title="Business validation pipeline", lifespan=lifespan)


@app.exception_handler(RunNotFound)
def run_not_found(_: Request, error: RunNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(InvalidAction)
def invalid_action(_: Request, error: InvalidAction) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.post("/runs")
def start_run(request: PipelineRequest) -> Run:
    return manager.start(request)


@app.get("/runs/{run_id}")
def get_run(run_id: UUID) -> Run:
    return manager.get(run_id)


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: UUID) -> Run:
    return manager.stop(run_id)


@app.post("/runs/{run_id}/resume")
def resume_run(run_id: UUID) -> Run:
    return manager.resume(run_id)


@app.post("/runs/{run_id}/chat")
def chat(run_id: UUID, request: ChatRequest) -> ChatResponse:
    manager.get(run_id)
    return ChatResponse(reply=edit_agent.chat(run_id, request.message), run=manager.get(run_id))


def add_edit_path(stage: str, output_model: type[BaseModel]) -> None:
    @app.put(f"/runs/{{run_id}}/{stage}", name=f"edit_{stage}")
    def edit_stage(run_id: UUID, output: output_model) -> Run:
        return manager.edit(run_id, stage, output)


for stage, output_model in STAGE_OUTPUTS.items():
    add_edit_path(stage, output_model)
