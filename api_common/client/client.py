from uuid import UUID

import httpx
from pydantic import BaseModel

from api_common.assets.common import ChatRequest, ChatResponse, PipelineRequest, Run


class BackendError(Exception):
    pass


class BackendClient:
    """Python client for the agent backend."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def start_run(self, request: PipelineRequest) -> Run:
        return Run.model_validate(self._send("POST", "/runs", request))

    def get_run(self, run_id: UUID) -> Run:
        return Run.model_validate(self._send("GET", f"/runs/{run_id}"))

    def stop_run(self, run_id: UUID) -> Run:
        return Run.model_validate(self._send("POST", f"/runs/{run_id}/stop"))

    def resume_run(self, run_id: UUID) -> Run:
        return Run.model_validate(self._send("POST", f"/runs/{run_id}/resume"))

    def edit_stage(self, run_id: UUID, stage: str, output: BaseModel) -> Run:
        return Run.model_validate(self._send("PUT", f"/runs/{run_id}/{stage}", output))

    def chat(self, run_id: UUID, message: str) -> ChatResponse:
        return ChatResponse.model_validate(self._send("POST", f"/runs/{run_id}/chat", ChatRequest(message=message)))

    def _send(self, method: str, path: str, body: BaseModel | None = None) -> dict:
        response = self.http.request(method, path, json=body and body.model_dump(mode="json"))
        if response.is_error:
            raise BackendError(response.json().get("detail", response.text))
        return response.json()
