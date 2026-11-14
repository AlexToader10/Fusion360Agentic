# Business idea validation

Takes a business idea and a target audience and validates the idea through simulated customer interviews.

## Layout

- `agent_backend/` is the FastAPI service.
  - `pipeline.py` chains the stages. Each stage runs in its own process pool and its output is the input of the next one.
  - `stages/` holds one class per stage, each with a `run` method: assumptions, personas, questions (generator and judge loop), interviews (deep research persona and interviewer), validations, conclusions.
  - `runs.py` starts, stops, edits and resumes runs. The HTTP paths and the edit agent call the same methods.
  - `edit_agent/` is the Hermes agent behind the chat.
  - `openrouter_client.py` wraps the OpenRouter endpoint.
- `api_common/assets/` holds the pydantic models of every input and output.
- `api_common/client/` is the Python client of the backend.
- `frontend_demo/` is the Streamlit frontend. It only talks to the backend through the client.

## Setup

Hermes Agent cannot be installed from a wheel, so it is installed in editable mode from a local checkout:

```bash
git clone https://github.com/NousResearch/hermes-agent.git vendor/hermes-agent
git -C vendor/hermes-agent checkout c0ef5af5205dbdc25e9e44acd66873b4bb3503c9
uv sync --all-extras
cp .env.example .env   # then set OPENROUTER_API_KEY
```

`.env` selects the models. `OPENROUTER_MODEL` is the general model. `OPENROUTER_MODEL_<TASK>` overrides it for one task.

## Run

```bash
uv run uvicorn agent_backend.main:app --port 8000
uv run streamlit run frontend_demo/app.py
```

The frontend reads the backend address from `BACKEND_URL` (default `http://localhost:8000`).

## API

| Path | What it does |
| --- | --- |
| `POST /runs` | Start a run from a business idea and a target audience |
| `GET /runs/{id}` | Status of the run and the output of every finished stage |
| `POST /runs/{id}/stop` | Put the run on hold |
| `PUT /runs/{id}/{stage}` | Replace the output of a finished stage |
| `POST /runs/{id}/resume` | Continue from the first stage without an output |
| `POST /runs/{id}/chat` | Ask the edit agent to change a finished stage |

Editing a stage clears the stages after it, so a resumed run continues right after the earliest edit.
