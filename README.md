# Business idea validation

Takes a business idea and a target audience and validates the idea through simulated customer interviews.

## Layout

- `agent_backend/` is the FastAPI service.
  - `pipeline.py` chains the stages. Each stage runs in its own thread pool and its output is the input of the next one.
  - `events.py` is the live reporter and event log behind the WebSocket.
  - `stages/` holds one class per stage, each with a `run` method: assumptions, personas, questions (generator and judge loop), interviews (deep research persona and interviewer), validations, conclusions.
  - `runs.py` starts, stops, edits and resumes runs. The HTTP paths, the WebSocket and the edit agent call the same methods.
  - `edit_agent/` is the Hermes agent behind the chat.
  - `openrouter_client.py` wraps the OpenRouter endpoint. Text calls can forward token deltas.
- `api_common/assets/` holds the pydantic models of every input and output.
- `api_common/client/` is the Python client of the backend. `events_url()` returns the WebSocket address.
- `frontend_demo/` is the Streamlit frontend. It only talks to the backend through the client. `stream.py` reads the WebSocket in a background thread so the Stop button stays clickable while tokens arrive.

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

## What the screen does

The start screen has two ways in, side by side:

- **Left:** business idea, target audience, additional info, and three optional counts. Leave a count empty and the model picks the size; set it and the prompt asks for exactly that many assumptions, personas or questions.
- **Right:** upload the business plan or market-analysis PDF you already have. The extracted text becomes the brief, so the three fields can stay empty.

After the run starts, each stage appears as a chat message. The current stage shows a live `Thinking…` trace: status lines from the backend and, during interviews, the persona's answer as it is generated. A Stop button sits in the chat and in the sidebar. Stopping pauses at the current step; every finished stage has an inline editor; saving an edit clears the later stages, and Resume regenerates them from that point. The final conclusions appear in the same chat transcript.

## API

| Path | What it does |
| --- | --- |
| `POST /runs` | Start a run from a business idea and a target audience |
| `GET /runs/{id}` | Status of the run and the output of every finished stage |
| `POST /runs/{id}/stop` | Put the run on hold |
| `PUT /runs/{id}/{stage}` | Replace the output of a finished stage |
| `POST /runs/{id}/resume` | Continue from the first stage without an output |
| `POST /runs/{id}/chat` | Ask the edit agent to change a finished stage |
| `WS /runs/{id}/events?after=<seq>` | Stream stage events, token deltas and the run snapshot |

The WebSocket sends `stage_started`, `status`, `token`, `stage_completed`, `stage_failed`, `stage_edited`, `run_stopped`, `run_resumed`, `run_completed`, `run_failed` and `run_state` messages. Each event has a monotonic `seq`; reconnect with `?after=<last seq>` to resume without missing or repeating events. A `run_state` message carries the full `Run` snapshot, so a reconnecting frontend can render the finished stages immediately.

Editing a stage clears the stages after it, so a resumed run continues right after the earliest edit.

## Why Python

The heavy computation is not in this service. It is in the models behind OpenRouter's inference infrastructure; the Python layer only builds prompts, validates JSON and coordinates I/O-bound HTTP calls. Python is the right level of abstraction for that control plane, and the thread pool in `pipeline.py` keeps the network waits overlapping.

## Tests

```bash
pytest                                  # offline contract/state-machine suite, no network, no credits
RUN_LIVE_TESTS=1 pytest -m live         # real OpenRouter model tests, costs credits, stochastic
```

See `tests/README.md` for the fixtures, markers, live gate and the full test map.