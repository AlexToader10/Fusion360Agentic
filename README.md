# Business idea validation

Takes a business idea and a target audience and validates the idea through simulated customer interviews.

## Layout

- `agent_backend/` is the FastAPI service.
  - `pipeline.py` chains the stages. Each stage runs in its own process pool and its output is the input of the next one. The stages report their progress to the main process through a queue.
  - `events.py` is the live reporter and event log behind the WebSocket.
  - `stages/` holds one class per stage, each with a `run` method: assumptions, personas, questions (generator and judge loop), interviews (deep research persona and interviewer), validations, conclusions.
  - `runs.py` starts, stops, edits and resumes runs. The HTTP paths, the WebSocket and the edit agent call the same methods.
  - `edit_agent/` is the Hermes agent behind the chat.
  - `openrouter_client.py` wraps the OpenRouter endpoint. Text calls can forward token deltas.
- `api_common/assets/` holds the pydantic models of every input and output.
- `api_common/client/` is the Python client of the backend. `events_url()` returns the WebSocket address.
- `frontend_demo/` is the Streamlit frontend. It only talks to the backend through the client.
  - `app.py` picks the screen: `start.py` before a run, `run.py` during and after one.
  - `views.py` draws the output of each finished stage, `live.py` the stage that is running, `edit.py` the edit forms and the chat.
  - `stream.py` reads the WebSocket in a background thread so the Stop button stays clickable while tokens arrive.
  - `session.py` holds the backend client and the run of the browser session.
  - The look lives in `.streamlit/config.toml` (colors, fonts, corners) and `style.css` (what the theme cannot express).

## Setup

```bash
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

## Deploy

```bash
./deploy.sh
```

Builds both images and starts the backend on port 8000 and the frontend on port 8501 with Docker Compose. It needs `.env`. `docker compose down` stops them.

## What the screen does

The start screen has two ways in, side by side:

- **Left:** business idea, target audience and additional info.
- **Right:** upload the business plan or market-analysis PDF you already have. The extracted text becomes the brief, so the fields can stay empty.

The additional info field is also where the size of the run goes, in plain words, for example "5 assumptions, 3 personas and 6 questions". Without it the model picks the sizes.

After the run starts, the page is one document that grows as the stages finish. The sidebar lists the six stages with their progress, and each finished one links to its section. The running stage shows a live trace: status lines from the backend and, during interviews, the persona's answer as it is generated. Stop run sits under that trace and in the sidebar. Stopping holds the run at the current stage; every finished stage has an inline editor; saving an edit clears the later stages, and Resume run regenerates them from that point. The chat for asking the agent for edits stays at the bottom of the page.

Validations are one table with a row per assumption and a column per interview. A verdict is a shape as well as a color (check, cross or empty ring), and hovering one shows its evidence.

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

## Tests

```bash
pytest                                  # offline contract/state-machine suite, no network, no credits
RUN_LIVE_TESTS=1 pytest -m live         # real OpenRouter model tests, costs credits, stochastic
```

See `tests/README.md` for the fixtures, markers, live gate and the full test map.