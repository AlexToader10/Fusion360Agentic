# Tests

The suite checks the business-idea-validation pipeline: assumptions, personas, questions with a bias judge,
simulated interviews, per-interview verdicts and conclusions.

## Running

```bash
# Offline contract and state-machine suite. Fast, free, no network.
.venv/bin/python -m pytest

# Real OpenRouter model tests. These cost money and results vary from run to run.
RUN_LIVE_TESTS=1 .venv/bin/python -m pytest -m live

# Real model tests, skipping the ones that use OpenRouter web search.
RUN_LIVE_TESTS=1 .venv/bin/python -m pytest -m "live and not web"

# A single theme, e.g. the state machine
.venv/bin/python -m pytest -m e2e
```

`pytest` alone is the offline suite. Tests marked `live` or `web` are skipped unless `RUN_LIVE_TESTS=1` is set
and `OPENROUTER_API_KEY` is available. The skip message says which of the two is missing.

## Offline guarantees

- Tests that are not marked `live` or `web` cannot open network connections. Any attempt raises a `RuntimeError`.
  A test that only needs localhost can request the `allow_network` fixture to opt out.
- These tests see `OPENROUTER_API_KEY=offline-test-key` and `OPENROUTER_MODEL=offline/test-model`. They never
  use your real key.
- Stage tests replace the model client with `FakeOpenRouterClient` (see `tests/helpers.py`).

## API key and model selection

Live tests read the same settings as the app:

1. Copy `.env.example` to `.env` (git-ignored) and set `OPENROUTER_API_KEY`. Do not put the key in any other
   file. `conftest.py` loads `.env` with `python-dotenv`. Variables already in the environment take precedence.
2. `OPENROUTER_MODEL` is the general model. `OPENROUTER_MODEL_<TASK>` overrides it for one task, for example
   `OPENROUTER_MODEL_PERSONA` or `OPENROUTER_MODEL_QUESTIONS_JUDGE`. Set these in `.env` or the environment to
   test a different model for one stage.
3. `RUN_LIVE_TESTS=1` can also go in `.env`. Remove it from `.env` to keep live tests off by default.

**Warning: live tests call the real models.** Each run spends OpenRouter credits, and interview stages make many
calls per persona. Model output is stochastic, so live assertions check structure and grounding, not exact
wording. A failing live test can be a bad sample rather than a regression. Rerun it before you act on it.

## Fixtures

Defined in `tests/conftest.py`:

| Fixture | What it is |
| --- | --- |
| `sample_request` | `PipelineRequest` for a meal-kit idea aimed at dual-income parents, with `persona_count=3` |
| `sample_assumptions` | Six assumptions A1 to A6: problem, customer, solution, willingness to pay, channel, retention |
| `sample_personas` | Personas P1 to P3, all referencing valid assumption ids |
| `sample_questions` | Questions Q1 to Q6. Every assumption is tested by at least one question |
| `sample_interviews` | Interviews for P1 and P2, using only ids from the fixtures above |
| `sample_validations` | One verdict per assumption for each of P1 and P2 |
| `sample_conclusions` | One conclusion per assumption A1 to A6 |
| `full_state` | `PipelineState` with every stage filled from the fixtures above |
| `empty_state` | `PipelineState` with only the request |
| `allow_network` | Opt-out from the offline network block |

Fixtures are function-scoped, so a test can mutate its own copy.

## Helpers

`tests/helpers.py` is dependency-free. Import it with `from tests.helpers import ...`.

- Factories: `make_assumption`, `make_persona`, `make_question`, `make_exchange`, `make_interview`,
  `make_validation`, `make_conclusion`. Each takes its id positionally and extra fields as keyword overrides.
- Id checks: `assert_unique_ids(items, attr="id")`, `assert_sequential_ids(items, prefix)`, `extract_ids(text)`.
- Text metrics: `sentence_count`, `token_set`, `jaccard`, `contains_url`.
- Fakes: `FakeOpenRouterClient` queues responses (or exceptions to raise) and records every call in `.calls`.
  `install_fake_client(monkeypatch, "agent_backend.stages.<stage>", fake)` patches a stage module.
  `DeepResearchHarness` in `deep_research.py` builds its own client, so patch that module too.
- Live gating: `live_skip_reason()`, and `require_live()` for skipping from inside a test.

## Markers

| Marker | Meaning |
| --- | --- |
| `live` | Calls the real models. Needs `RUN_LIVE_TESTS=1` and `OPENROUTER_API_KEY` |
| `web` | Live test that also uses OpenRouter web search. Also needs the live gate |
| `slow` | Takes several seconds or more, even offline |
| `e2e` | Drives the whole pipeline from request to conclusions |

`pytest-timeout` caps each test at 600 seconds (`timeout` in `pyproject.toml`). Install the dev group with
`uv sync --extra dev`, or `pip install -e ".[dev]"`.

## Test map

Each theme has its own module.

Offline, no network, no credits:

- **State machine** (`test_state_machine.py`): run statuses, stop, resume, edit, event ordering, and the cascade that clears later stages.
- **Stage choreography** (`test_stage_choreography.py`): prompt assembly, judge-loop bounds, interview question picks, validation prompts, token forwarding.
- **API contract** (`test_api_contract.py`): HTTP paths, status codes, WebSocket history and reconnect, the Python client round trip.
- **Event stream** (`test_event_stream.py`): the Streamlit background reader, reconnect, terminal states, sequence numbers, bounded event log.

Live, marked `live` (some also `web`, `slow` or `e2e`):

- **Assumption quality:** falsifiable statements, five-category coverage, cross-idea diversity.
- **Persona ecology:** ids, segment coverage, diversity, no product leakage, exact-count boundaries.
- **Question inquisition:** judge recall/restraint on handcrafted leading questions, generated-question quality and coverage.
- **Interview character:** persona fidelity, anti-sycophancy, not claiming a nonexistent product.
- **Research grounding:** web research has sources and constraints, and persona answers stay grounded in it.
- **Verdict grounding:** each verdict cites the interview, and empty interviews stay inconclusive.
- **Conclusion calibration:** doom, euphoria, split and void scenarios; no sugar-coating simulated evidence.
- **Red team:** prompt injection through the idea, audience and business plan; absurd ideas; canaries.
- **End-to-end:** the full chain with two personas, referential integrity and final candour.
- **Stability:** paraphrase invariance and counterfactual sensitivity.
- **Counterfactual:** the same product aimed at dental practices vs tattoo studios.
- **Multilingual:** Spanish, German and emoji briefs keep their context and schema.
- **Bias bazaar:** a larger 20-question judge benchmark with a confusion matrix.
- **Interviewer policy:** picks relevant questions, ends on an irrelevant queue, never invents ids.
