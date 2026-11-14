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
  - `views.py` draws the output of each finished stage and `editors.py` the form fields that edit it, `live.py` the stage that is running, `edit.py` the edit form around those fields and the chat.
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

After the run starts, the page is one document that grows as the stages finish. The sidebar lists the six stages with their progress, and each finished one links to its section. The running stage shows a live trace: status lines from the backend and, during interviews, the persona's answer as it is generated. Stop run sits under that trace and in the sidebar. Stopping holds the run at the current stage; every finished stage has an Edit button that turns its section into a form, with no JSON or ids to type: text fields for the wording, a picker for the assumptions a persona or question is about, three choices for a verdict, and buttons that add or remove assumptions, personas and questions. Saving an edit clears the later stages, and Resume run regenerates them from that point. The chat for asking the agent for edits stays at the bottom of the page.

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

## Research: evidence for LLM personas

The pipeline interviews simulated customers, so we looked for published work on whether using LLMs and agents as personas is a good idea (November 2026). There is a solid body of work in favour, but the support is uneven:

- **Simulating people has the strongest evidence**, including a 2026 Nature paper. Every academic source frames it as a way to pilot and screen before research with real people, and the accuracy comes from grounding the agents in real data about real people.
- **Role-specialised agents show large gains in multi-agent coding papers**, but only two isolate the effect of the role itself. Elsewhere the gain is mixed with workflow structure, tools and extra tokens.
- **Assigning a role to improve accuracy is the weakest case.** Detailed, task-matched personas helped in some 2023 studies; generic "you are an expert" lines did not, and later studies found no reliable gain.

A run of this pipeline is therefore best read as a pilot: it shows which assumptions to take to real customers first.

### Simulating people (claims checked against the source)

| Source | What it argues | Evidence | Caveat |
| --- | --- | --- | --- |
| [Park et al., "Generative Agent Simulations of 1,000 People"](https://arxiv.org/abs/2411.10109) (arXiv 2024, revised June 2026) | Agents built from two-hour interviews can stand in for specific individuals | Agents of 1,052 US adults reached 83% of the participants' own two-week retest consistency on unseen survey items (86% with interview plus survey data) | A short persona paragraph reached 71% and demographics alone 74%, so the gain comes from rich real data |
| [Ashokkumar, Hewitt, Ghezae & Willer, "Large language models can predict the results of social science experiments"](https://www.nature.com/articles/s41586-026-10742-x) (Nature, July 2026) | Simulated samples can predict experiment outcomes and supplement pilot testing | Across 70 preregistered US survey experiments (469 effects), effects in GPT-4 simulated samples correlated strongly with the real ones, on par with aggregated human forecasts | Effect sizes were overestimated about twofold, and accuracy was weaker for field experiments |
| [Argyle et al., "Out of One, Many"](https://arxiv.org/abs/2209.06899) (Political Analysis, 2023) | "Silicon samples" conditioned on real respondents' profiles can substitute for survey respondents | GPT-3 reproduced reported presidential vote choice, with correlations of 0.90, 0.92 and 0.94 for 2012, 2016 and 2020; Study 1 cost $29 | The profiles included party and ideology, which already predict the vote; US politics only |
| [Horton, Filippas & Manning, "Homo Silicus"](https://arxiv.org/abs/2301.07543) (NBER working paper 2023, revised 2026) | LLMs can be given preferences and simulated as economic agents, quickly and cheaply | Five classic experiments gave qualitatively similar results; a calibrated mixture of three personas halved the prediction error of persona-free models | The authors say results must be confirmed with humans, and they have a financial interest in simulation tools |
| [Aher, Arriaga & Kalai, "Using LLMs to Simulate Multiple Humans"](https://arxiv.org/abs/2208.10264) (ICML 2023) | Simulated participants can replicate classic human-subject studies | Three of four experiments replicated (Ultimatum Game, garden-path sentences, Milgram) | The fourth failed: newer models gave inhumanly accurate answers |
| [Anthis et al., "LLM Social Simulations Are a Promising Research Method"](https://arxiv.org/abs/2504.02234) (ICML 2025) | Simulations can already be used for pilot and exploratory studies | Review of 53 papers comparing LLM data with human data | Position paper; the "tens of dollars per experiment" figure is an estimate |

### Giving a model a role (claims checked against the source)

| Source | What it argues | Evidence | Caveat |
| --- | --- | --- | --- |
| [Kong et al., "Better Zero-Shot Reasoning with Role-Play Prompting"](https://arxiv.org/abs/2308.07702) (NAACL 2024) | A task-relevant role improves reasoning | Beat plain zero-shot on 10 of 12 benchmarks with gpt-3.5-turbo (AQuA from 53.5% to 63.8%) | Lost to "let's think step by step" on 3 of 12; a persona described as incompetent scored below no persona |
| [Xu et al., "ExpertPrompting"](https://arxiv.org/abs/2305.14688) (arXiv preprint, 2023) | A detailed, automatically generated expert identity improves answers | A GPT-4 judge preferred the expert-persona answers 48.5% of the time, against 23.0% for plain prompts | A generic one-line expert role did not reproduce the gain; not peer reviewed |
| [Salewski et al., "In-Context Impersonation"](https://arxiv.org/abs/2305.14930) (NeurIPS 2023) | Impersonating a task expert beats impersonating a non-expert | On 57 MMLU tasks, accuracy ranked task expert, then same-domain expert, then off-domain expert | Measured against off-domain personas, not against no persona |
| [Anthropic, "Give Claude a role"](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) (prompting documentation) | A role in the system prompt focuses behaviour and tone | None; a recommendation without a benchmark | Argues for tone and focus, not accuracy |

### Role-specialised agents (read once, not cross-checked)

| Source | What it argues | Evidence | Caveat |
| --- | --- | --- | --- |
| [ChatDev](https://arxiv.org/abs/2307.07924) (Qian et al., ACL 2024) | Roles such as programmer, reviewer and tester produce better software | Removing the role descriptions was the most damaging ablation: quality fell from 0.395 to 0.221 | About 3.2 times the tokens of a single agent; no significance test |
| [MetaGPT](https://arxiv.org/abs/2308.00352) (Hong et al., ICLR 2024) | Role specialisation lets complex work be decomposed | An engineer-only agent scored 1.0 of 4 on executability; the four-role team scored 4.0 | Two tasks only; the authors say role play alone is not enough and credit the structured workflows |
| [Dong et al., "Self-collaboration Code Generation via ChatGPT"](https://arxiv.org/abs/2304.07590) (ACM TOSEM 2024) | A team of analyst, coder and tester beats a single agent | HumanEval pass rate rose from 57.3 to 74.4; removing only the role wording cost about 4 points | More roles was not always better |
| [CAMEL](https://arxiv.org/abs/2303.17760) (Li et al., NeurIPS 2023) | Two agents with assigned roles can finish tasks with minimal human input | Human raters preferred the pair's solution 76.3% of the time, against 10.4% for a single-shot answer | No ablation isolates the roles; the agents often flipped roles |
| [AutoGen](https://arxiv.org/abs/2308.08155) (Wu et al., 2023) | Splitting tasks between specialised agents improves results | Separating code writing from safety checking raised F1 on unsafe-code detection by 8 points (GPT-4) and 35 (GPT-3.5) | The authors found role play without tools or code execution did not work |

### Industry and practitioners (read once, not cross-checked)

| Source | What it argues | Evidence | Caveat |
| --- | --- | --- | --- |
| [Bain, "Synthetic Customers Earn Their Stripes"](https://www.bain.com/insights/synthetic-customers-earn-their-stripes/) (May 2026) | Synthetic customers already support product and pricing decisions | 1,500 digital twins replicated about 90% of the key outcomes of a conjoint study with people | Reported by the firm that sells this work; it says accuracy needs proprietary data |
| [PyMC Labs and Colgate-Palmolive, "LLMs Reproduce Human Purchase Intent"](https://arxiv.org/abs/2510.08338) (arXiv, Oct 2025) | Synthetic consumers are a cheap first filter for product concepts | Across 57 surveys, product rankings reached about 90% of human retest reliability, and about 50% without demographic personas | Calibrated on the same 57 surveys; missed gender, region and ethnicity patterns |
| [EY with Aaru, "How AI simulation accelerates growth"](https://www.ey.com/en_us/insights/wealth-asset-management/how-ai-simulation-accelerates-growth-in-wealth-and-asset-management) (Oct 2025) | Persona simulation is faster and cheaper than panels | Recreated a 3,600-investor survey in a day; median correlation of 0.90 across 53 questions | Promotional material; one theme correlated negatively |
| [Walmart Global Tech, "AI Personas for UX Research"](https://ceur-ws.org/Vol-4249/paper2.pdf) (CHI'26 workshop) | AI personas surface usability problems before testing with people | 16 personas found the same main themes and defects as testing with people | Qualitative match only; the personas were built from the same human data |
| [NN/g, "Synthetic Users: If, When, and How"](https://www.nngroup.com/articles/synthetic-users/) (June 2024) | Useful for desk research and generating hypotheses | A simulated sales rep's account of a working day closely matched real interviews | Synthetic users idealised their behaviour and praised the concepts; not for final decisions |

### Counter-evidence (read once, not cross-checked)

- **[Zheng et al.](https://arxiv.org/abs/2311.10054) (Findings of EMNLP 2024):** none of 162 one-line roles beat no persona on 2,410 factual questions, across four open model families.
- **[Wharton Generative AI Labs](https://gail.wharton.upenn.edu/research-and-insights/playing-pretend-expert-personas/) (Dec 2025):** expert personas did not consistently improve factual accuracy across six models.
- **["Principled Personas"](https://arxiv.org/abs/2508.19764) (2025):** mixed results. Generated, task-specific expert personas gave significant gains, hand-written ones did not, and irrelevant persona details cut accuracy by up to almost 30 points.
- **[Bisbee et al.](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE) (Political Analysis, 2024):** ChatGPT with personas reproduced survey averages, but 48% of regression coefficients differed significantly and the answers varied less than real ones.
- **[Wang, Morgenstern & Dickerson](https://arxiv.org/abs/2402.01908) (Nature Machine Intelligence, 2025):** answers from demographic personas misrepresent and flatten identity groups.
- **[Gupta et al.](https://arxiv.org/abs/2311.04892) (ICLR 2024):** demographic personas degraded reasoning accuracy.

### How far to trust this

- **Checked tables:** 23 claims from those 10 sources were each compared with the primary source by independent reviewer agents trying to refute them; none was refuted.
- **Other tables and the counter-evidence:** each source was read and summarised by a single agent, without a second check.
- **Not read:** the [HBR article on generative AI in early-stage market research](https://hbr.org/2025/07/using-gen-ai-for-early-stage-market-research) (July 2025) came up in the search but could not be read.