# Multi-Model Agentic Technical Support

One technical-support agent routes requests to an intent classifier, extractive QA, a LoRA support
specialist, diagnostic tools, or human escalation. The API is OpenAI-compatible in shape;
**no OpenAI models are called**.

**Status:** application implementation and deterministic tests are available. Production is gated.
The original notebook model gates failed; see `reports/historical_metrics.json` and
`reports/IMPLEMENTATION_STATUS.md` for measured status and remaining blockers. Demo mode does
not establish that any trained model passed.

## Architecture

```mermaid
flowchart LR
  UI[Open WebUI: one model] --> API[FastAPI /v1/chat/completions]
  API --> R[LangGraph route_node]
  R --> S[Safety rules]
  S --> A[Model A confidence]
  A -->|ambiguous| L[Small LLM JSON router]
  R --> Q[qa_node: KB retrieval + Model B]
  R --> T[tools_node: typed diagnostic tools]
  T --> C[support_node: Model C synthesis]
  R --> C
  R --> E[escalation_node: durable local evidence]
  API -. redacted spans .-> LF[Optional Langfuse]
```

## Existing work and files

The three original executed notebooks are preserved unchanged in `notebooks/archive/`.
Corrected notebooks in `notebooks/` call shared modules in `src/training/`; this keeps training,
inference, and tests aligned. The 192 original intent examples, 48 QA pairs, and 80 conversations
are extracted into `data/`. The fixed ten Golden Set prompts are preserved. Eighty new paired
grounding/safety/formatting examples are training-only.

| Component | Implementation |
|---|---|
| Model A | DistilBERT classifier, dynamic padding, stratified 144/24/24 split, macro/per-class metrics, confusion/errors, validation threshold sweep |
| Model B | DistilBERT extractive QA, 384-token windows, stride 96, original document splits, literal spans, held-out EM/token F1 |
| Model C | SmolLM2-135M-Instruct, assistant-only labels, separate nonquantized baseline, NF4 QLoRA or LoRA, adapter reload, Golden Set |
| Routing | Rules + A; strict JSON LLM; hybrid safety → confident A → LLM |
| API | `/health`, `/v1/models`, `/v1/chat/completions`, structured errors, bounded concurrency, optional bearer key |
| Storage | SQLite tickets and escalation records persisted in a volume |
| Observability | Optional Langfuse request/route/model/tool spans with redacted inputs/outputs and durations |

## Local setup (PowerShell, Python 3.11 or 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
$env:AGENT_MODE = 'demo'
$env:AGENT_API_KEY = 'choose-a-long-random-key'
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, use the same key:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$headers = @{ Authorization = 'Bearer choose-a-long-random-key' }
Invoke-RestMethod http://127.0.0.1:8000/v1/models -Headers $headers
$body = @{ model='tuwaiq-tech-support-agent'; messages=@(@{role='user'; content='The database pool is exhausted'}) } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -Headers $headers -ContentType application/json -Body $body
```

Linux/macOS: activate with `source .venv/bin/activate`, use `export AGENT_MODE=demo` and
`export AGENT_API_KEY=...`; the Python commands are identical.
The native Python startup does not automatically read `.env`; export variables or use
`uvicorn src.api:app --env-file .env` after replacing Docker-specific paths with local paths.

## Training and evaluation

```powershell
python -m scripts.build_contrastive_data
python -m src.training.train_a
python -m src.training.train_b
python -m src.training.train_c
python -m src.evaluation.routers
python -m src.evaluation.release
$env:RUN_MODEL_TESTS = '1'
python -m pytest -q
```

These commands download base weights and train real models. CUDA is recommended; CPU uses
standard LoRA. QLoRA requires a compatible CUDA/bitsandbytes environment. Reports are generated
from actual outputs under `reports/`; never replace them with demo metrics. No upload is automatic.
The notebooks provide the same entry points for Colab.

Gates stay fixed: A macro F1 >= .80 and every recall >= .60; B EM >= .65 and token F1 >= .80;
C validation loss/perplexity improves and every required Golden Set case passes; all reload checks pass.
Production startup also compares hashes of local evaluated artifacts with release evidence.

The original test sets were already inspected in committed notebooks. New model changes therefore
need a new external test set before making unbiased generalization claims. Golden Set prompts
overlap with original training themes; exact prompt overlaps are excluded from new training, and
remaining near-duplicate bias is documented. Keyword Golden checks are necessary regression
checks, not comprehensive safety guarantees.

### Model sources

All IDs are centralized in `src/config.py`. A/B prefer locally trained model directories,
then the original verified public repositories:

- `Lammem310/multi-model-support-intent-classifier`
- `Lammem310/multi-model-support-extractive-qa`
- Historical C: `Lammem310/multi-model-support-specialist-lora` uses **360M**, so it is deliberately
  incompatible with the required **135M** production path.

Set `MODEL_A_PATH`, `MODEL_B_PATH`, `MODEL_C_ADAPTER` to local directories or Hub IDs for
evaluation. Production requires local snapshots so hashes bind the weights/tokenizers to the
evaluated release. New Model C is saved to `models/support_adapter`; no unrelated model is substituted.
Public downloads require no token. For private repositories use `HF_TOKEN` through environment
variables or Colab Secrets. Never place it in a notebook, `.env.example`, or Git.

### Router comparison

`python -m src.evaluation.routers` measures real A/B/Hybrid routing on the fixed `data/router_test.json`.
It writes accuracy, macro F1, latency, invalid JSON rate, call counts and per-case failure details.
`--demo` explicitly measures fixture plumbing and writes a separate `router_demo.json`.
The threshold sweep uses validation only. If no candidate satisfies the acceptance/coverage policy,
the result says so rather than inventing a successful threshold.

## Tools

All inputs are typed, unexpected fields are rejected, and outputs follow `ToolResult`.

| Tool | Backend and behavior |
|---|---|
| knowledge_base_search, documentation_search | Local lexical search over course KB |
| ticket_search, ticket_create | Real SQLite persistence; parameterized queries |
| system_health_check | Deterministic mock; always marked `is_live=false` |
| log_analyzer | Deterministic diagnostic pattern analysis |
| package_lookup | Actual installed agent package metadata |
| sql_query | Read-only SQLite, allowlisted table, bounded execution/rows |
| calculator | Bounded arithmetic AST; no arbitrary evaluation |
| file_search | Search bundled KB index only |
| web_search | Explicit unconfigured error; no fabricated live results |
| escalate_to_human | Persist reason/evidence locally; explicitly does not notify a person |
| diagnostic_runbook | Read-only guidance; no commands executed |

The graph automatically uses retrieval or read-only diagnostic tools. Ticket creation and SQL
are available through typed contracts, not arbitrary model-generated code execution.

## Langfuse

Export `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` to enable it.
With no keys it is optional and inactive. Trace spans include request input, routing, intent,
confidence, model/tool inputs and outputs, answer, duration and error class. Patterns redact
credential fields, bearer values and common token formats. Do not send sensitive production data
without reviewing the redaction policy. See the [Langfuse SDK reference](https://python.reference.langfuse.com/langfuse).

## Docker, Open WebUI and Dokploy

```powershell
Copy-Item .env.example .env
# Edit .env: set strong AGENT_API_KEY and WEBUI_SECRET_KEY.
New-Item -ItemType Directory -Force models
docker compose config
docker compose build
docker compose up -d
```

Open WebUI is at `http://localhost:3000`; create the first administrator account locally.
Only `http://support-agent:8000/v1` is configured. In the model's advanced parameters turn
**Stream Chat Response off** (the core API is non-streaming). Select `tuwaiq-tech-support-agent`.
Do not add specialist models as providers. Review the [Open WebUI environment reference](https://docs.openwebui.com/reference/env-configuration/).

For production and exact Dokploy steps, see `docs/DEPLOYMENT.md`.

## Limitations

- Historical model quality gates fail; passing software tests is not model acceptance.
- Course datasets and routing test set are small; test/Golden overlap limits external validity.
- Health is mocked, web search is unconfigured, escalation is local only.
- QA lacks calibrated no-answer detection; lexical retrieval is limited.
- Greedy generation, one concurrent request and bounded context; no streaming, vision, tool-call API,
  external ticket integration or distributed state.
- Docker/Dokploy/Open WebUI and live Langfuse verification require their respective runtime/access.

## Continuation validation and routing details
See `reports/CONTINUATION_REPORT.md` for the latest executed checks and file changes.
The read-only selector supports `calculate (2+3)*4`, `package pydantic`, and
`find tickets outage`, plus the existing diagnostic sequence. Alias normalization is an
explicit allowlist in `src/routing/normalization.py`. Unknown intents are rejected.
Graph failures lead to local escalation; storage failures return direct-contact guidance.

For a locally running API, Open WebUI's base URL is `http://127.0.0.1:8000/v1` when
both run on the same host. From a Docker Desktop WebUI container, use
`http://host.docker.internal:8000/v1` with a host binding reachable from that container.
The bundled Compose connection remains `http://support-agent:8000/v1`.
Set the WebUI API key to `AGENT_API_KEY`; when the API variable is empty authentication
is disabled (a client-required placeholder key is ignored). Compose requires a nonempty key.
Only `tuwaiq-tech-support-agent` is listed; specialist names are rejected.

## Windows test troubleshooting
Run from the repository root using its interpreter: `.\.venv\Scripts\python.exe -m pytest -q`.
Pytest discovers only the root `tests/` directory. A nested extracted repository can contain
identically named test modules; collecting both copies causes `import file mismatch` errors.
The nested copy is preserved, but it is not the authoritative test directory.

If pytest reports `PermissionError: [WinError 5]` for its Windows temporary directory,
choose a writable project-local temporary root in the current PowerShell session:

```powershell
New-Item -ItemType Directory -Force .cache/pytest-tmp | Out-Null
$env:PYTEST_DEBUG_TEMPROOT = (Resolve-Path .cache/pytest-tmp).Path
.\.venv\Scripts\python.exe -m pytest -q
```

Python 3.11.9 was exercised directly. Graph schemas use `typing_extensions.TypedDict`
for Pydantic compatibility before Python 3.12. Model imports remain lazy; software tests
do not download models. The Starlette/AnyIO BlockingPortal deprecation warning remains
upstream and does not fail the tests. Real model acceptance still requires RUN_MODEL_TESTS=1
and genuine evaluated artifacts; the default skip remains intentional.
