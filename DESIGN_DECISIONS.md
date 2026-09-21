# Design decisions

## Continue the existing work
The original three executed notebooks are preserved byte-for-byte in `notebooks/archive/`.
Their datasets and splits are reused by the corrected module-backed notebooks. No historical
metric is described as a new measurement. `reports/historical_metrics.json` identifies the source commit.

## Three specialists, one external agent
DistilBERT A supplies intent probabilities; DistilBERT B extracts a literal span from retrieved
course documentation; SmolLM2-135M-Instruct with PEFT supplies troubleshooting and synthesis.
All are internal to one LangGraph workflow exposed as `tuwaiq-tech-support-agent`.
No OpenAI model or inference service is called. Langfuse's SDK may install an OpenAI dependency;
that is not an inference integration.

## Taxonomy and overlap
Authentication concerns identity and authorization; network concerns transport; deployment
concerns packaging/startup; database concerns queries/pools; GPU concerns devices/memory;
API concerns HTTP/schema; package concerns dependency versions; general covers concepts.
The original taxonomy overlaps on 401, 503 and deployment/database symptoms. Preserve the
original labels for comparability, record confusion/error cases, and do not present three examples
per class as evidence of robust production generalization.

## Hybrid routing
Safety rules run first in all strategies. Router A then uses the classifier. Router B uses
SmolLM2 JSON output validated against an exact schema. Hybrid uses A above a threshold and
B below it. Invalid JSON/model failure escalates rather than executing an unvalidated action.
Classifier thresholds are chosen only on validation: maximize coverage subject to >=90% accepted
accuracy and >=50% coverage. If none qualify, use threshold 1.0 and report the policy unqualified.
This does not lower the model's macro-F1/recall gate. Hybrid is an architectural choice to reduce
LLM calls; it is not claimed to outperform alternatives until real router measurements support it.

## Grounding
Model B keeps 384-token windows, stride 96, sample mapping and context-only character offsets.
It never invents answer text outside the context. Lexical retrieval searches a small course KB;
this is not live infrastructure documentation. Missing retrieval results return an evidence refusal.
Extractive span scoring is not calibrated answerability confidence; SQuAD2-style no-answer
training and retrieval evaluation remain useful future work.

## Model C training
The published historical adapter's base is 360M, incompatible with the required 135M base.
It is retained as historical work, not silently loaded into a different model. The corrected
135M run uses r=16, alpha=32, dropout=.05 on q_proj/v_proj, NF4 only with CUDA and bitsandbytes,
standard LoRA otherwise. Five epochs, effective batch 16, cosine schedule, warmup, clipping
and best validation loss follow the requested configuration.
Explicit labels mask system/user/header/padding tokens with -100. Target tokens alone contribute
to causal loss. The separate nonquantized baseline is measured and freed before creating the
fresh quantized base and attaching PEFT. Trainer is never created around a bare quantized baseline.

## Evaluation integrity
Original Golden Set prompts are fixed. G04's over-escaped numbered-list regex is corrected;
G07 preserves the already-present `should not` refusal acceptance. Historical outputs are not
relabelled as passing. The Golden Set is a development regression set, not independent testing:
original conversations include exact and near overlaps. Exact training overlaps are removed;
near-duplicate bias is documented. After historical test inspection, future changes need a new
external test set for unbiased claims. Neither generated fixture responses nor unit-test success
can satisfy model gates.

## Tools, safety and side effects
All 13 contracts reject unexpected fields. Twelve have local implementations; web_search returns
an explicit unconfigured error. SQLite supports durable tickets and escalation evidence. SQL uses
a read-only connection, query_only, an authorizer, row and instruction limits. The graph invokes
read-only diagnostic tools automatically. Ticket creation is available through its typed contract
but is not inferred automatically from arbitrary conversation. Escalation records are local and
do not claim a human was notified. Health results are always labelled deterministic mocks.

## Release, API and observability
Production startup requires reports satisfying all model gates and matching local artifact hashes.
Demo mode uses named deterministic fixtures and never counts as model acceptance. /health reports
503 when production cannot initialize. Requests have a trace ID, bounded context/concurrency,
optional bearer authentication, structured errors and greedy generation. Token usage is omitted
instead of fabricated as zero. Client system messages cannot override the service's safety prompt.
Langfuse spans wrap requests/routes/models/tools with redaction; raw LangGraph callbacks are
avoided because they can upload unredacted state. Redaction is pattern-based and not a guarantee
against arbitrary sensitive data; connect only approved data sources.

## Deployment
Docker serves only the unified agent. Open WebUI has one backend connection; Ollama/direct
connections and persisted provider overrides are disabled. SQLite avoids an unnecessary database
service and is appropriate for this single-worker demo. Multiple workers/replicas need a shared
transactional backend and tested concurrency. Pin Open WebUI to a tested digest before deployment.

## Continuation: explicit normalization and failure handling
Hard rules precede ML because destructive actions and high-risk incidents must not depend
on classifier confidence or generated JSON. Hybrid routing retains the existing threshold;
low confidence or invalid classifier output falls back to the schema-validated LLM router.
The known Model A artifact/calibration issue (including the .207-confidence regression case)
is not repaired by lowering that threshold. Historical reports are evidence, not new measurements.

Intent normalization maps a small documented allowlist of spelling variants (DB, auth,
GPU issue, general-question) to existing labels before route selection. Unknown labels fail
validation. This preserves canonical routing and avoids arbitrary fuzzy intent guesses.

Structured ToolResult outputs distinguish success, failure, backend and evidence so synthesis
can disclose fixture data and the graph can escalate on failed diagnostics. The selector preserves
the diagnostic sequence and adds explicit calculator, installed-package and ticket-search requests.
It never infers ticket creation or SQL execution. Deterministic mocks make CPU-only tests repeatable;
they verify contracts and control flow, never model quality or live infrastructure health.

Graph node exceptions now take the escalation branch without exposing exception text. If local
escalation storage fails, the answer explicitly asks the user to contact their support owner.
