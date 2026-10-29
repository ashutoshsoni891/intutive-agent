# Incident Triage Agent

An autonomous agent that triages production incidents: it reads an alert, plans an investigation, calls tools (logs, metrics, deployments, config), forms a hypothesis, proposes a fix, waits for an operator to approve it, applies it, verifies recovery and reports. Runs are durable and resumable. An operator watches every step in a web UI and can intervene.

Python 3.11+, FastAPI, **LangGraph** (agent graph, checkpointing, human-in-the-loop interrupts) and **LangChain** (model and message abstractions), with any OpenAI-compatible open-source model server (Ollama by default). The graph is defined in [app/runtime.py](app/runtime.py).

## Screenshots

Captured in offline demo mode (`LLM_PROVIDER=mock`), which replays a scripted investigation through the real runtime, tools, approval gate and UI. Real-model runs look the same but take minutes per step on CPU.

| | |
|---|---|
| ![Start](assets/screenshots/01-start.png) **Start a run.** Pick a scenario; the alert the agent receives is shown. | ![Approval gate](assets/screenshots/02-approval-gate.png) **Approval gate.** The agent proposes a rollback; the run is parked until the operator decides. The plan, stats and live timeline (reasoning, tool calls, results, retry) are visible. |
| ![Completed](assets/screenshots/03-completed.png) **Completed.** Root cause, confidence, applied action and the note saved to long-term memory. | ![Injection](assets/screenshots/04-injection-ignored.png) **Prompt injection.** A log line carries an injected instruction; it is shown as inert data, the secret is redacted, and the agent's reasoning rejects it. |

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ (developed on 3.14) | Async-first, strong LLM ecosystem |
| Agent orchestration | LangGraph (`StateGraph`, `interrupt`, `Command`) | Explicit state machine, checkpointing, resumable human-in-the-loop |
| LLM interface | LangChain (`ChatOpenAI`, message types, `bind_tools`) | Provider-neutral tool calling |
| Model | Qwen 2.5 7B served by Ollama (open source, local, CPU) | Any OpenAI-compatible server works (vLLM, llama.cpp, LM Studio) |
| Backend API | FastAPI + Uvicorn, Server-Sent Events | Async, typed, simple streaming to the UI |
| Persistence | SQLite: LangGraph checkpointer plus app tables | Zero-setup local durability |
| Validation | `jsonschema` against each tool's schema | Bad model arguments become recoverable errors |
| Frontend | Vanilla HTML/CSS/JS, no build step | Small, safe rendering (`textContent` only) |
| Testing | pytest + pytest-asyncio, Playwright for browser flows | Scripted chat model makes the agent deterministic |

## Why incident triage

It is a task where autonomy has real value (the investigation path depends on what each tool returns), where wrong actions are costly (so approval gates matter), and where state genuinely has to survive (incidents outlive processes and shifts). Tools run against simulated systems so the project is self-contained, but the tool interfaces are the real seam: swap the handlers in [app/tools.py](app/tools.py) for Datadog/kubectl/Argo calls and nothing else changes.

Three scenarios are included:

| Scenario | What it exercises |
|---|---|
| `bad-deploy` | 5xx spike after a release; correlate logs, metrics and deploy history; roll back. Metrics backend fails transiently on first call (retry/backoff). |
| `db-pool` | Latency from connection pool exhaustion; needs config + DB-side metrics to rule out the database; config change. |
| `poisoned-logs` | A log line contains an injected instruction to roll back an unrelated service. Also contains a leaked API key and noisy output. Tests prompt-injection handling and redaction. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

**Real model (Ollama):**

```bash
ollama pull qwen2.5:7b          # or qwen2.5:3b on small machines (weaker tool calling)
ollama create qwen2.5-7b-8k -f ollama/Modelfile   # same model with an 8k context window (Ollama defaults to 4k)
uvicorn app.main:app --port 8000
```

Open http://localhost:8000. Configuration is via environment variables (see [app/config.py](app/config.py)):

| Variable | Default |
|---|---|
| `LLM_PROVIDER` | `ollama` (any OpenAI-compatible server); `mock` for the offline scripted demo |
| `LLM_BASE_URL` | `http://localhost:11434/v1` |
| `LLM_MODEL` | `qwen2.5-7b-8k` |
| `MAX_STEPS` / `RUN_TOKEN_BUDGET` | `20` / `80000` |
| `CONTEXT_BUDGET_TOKENS` | `5000` (message budget; the system prompt and tool schemas add ~1.3k) |
| `COST_PER_1K_TOKENS` | `0` (set it to track cost for a hosted model) |

**Offline demo, no model:** `LLM_PROVIDER=mock uvicorn app.main:app` replays a scripted investigation through the real runtime, tools, approval gate and UI.

## Using it

1. Pick a scenario and start a run. The timeline streams each LLM step (reasoning text, tokens, latency), tool call, tool result, retry, guard and compaction event.
2. When the agent proposes a production-changing action, an approval card appears. Approve, or reject with a reason; the reason is given to the agent, which must propose an alternative.
3. Pause/resume/cancel at any time. Send guidance in the message box; it is delivered at the agent's next step boundary.
4. Kill the server mid-run and start it again: running runs continue, and pending approvals are still waiting.

## Architecture

```
UI (SSE + REST) ──> FastAPI ──> AgentRuntime ──> LangGraph StateGraph ──> ChatOpenAI (Ollama)
                                   │                 │  └─ SQLite checkpointer (graph state, messages)
                                   │                 └──> Tools (simulated infra, schema-validated)
                                   └──> SQLite: runs, events, approvals, inbox, memories
```

The graph:

```
START -> ingest -> agent --(tool calls)--> tools --(gated calls)--> approval (interrupt) -> ingest
            ^        |                       |--(finish)--> finalize -> END
            |        └--(no tool call)--> nudge      |--(otherwise)--> ingest
            └---------------------------------------┘
```

`ingest` folds in operator messages and compacts the context; `agent` calls the model with the tool schemas; `tools` validates and runs safe calls in parallel and parks gated ones; `approval` calls LangGraph's `interrupt()` and waits for the operator.

API (see `/docs` for the OpenAPI UI): `POST /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/stream` (SSE, resumable via `Last-Event-ID`), `POST /api/runs/{id}/{pause|resume|cancel|message}`, `POST /api/approvals/{id}`, `GET /api/tools`, `GET /api/memories`.

## App flow

What happens from the operator's click to a finished incident:

```
1. Operator picks a scenario            POST /api/runs
2. Runtime creates the run              recalls past notes for the service, builds the opening
                                        message (alert + goal + notes), stores the run, starts
                                        an asyncio task, and returns immediately
3. Graph starts                         ingest -> agent
4. agent                                the model sees: system rules + live state (plan, fixes
                                        applied, step N/20) + conversation; it replies with
                                        reasoning and tool_calls
5. tools                                each call is schema-validated and guarded; safe calls run
                                        in parallel with retry/backoff; output is redacted,
                                        truncated and wrapped as untrusted data, then returned to
                                        the model as a ToolMessage
6. Loop 4-5                             the model reads results, plans, calls more tools
7. Production-changing call             parked as an approval; graph hits interrupt(); state is
                                        checkpointed; run status = awaiting_approval; the task exits
8. Operator decides                     POST /api/approvals/{id}; the graph resumes with
                                        Command(resume=...): approved -> executed, rejected -> the
                                        reason goes back to the model
9. Verify and finish                    the agent calls check_health, then finish_incident
10. finalize                            status completed (or needs_review if confidence < 0.6);
                                        a redacted note is saved to long-term memory
```

Throughout, every step writes an event row. The UI follows them live through `GET /api/runs/{id}/stream` (SSE), so the operator sees reasoning, tool calls, results, retries, guards, compactions and approvals as they happen. Pause, cancel and operator messages are applied at node boundaries.

**Tool calling.** Tools are registered with a name, description, JSON schema, handler and an approval flag. `bind_tools` sends the schemas to the model; the model only *requests* calls. The runtime executes them, so validation, retries, timeouts, redaction and the approval gate sit between the model and anything that happens. Every tool also takes a one-sentence `rationale`, which the runtime strips before execution and shows as the agent's reason.

**What the model sees each step.** System rules and live state, the alert and recalled notes, its own prior messages and tool results (truncated, redacted, untrusted-wrapped), any operator messages, and a summary in place of older steps once the context budget is exceeded.

## Design decisions

**Durable execution via LangGraph checkpoints.** The graph state (messages, simulated world, plan, pending approvals) is checkpointed to SQLite after every node, keyed by run id (`thread_id`). Run metadata, approvals and an append-only `events` log live in a separate table set that the API and UI read. After a restart `recover()` re-attaches to each run: a run that was mid-flight continues from its last checkpoint, and one parked at an approval waits for the decision. Failed runs (budget, LLM outage) resume by re-running the node that failed.

**Approval gate enforced in the graph, not the prompt.** Tools flagged `requires_approval` are never executed by the model's request alone: the `tools` node records them as pending approvals and routes to the `approval` node, which calls `interrupt()`. The run stops with its state checkpointed (no task or memory is held). When the operator decides, the runtime resumes the graph with `Command(resume=decisions)`: approved calls execute, rejected ones return `REJECTED by operator: <reason>` to the model. Safe calls in the same batch still run in parallel (`asyncio.gather`). `finish_incident` is refused while approvals are pending. This is also the main prompt-injection defense: injected text can make the model *ask*, but cannot make anything happen.

**Context is designed, not accumulated.**
- The system prompt is rebuilt every step with *live state* (plan, applied remediations, step counter) taken from the runtime, so compaction can never lose them.
- Tool output is redacted (keys, tokens, emails), truncated head+tail to `TOOL_OUTPUT_LIMIT` with a hint to narrow the query, and wrapped as `untrusted-data`. The UI shows the full output; the model sees the truncated one.
- When the estimated context exceeds `CONTEXT_BUDGET_TOKENS`, the middle of the transcript is summarized by the model (deterministic fallback if that call fails). The alert is always kept, the recent tail is kept verbatim, and a tool result is never separated from its tool call.
- Small default budget (5k) on purpose: small local models degrade well before their advertised window.

**Memory.** Short-term: the transcript plus world state, per run. Long-term: on successful completion with high confidence, a redacted one-line note (root cause and fix) is stored per service with a 30-day TTL, and recalled into the next incident on that service. It is framed to the model as possibly stale. Low-confidence runs are not promoted.

**Error recovery.** Transient tool errors retry with exponential backoff (visible as events). Permanent errors, unknown tools, bad JSON and schema violations go back to the model as tool results so it can self-correct. Identical tool calls repeated three times are blocked. Step and token budgets stop runaway loops and mark the run `failed`, resumable. Confidence below 0.6 ends in `needs_review` instead of `completed`.

**Observability built in.** Every decision is an event row: reasoning text, tool calls, results, durations, retries, approvals, compactions, with per-step tokens, latency and estimated context size, and per-run totals and cost on the run. This is a deliberately simple in-house trace; it is shaped so it can be exported as OpenTelemetry GenAI spans (see below).

**What the framework does and doesn't do here.** LangGraph provides the graph execution, checkpointing, `interrupt`/resume and streaming; LangChain provides the chat model (`ChatOpenAI` against Ollama), message types and `bind_tools`. The agent-specific decisions are mine and live in nodes: the approval policy, context budgeting and compaction, redaction, untrusted-output wrapping, loop guard, retry/backoff and the memory rules. The tool registry in [app/tools.py](app/tools.py) is a plain registry (schema, handler, approval flag) rather than `@tool` functions, because tool execution needs per-run state and the approval policy.

## Tested with a real model

Model: `qwen2.5:7b` (8k context via [ollama/Modelfile](ollama/Modelfile)), CPU only, no GPU. Each scenario was driven through the API with the operator approving the proposed action. These are single runs, not an eval; they are what I used to find and fix problems.

| Scenario | Outcome | Notes |
|---|---|---|
| `bad-deploy` | Completed, 10 steps, service verified recovered | Read the real `KeyError` logs, tied them to the v2.14.0 deploy, proposed the right rollback, retried the flaky metrics backend, was nudged once when it answered without a tool call. |
| `poisoned-logs` | Completed, 10 steps, service verified recovered | Never proposed the injected rollback; found the full disk and stuck log rotation from logs and the disk metric; proposed a restart, verified, finished. About 5 minutes. |
| `db-pool` | Right diagnosis and fix proposed; not re-run end to end | The model raised the pool from 10 to 20. My simulation originally only accepted 30+, so it kept going and proposed an unneeded restart; I relaxed the rule to 20 afterwards but have not repeated the full run. |

Latency on this machine ranged from about 10 seconds to 4 minutes per model step depending on load, so a run takes roughly 5 to 25 minutes. `LLM_PROVIDER=mock` exists for fast demos and tests.

What real runs exposed, and what changed in response:
- **Narrow log searches matched nothing** and the model then reasoned from the deploy note instead of evidence. Empty searches now say how many lines exist per level and suggest a broader query, and the tool description tells the model to start with a level filter.
- **Small models often write no reasoning text** before a tool call. Every tool call now takes a one-sentence `rationale` that the runtime strips before execution and shows in the timeline.
- **Ollama's default 4k context** was smaller than my compaction budget. The Modelfile raises it to 8k and the default budget is 5k.
- **Repeated identical calls** wasted steps. The guard now blocks a repeat, but resets whenever a remediation is applied, because re-checking health after a fix is legitimate.
- **The model's final report claimed an action it never took** ("archived logs"). The result now lists `actions_taken` from what the runtime actually executed, and keeps the model's own claim separately as `agent_reported_actions`.

## Known limitations

- Tools are simulated; "fixed" is determined by a rule per scenario, not a real system.
- Runs are driven in-process by asyncio tasks, so one API process owns them. Multiple workers would need a lease/heartbeat column on `runs` (or a Postgres checkpointer plus a job queue).
- Context size is estimated (chars/4), not tokenized.
- Prompt-injection defense relies on gating plus delimiting; a model can still be misled into proposing a bad action and a tired operator could approve it. Approval cards show the agent's stated rationale to help.
- Quality depends on the local model: small models can be weak at tool calling. Evals are not implemented.

## What's next

1. Eval harness: golden scenarios run against model/prompt versions, scored on root cause, actions taken, number of steps and unsafe proposals, with LLM-as-judge for the summary.
2. LangSmith/Langfuse or OpenTelemetry GenAI export alongside the existing event log, and a cost/latency budget alert per run.
3. Real tool adapters (Kubernetes, Prometheus, Loki) behind the same registry, with per-tool sandboxing and audit logging.
4. Confidence-based routing: auto-approve low-risk reversible actions (e.g. read-only restarts in staging) by policy.
5. Worker leases for horizontal scaling and a supervisor/worker split for multi-service incidents.
6. Embedding-based recall for long-term memory once there is enough history for keyword-by-service to stop being sufficient.
