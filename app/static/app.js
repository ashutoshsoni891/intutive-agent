const $ = (id) => document.getElementById(id);
let currentId = null;
let source = null;
let refreshTimer = null;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

async function init() {
  const [scenarios, health] = await Promise.all([api("/scenarios"), api("/health")]);
  $("llm").textContent = `${health.provider} · ${health.model}`;
  const sel = $("scenario");
  scenarios.forEach((s) => sel.append(new Option(s.title, s.id)));
  const showAlert = () => ($("alert").textContent = scenarios.find((s) => s.id === sel.value).alert);
  sel.onchange = showAlert;
  showAlert();

  $("new-run").onsubmit = async (e) => {
    e.preventDefault();
    const run = await api("/runs", { method: "POST", body: { scenario: sel.value, goal: $("goal").value || null } });
    $("goal").value = "";
    select(run.id);
  };
  for (const action of ["pause", "resume", "cancel"]) {
    $(action).onclick = () => api(`/runs/${currentId}/${action}`, { method: "POST" }).then(refreshRun).catch((e) => alert(e.message));
  }
  $("say").onsubmit = async (e) => {
    e.preventDefault();
    const text = $("say-text").value.trim();
    if (!text) return;
    await api(`/runs/${currentId}/message`, { method: "POST", body: { text } });
    $("say-text").value = "";
  };
  setInterval(() => loadSidebar().catch(() => {}), 3000);
  loadSidebar();
}

async function loadSidebar() {
  const [runs, memories] = await Promise.all([api("/runs"), api("/memories")]);
  const ul = $("runs");
  ul.replaceChildren(
    ...runs.map((r) => {
      const li = el("li", r.id === currentId ? "active" : "");
      const badge = el("span", `badge ${r.status}`, r.status.replace("_", " "));
      li.append(badge, el("span", "", r.scenario), el("div", "muted", `step ${r.step} · ${r.id}`));
      li.onclick = () => select(r.id);
      return li;
    })
  );
  $("memories").replaceChildren(
    ...(memories.length ? memories.map((m) => el("li", "", `${m.service}: ${m.text}`)) : [el("li", "", "none yet")])
  );
}

function select(id) {
  currentId = id;
  if (source) source.close();
  $("timeline").replaceChildren();
  $("empty").hidden = true;
  $("run").hidden = false;
  refreshRun();
  source = new EventSource(`/api/runs/${id}/stream`);
  const handle = (msg) => {
    if (!msg.data) return;
    const ev = JSON.parse(msg.data);
    $("timeline").prepend(renderEvent(ev));
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refreshRun, 150);
  };
  for (const t of EVENT_TYPES) source.addEventListener(t, handle);
  source.addEventListener("end", () => source.close());
  loadSidebar();
}

const EVENT_TYPES = [
  "run_started", "llm_call", "tool_call", "tool_result", "tool_retry", "tool_error", "guard", "compaction",
  "approval_requested", "approval_decided", "operator_message", "memory_recalled", "memory_saved",
  "recovered", "status", "error", "run_finished",
];

async function refreshRun() {
  if (!currentId) return;
  const run = await api(`/runs/${currentId}`);
  renderRun(run);
  loadSidebar().catch(() => {});
}

function stat(label, value) {
  const d = el("div");
  d.append(el("strong", "", value), el("span", "", label));
  return d;
}

function renderRun(run) {
  $("title").textContent = `${run.scenario} · ${run.id}`;
  const badge = $("status");
  badge.className = `badge ${run.status}`;
  badge.textContent = run.status.replace("_", " ");
  const tokens = run.usage.prompt_tokens + run.usage.completion_tokens;
  $("stats").replaceChildren(
    stat("step", `${run.step} / ${run.max_steps}`),
    stat("tokens", `${tokens.toLocaleString()} / ${run.usage.token_budget.toLocaleString()}`),
    stat("LLM time", `${(run.usage.llm_ms / 1000).toFixed(1)}s`),
    stat("cost", `$${run.usage.cost.toFixed(4)}`)
  );
  $("pause").disabled = run.status !== "running";
  $("resume").disabled = !["paused", "failed"].includes(run.status);
  $("cancel").disabled = ["completed", "needs_review", "failed", "cancelled"].includes(run.status);
  $("say-text").disabled = ["completed", "needs_review", "cancelled"].includes(run.status);

  $("plan").replaceChildren(...(run.plan.length ? run.plan.map((s) => el("li", "", s)) : [el("li", "muted", "waiting for agent plan")]));
  $("actions").replaceChildren(
    ...(run.actions.length
      ? run.actions.map((a) => el("li", "", `${a.tool}(${Object.values(a.args).join(", ")})`))
      : [el("li", "muted", "none")])
  );

  const box = $("approvals");
  box.replaceChildren(...run.approvals.filter((a) => a.status === "pending").map(approvalCard));

  const res = $("result");
  res.replaceChildren();
  if (run.result) {
    const d = el("div", "result");
    d.append(
      el("strong", "", `Root cause (confidence ${Math.round(run.result.confidence * 100)}%)`),
      el("p", "", run.result.root_cause),
      el("p", "muted", run.result.summary),
      el("p", "muted", run.result.service_recovered ? "Service verified recovered by runtime state." : "No effective remediation applied.")
    );
    if (run.status === "needs_review") d.append(el("p", "", "Low confidence: flagged for human review."));
    res.append(d);
  }
  if (run.error) res.append(el("div", "result", `Error: ${run.error}`));
}

function approvalCard(a) {
  const card = el("div", "card approval");
  card.append(el("div", "meta", "Approval required: production-changing action"));
  card.append(el("strong", "", a.tool), el("pre", "", JSON.stringify(a.args, null, 2)));
  const reason = el("input");
  reason.placeholder = "Reason (optional, shown to the agent)";
  const ok = el("button", "ok", "Approve");
  const no = el("button", "danger", "Reject");
  const decide = (approve) => api(`/approvals/${a.id}`, { method: "POST", body: { approve, reason: reason.value || null } }).then(refreshRun);
  ok.onclick = () => decide(true);
  no.onclick = () => decide(false);
  card.append(reason, ok, document.createTextNode(" "), no);
  return card;
}

function card(cls, meta, body, pre) {
  const c = el("div", `card ${cls}`);
  c.append(el("div", "meta", meta));
  if (body) c.append(el("div", "", body));
  if (pre) c.append(el("pre", "", pre));
  return c;
}

function renderEvent(ev) {
  const d = ev.data;
  const time = new Date(ev.ts * 1000).toLocaleTimeString();
  switch (ev.type) {
    case "run_started": return card("", `${time} · run started`, d.alert);
    case "llm_call":
      return card(
        "thought",
        `${time} · step ${d.step} · ${d.prompt_tokens}+${d.completion_tokens} tok · ${d.latency_ms}ms · ctx≈${d.context_tokens_est}`,
        d.thought || "(no reasoning text)",
        d.tool_calls.length ? "→ " + d.tool_calls.join(", ") : "→ no tool call"
      );
    case "tool_call": return card("tool_call", `${time} · tool call`, d.tool, JSON.stringify(d.args) + (d.rationale ? `
why: ${d.rationale}` : ""));
    case "tool_result":
      return card(
        d.is_error ? "error" : "tool_result",
        `${time} · result of ${d.tool} · ${d.duration_ms}ms · ${d.chars} chars${d.truncated_for_context ? " (truncated in context)" : ""}`,
        "", d.output
      );
    case "tool_retry":
      return card("retry", `${time} · retry`, `${d.tool} attempt ${d.attempt} failed: ${d.error}` + (d.backoff_s != null ? ` · backing off ${d.backoff_s}s` : " · giving up"));
    case "tool_error": return card("error", `${time} · tool error`, `${d.tool}: ${d.error}`);
    case "guard": return card("guard", `${time} · guard`, `Blocked repeated call to ${d.tool}`);
    case "compaction":
      return card("compaction", `${time} · context compacted`, `${d.dropped_messages} messages summarized · ~${d.tokens_before} → ~${d.tokens_after} tokens`);
    case "approval_requested": return card("approval", `${time} · approval requested`, d.tool, JSON.stringify(d.args));
    case "approval_decided": return card("approval", `${time} · approval ${d.approved ? "granted" : "rejected"}`, d.tool, d.reason || "");
    case "operator_message": return card("operator", `${time} · operator`, d.text);
    case "memory_recalled": return card("", `${time} · recalled ${d.count} past incident note(s)`, "", d.notes.join("\n"));
    case "memory_saved": return card("", `${time} · saved to long-term memory`, d.text);
    case "recovered": return card("retry", `${time} · run recovered`, d.reason);
    case "status": return card("", `${time} · status → ${d.status}`, d.error || "");
    case "error": return card("error", `${time} · error`, d.message);
    case "run_finished": return card("operator", `${time} · finished`, d.root_cause, d.summary);
    default: return card("", ev.type, "", JSON.stringify(d));
  }
}

init();
