import json
import re

SECRET_PATTERNS = [
    (re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_-]{10,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_KEY]"),
]


def redact(text: str) -> str:
    for pattern, repl in SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.4)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} chars omitted; narrow the query with level/query/limit ...]\n{text[-tail:]}"


def wrap_untrusted(tool: str, text: str) -> str:
    return f"<tool_output name=\"{tool}\" trust=\"untrusted-data\">\n{text}\n</tool_output>"


def _split_for_compaction(messages: list[dict], keep_recent: int) -> int:
    """Index where the kept tail starts. Never splits an assistant tool-call from its tool results."""
    cut = max(1, len(messages) - keep_recent)
    while cut < len(messages) and messages[cut]["role"] == "tool":
        cut += 1
    return cut


def render_for_summary(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "assistant":
            calls = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls", [])
            )
            lines.append(f"assistant: {m.get('content', '')} {calls}".strip())
        else:
            lines.append(f"{m['role']}: {m.get('content', '')[:600]}")
    return "\n".join(lines)


SUMMARY_PROMPT = (
    "Summarize this incident investigation transcript for the same agent to continue from. Keep: services checked, "
    "key evidence (exact error strings, numbers, versions), hypotheses confirmed or ruled out, remediation actions "
    "taken or rejected by the operator, and open questions. Be concise; no preamble."
)


async def compact(messages: list[dict], llm, keep_recent: int) -> tuple[list[dict], dict] | None:
    """Replace the middle of the transcript with a summary. The first message (the alert) is always kept."""
    cut = _split_for_compaction(messages, keep_recent)
    middle = messages[1:cut]
    if len(middle) < 2:
        return None
    before = estimate_tokens(messages)
    transcript = render_for_summary(middle)
    try:
        resp = await llm.chat(
            [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": transcript[-12000:]}]
        )
        summary = resp.content.strip() or transcript[-1500:]
    except Exception:
        summary = "Deterministic digest of earlier steps:\n" + transcript[-1500:]
    summary_msg = {"role": "user", "content": f"[Summary of earlier investigation steps]\n{redact(summary)}"}
    compacted = [messages[0], summary_msg, *messages[cut:]]
    return compacted, {"dropped_messages": len(middle), "tokens_before": before, "tokens_after": estimate_tokens(compacted)}
