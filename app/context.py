import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage

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


def _size(m: BaseMessage) -> int:
    return len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", None) or []))


def estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum(_size(m) for m in messages) // 4


def truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.4)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} chars omitted; narrow the query with level/query/limit ...]\n{text[-tail:]}"


def wrap_untrusted(tool: str, text: str) -> str:
    return f"<tool_output name=\"{tool}\" trust=\"untrusted-data\">\n{text}\n</tool_output>"


def _split_for_compaction(messages: list[BaseMessage], keep_recent: int) -> int:
    """Index where the kept tail starts. Never splits an assistant tool-call from its tool results."""
    cut = max(1, len(messages) - keep_recent)
    while cut < len(messages) and messages[cut].type == "tool":
        cut += 1
    return cut


def render_for_summary(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        if m.type == "ai":
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
            lines.append(f"assistant: {m.content} {calls}".strip())
        else:
            lines.append(f"{m.type}: {str(m.content)[:600]}")
    return "\n".join(lines)


SUMMARY_PROMPT = (
    "Summarize this incident investigation transcript for the same agent to continue from. Keep: services checked, "
    "key evidence (exact error strings, numbers, versions), hypotheses confirmed or ruled out, remediation actions "
    "taken or rejected by the operator, and open questions. Be concise; no preamble."
)


async def compact(messages: list[BaseMessage], llm, keep_recent: int) -> tuple[list[BaseMessage], dict] | None:
    """Replace the middle of the transcript with a summary. The first message (the alert) is always kept.

    Returns graph state updates: removals for everything after the first message, then the summary and the
    kept tail re-added in order (the add_messages reducer appends, so the tail has to be re-added).
    """
    cut = _split_for_compaction(messages, keep_recent)
    middle = messages[1:cut]
    if len(middle) < 2:
        return None
    before = estimate_tokens(messages)
    transcript = render_for_summary(middle)
    try:
        resp = await llm.ainvoke([SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=transcript[-12000:])])
        summary = str(resp.content).strip() or transcript[-1500:]
    except Exception:
        summary = "Deterministic digest of earlier steps:\n" + transcript[-1500:]
    summary_msg = HumanMessage(content=f"[Summary of earlier investigation steps]\n{redact(summary)}")
    tail = [m.model_copy(update={"id": None}) for m in messages[cut:]]
    after = [messages[0], summary_msg, *tail]
    updates = [*(RemoveMessage(id=m.id) for m in messages[1:]), summary_msg, *tail]
    return updates, {
        "dropped_messages": len(middle),
        "tokens_before": before,
        "tokens_after": estimate_tokens(after),
    }
