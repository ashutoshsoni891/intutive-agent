import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

import httpx

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    def message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class LLMError(Exception):
    pass


class OpenAICompatLLM:
    """Chat client for any OpenAI-compatible server (Ollama, vLLM, llama.cpp, LM Studio)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.client = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_s,
        )

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict = {"model": self.s.llm_model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        last_err: Exception | None = None
        for attempt in range(3):
            started = time.monotonic()
            try:
                resp = await self.client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                return self._parse(resp.json(), int((time.monotonic() - started) * 1000))
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM request failed after retries: {last_err!r}")

    @staticmethod
    def _parse(data: dict, latency_ms: int) -> LLMResponse:
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(c.get("id") or f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], args))
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    async def aclose(self):
        await self.client.aclose()


class ScriptedLLM:
    """Deterministic stand-in for tests and offline demos: replays a list of (content, calls) turns."""

    def __init__(self, turns: list[tuple[str, list[tuple[str, dict]]]]):
        self.turns = list(turns)
        self.calls = 0

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls += 1
        if not tools:
            return LLMResponse(content="Earlier steps: investigated logs and metrics; see transcript.")
        if not self.turns:
            raise LLMError("script exhausted")
        content, calls = self.turns.pop(0)
        return LLMResponse(
            content=content,
            tool_calls=[ToolCall(f"call_{uuid.uuid4().hex[:8]}", n, json.dumps(a)) for n, a in calls],
            prompt_tokens=sum(len(json.dumps(m)) for m in messages) // 4,
            completion_tokens=40,
            latency_ms=5,
        )


class DemoLLM(ScriptedLLM):
    """Offline demo: replays the scripted investigation for the scenario named in the first user message.
    Stateless, so it keeps working across restarts: the turn is the number of assistant messages so far."""

    def __init__(self):
        super().__init__([])

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        if not tools:
            return await super().chat(messages, tools)
        from .demo_script import demo_turns

        first = next(m["content"] for m in messages if m["role"] == "user")
        scenario = first.split("[scenario:", 1)[1].split("]", 1)[0]
        turns = demo_turns(scenario)
        turn = min(sum(1 for m in messages if m["role"] == "assistant"), len(turns) - 1)
        self.turns = [turns[turn]]
        return await super().chat(messages, tools)


def build_llm(settings: Settings):
    if settings.llm_provider == "mock":
        return DemoLLM()
    return OpenAICompatLLM(settings)
