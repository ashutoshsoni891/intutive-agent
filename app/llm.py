import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from .config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "mock":
        return DemoChatModel()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model for tests: replays (content, [(tool, args)]) turns, one per tool-enabled call."""

    turns: list = []
    tools_bound: bool = False
    cursor: list = Field(default_factory=lambda: [0])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self.model_copy(update={"tools_bound": True})

    def _pick(self, messages: list[BaseMessage]):
        if self.cursor[0] >= len(self.turns):
            raise RuntimeError("script exhausted")
        turn = self.turns[self.cursor[0]]
        self.cursor[0] += 1
        return turn

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if not self.tools_bound:
            msg = AIMessage(content="Earlier steps: investigated logs and metrics; see transcript.")
        else:
            content, calls = self._pick(messages)
            msg = AIMessage(
                content=content,
                tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"} for n, a in calls],
                usage_metadata={
                    "input_tokens": sum(len(str(m.content)) for m in messages) // 4,
                    "output_tokens": 40,
                    "total_tokens": 0,
                },
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class DemoChatModel(ScriptedChatModel):
    """Offline demo. Stateless, so it survives restarts: the turn is the number of AI messages seen so far."""

    def _pick(self, messages: list[BaseMessage]):
        from .demo_script import demo_turns

        first = next(m.content for m in messages if m.type == "human")
        scenario = first.split("[scenario:", 1)[1].split("]", 1)[0]
        turns = demo_turns(scenario)
        done = sum(1 for m in messages if m.type == "ai")
        return turns[min(done, len(turns) - 1)]
