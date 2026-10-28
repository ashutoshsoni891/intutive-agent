import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:7b"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "300")))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/agent.db"))

    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "20")))
    run_token_budget: int = field(default_factory=lambda: int(_env("RUN_TOKEN_BUDGET", "80000")))
    context_budget_tokens: int = field(default_factory=lambda: int(_env("CONTEXT_BUDGET_TOKENS", "6000")))
    compaction_keep_recent: int = 6
    tool_output_limit: int = field(default_factory=lambda: int(_env("TOOL_OUTPUT_LIMIT", "2000")))

    tool_max_attempts: int = 3
    tool_backoff_s: float = field(default_factory=lambda: float(_env("TOOL_BACKOFF_S", "0.5")))
    tool_timeout_s: float = 20.0

    low_confidence: float = 0.6
    memory_ttl_days: int = 30
    cost_per_1k_tokens: float = field(default_factory=lambda: float(_env("COST_PER_1K_TOKENS", "0")))
