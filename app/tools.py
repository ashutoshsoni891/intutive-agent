import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from .scenarios import SCENARIOS, fix_matches


class ToolError(Exception):
    """Permanent failure; the message is returned to the model so it can correct itself."""


class TransientToolError(ToolError):
    """Retryable failure (timeouts, upstream 5xx)."""


Handler = Callable[[dict, dict], Awaitable[str]]
NO_RATIONALE = {"finish_incident", "update_plan"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    requires_approval: bool = False

    def spec(self) -> dict:
        """Schema sent to the model. Adds a rationale argument that the runtime strips before execution."""
        params = {**self.parameters, "properties": dict(self.parameters["properties"]), "required": list(self.parameters["required"])}
        if self.name not in NO_RATIONALE:
            params["properties"]["rationale"] = {"type": "string", "description": "One sentence: why you are making this call"}
            params["required"].append("rationale")
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": params}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _service(world: dict, name: str) -> str:
    if name not in world["data"]["services"]:
        known = ", ".join(world["data"]["services"])
        raise ToolError(f"unknown service '{name}'. Known services: {known}")
    return name


def _fmt_age(minutes: int) -> str:
    return f"{-minutes}m ago" if minutes > -1440 else f"{-minutes // 1440}d ago"


async def update_plan(world: dict, args: dict) -> str:
    world["state"]["plan"] = [str(s) for s in args["steps"]]
    return "plan updated"


async def search_logs(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    level = args.get("level")
    query = (args.get("query") or "").lower()
    limit = int(args.get("limit", 40))
    lines = world["data"]["logs"].get(service, [])
    if level:
        lines = [l for l in lines if l["level"] == level.upper()]
    if query:
        lines = [l for l in lines if query in l["msg"].lower()]
    lines = sorted(lines, key=lambda l: l["t"])[-limit:]
    if not lines:
        all_lines = world["data"]["logs"].get(service, [])
        counts = ", ".join(f"{lv}={sum(1 for l in all_lines if l['level'] == lv)}" for lv in ("ERROR", "WARN", "INFO"))
        return (
            f"no matching log lines ({len(all_lines)} lines in window: {counts}). "
            "The query is a substring match on the message text; retry with only a level filter or a shorter term."
        )
    return "\n".join(f"[{_fmt_age(l['t'])}] {l['level']} {l['msg']}" for l in lines)


def _series(world: dict, service: str, metric: str, window: int) -> list[tuple[int, float]]:
    spec = world["data"]["metrics"].get(service, {}).get(metric)
    if spec is None:
        available = ", ".join(world["data"]["metrics"].get(service, {})) or "none"
        raise ToolError(f"metric '{metric}' not available for {service}. Available: {available}")
    fixed = world["state"]["fixed"] and service == SCENARIOS[world["scenario"]]["service"]
    points = []
    for age in range(window, -1, -5):
        degraded = spec["since_min"] > 0 and age <= spec["since_min"] and not (fixed and age <= 5)
        points.append((age, spec["incident"] if degraded else spec["baseline"]))
    return points


async def query_metrics(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    if not world["state"]["metrics_probe_failed"]:
        world["state"]["metrics_probe_failed"] = True
        raise TransientToolError("metrics backend returned 503")
    window = min(int(args.get("window_minutes", 30)), 120)
    points = _series(world, service, args["metric"], window)
    return f"{service} {args['metric']} (last {window}m):\n" + "\n".join(f"  -{a}m: {v}" for a, v in points)


async def list_deployments(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    deploys = world["data"]["deployments"].get(service, [])
    if not deploys:
        return "no deployments recorded"
    current = world["state"].get("rolled_back_to", {}).get(service)
    lines = [f"{d['version']} deployed {_fmt_age(d['t'])} by {d['author']}: {d['note']}" for d in deploys]
    return (f"currently running: {current}\n" if current else "") + "\n".join(lines)


async def get_config(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    cfg = world["data"]["config"].get(service, {})
    return "\n".join(f"{k}={v}" for k, v in cfg.items()) or "no config"


async def check_health(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    metrics = world["data"]["metrics"].get(service, {})
    key = "error_rate" if "error_rate" in metrics else next(iter(metrics), None)
    if key is None:
        return f"{service}: no health signals"
    value = _series(world, service, key, 5)[-1][1]
    spec = metrics[key]
    status = "healthy" if value == spec["baseline"] else "degraded"
    return f"{service}: {status} ({key}={value})"


def _record_action(world: dict, tool: str, args: dict) -> bool:
    world["state"]["actions"].append({"tool": tool, "args": args})
    improved = fix_matches(world["scenario"], tool, args)
    if improved:
        world["state"]["fixed"] = True
    return improved


async def rollback_deployment(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    versions = [d["version"] for d in world["data"]["deployments"].get(service, [])]
    if args["to_version"] not in versions:
        raise ToolError(f"version {args['to_version']} not found. Available: {', '.join(versions)}")
    await asyncio.sleep(0.2)
    _record_action(world, "rollback_deployment", args)
    world["state"].setdefault("rolled_back_to", {})[service] = args["to_version"]
    return f"{service} rolled back to {args['to_version']}; rollout complete"


async def update_config(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    cfg = world["data"]["config"].get(service, {})
    if args["key"] not in cfg:
        raise ToolError(f"unknown config key '{args['key']}'. Keys: {', '.join(cfg)}")
    await asyncio.sleep(0.2)
    cfg[args["key"]] = args["value"]
    _record_action(world, "update_config", args)
    return f"{service}: set {args['key']}={args['value']}; pods reloading"


async def restart_service(world: dict, args: dict) -> str:
    service = _service(world, args["service"])
    await asyncio.sleep(0.2)
    _record_action(world, "restart_service", args)
    return f"{service}: rolling restart complete"


async def finish_incident(world: dict, args: dict) -> str:
    return "recorded"


SERVICE = {"type": "string", "description": "service name"}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "update_plan",
            "Record or revise your investigation plan as a short list of steps. Call this first, and again when the plan changes.",
            _obj({"steps": {"type": "array", "items": {"type": "string"}}}, ["steps"]),
            update_plan,
        ),
        Tool(
            "search_logs",
            "Search recent logs of a service. Filter by level (ERROR/WARN/INFO) and/or a case-insensitive substring of the message. Start with a level filter and no query; narrow only if output is large. Returns newest lines last.",
            _obj(
                {
                    "service": SERVICE,
                    "level": {"type": "string", "enum": ["ERROR", "WARN", "INFO"]},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["service"],
            ),
            search_logs,
        ),
        Tool(
            "query_metrics",
            "Get a metric time series (5-minute resolution) for a service. Metrics: error_rate, latency_p95, cpu, db_connections, queue_depth, disk_used_pct.",
            _obj(
                {"service": SERVICE, "metric": {"type": "string"}, "window_minutes": {"type": "integer"}},
                ["service", "metric"],
            ),
            query_metrics,
        ),
        Tool(
            "list_deployments",
            "List recent deployments of a service, newest first.",
            _obj({"service": SERVICE}, ["service"]),
            list_deployments,
        ),
        Tool("get_config", "Show the current configuration of a service.", _obj({"service": SERVICE}, ["service"]), get_config),
        Tool(
            "check_health",
            "Check whether a service is currently healthy. Use it to verify a remediation worked.",
            _obj({"service": SERVICE}, ["service"]),
            check_health,
        ),
        Tool(
            "rollback_deployment",
            "Roll a service back to a previous version. Production-changing: requires operator approval.",
            _obj({"service": SERVICE, "to_version": {"type": "string"}}, ["service", "to_version"]),
            rollback_deployment,
            requires_approval=True,
        ),
        Tool(
            "update_config",
            "Change one config value of a service. Production-changing: requires operator approval.",
            _obj(
                {"service": SERVICE, "key": {"type": "string"}, "value": {"type": ["string", "number", "boolean"]}},
                ["service", "key", "value"],
            ),
            update_config,
            requires_approval=True,
        ),
        Tool(
            "restart_service",
            "Rolling restart of a service. Production-changing: requires operator approval.",
            _obj({"service": SERVICE}, ["service"]),
            restart_service,
            requires_approval=True,
        ),
        Tool(
            "finish_incident",
            "End the investigation. Call once you have verified the outcome or cannot make further progress.",
            _obj(
                {
                    "root_cause": {"type": "string"},
                    "confidence": {"type": "number", "description": "0.0 to 1.0"},
                    "actions_taken": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                ["root_cause", "confidence", "summary"],
            ),
            finish_incident,
        ),
    ]
}


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]
