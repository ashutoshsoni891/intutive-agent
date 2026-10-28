import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .llm import build_llm
from .runtime import AgentRuntime
from .scenarios import SCENARIOS
from .store import TERMINAL, Store
from .tools import TOOLS

STATIC = Path(__file__).parent / "static"


class NewRun(BaseModel):
    scenario: str
    goal: str | None = None


class Message(BaseModel):
    text: str


class Decision(BaseModel):
    approve: bool
    reason: str | None = None


def create_app(settings: Settings | None = None, llm=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(settings.db_path)
        runtime = AgentRuntime(store, llm or build_llm(settings), settings)
        app.state.store, app.state.runtime = store, runtime
        runtime.recover()
        yield
        for task in runtime.tasks.values():
            task.cancel()
        store.close()

    app = FastAPI(title="Incident Triage Agent", lifespan=lifespan)

    def get_run(request: Request, run_id: str) -> dict:
        run = request.app.state.store.get_run(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return run

    def view(request: Request, run: dict) -> dict:
        store = request.app.state.store
        tokens = run["prompt_tokens"] + run["completion_tokens"]
        return {
            "id": run["id"],
            "scenario": run["scenario"],
            "goal": run["goal"],
            "status": run["status"],
            "step": run["step"],
            "max_steps": settings.max_steps,
            "plan": run["world"]["state"]["plan"],
            "actions": run["world"]["state"]["actions"],
            "result": run["result"],
            "error": run["error"],
            "approvals": store.approvals(run["id"]),
            "usage": {
                "prompt_tokens": run["prompt_tokens"],
                "completion_tokens": run["completion_tokens"],
                "token_budget": settings.run_token_budget,
                "llm_ms": run["llm_ms"],
                "cost": round(tokens / 1000 * settings.cost_per_1k_tokens, 6),
            },
            "created_at": run["created_at"],
        }

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}

    @app.get("/api/scenarios")
    async def scenarios():
        return [{"id": k, "title": v["title"], "alert": v["alert"]} for k, v in SCENARIOS.items()]

    @app.get("/api/tools")
    async def tools():
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters, "requires_approval": t.requires_approval}
            for t in TOOLS.values()
        ]

    @app.get("/api/memories")
    async def memories(request: Request):
        return request.app.state.store.all_memories()

    @app.post("/api/runs", status_code=201)
    async def create_run(request: Request, body: NewRun):
        if body.scenario not in SCENARIOS:
            raise HTTPException(422, f"unknown scenario; choose from {list(SCENARIOS)}")
        run = request.app.state.runtime.start_run(body.scenario, body.goal)
        return view(request, run)

    @app.get("/api/runs")
    async def list_runs(request: Request):
        return request.app.state.store.list_runs()

    @app.get("/api/runs/{run_id}")
    async def run_detail(request: Request, run_id: str):
        return view(request, get_run(request, run_id))

    @app.get("/api/runs/{run_id}/events")
    async def run_events(request: Request, run_id: str, after: int = 0):
        get_run(request, run_id)
        return request.app.state.store.events(run_id, after)

    @app.get("/api/runs/{run_id}/stream")
    async def stream(request: Request, run_id: str, after: int = 0):
        get_run(request, run_id)
        store = request.app.state.store
        last_header = request.headers.get("last-event-id")
        cursor = int(last_header) if last_header and last_header.isdigit() else after

        async def gen():
            nonlocal cursor
            idle = 0
            while not await request.is_disconnected():
                events = store.events(run_id, cursor)
                for ev in events:
                    cursor = ev["id"]
                    yield f"id: {ev['id']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if events:
                    idle = 0
                else:
                    idle += 1
                    if store.get_run(run_id)["status"] in TERMINAL and idle > 2:
                        yield "event: end\ndata: {}\n\n"
                        return
                    if idle % 20 == 0:
                        yield ": keepalive\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    def control(name: str):
        async def handler(request: Request, run_id: str):
            get_run(request, run_id)
            if not getattr(request.app.state.runtime, name)(run_id):
                raise HTTPException(409, f"cannot {name} in current state")
            return view(request, get_run(request, run_id))

        return handler

    for action in ("pause", "resume", "cancel"):
        app.post(f"/api/runs/{{run_id}}/{action}")(control(action))

    @app.post("/api/runs/{run_id}/message", status_code=202)
    async def message(request: Request, run_id: str, body: Message):
        run = get_run(request, run_id)
        if run["status"] in TERMINAL:
            raise HTTPException(409, "run is finished")
        request.app.state.runtime.post_message(run_id, body.text)
        return {"queued": True}

    @app.post("/api/approvals/{approval_id}")
    async def decide(request: Request, approval_id: str, body: Decision):
        result = request.app.state.runtime.decide(approval_id, body.approve, body.reason)
        if result is None:
            raise HTTPException(409, "approval not found or already decided")
        return result

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    return app
