import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
create table if not exists runs(
    id text primary key,
    scenario text not null,
    goal text not null,
    status text not null,
    step integer not null default 0,
    prompt_tokens integer not null default 0,
    completion_tokens integer not null default 0,
    llm_ms integer not null default 0,
    world text not null,
    result text,
    error text,
    created_at real not null,
    updated_at real not null
);
create table if not exists messages(
    run_id text not null,
    seq integer not null,
    body text not null,
    primary key(run_id, seq)
);
create table if not exists events(
    id integer primary key autoincrement,
    run_id text not null,
    ts real not null,
    type text not null,
    data text not null
);
create index if not exists events_run on events(run_id, id);
create table if not exists approvals(
    id text primary key,
    run_id text not null,
    call_id text not null,
    tool text not null,
    args text not null,
    status text not null,
    reason text,
    applied integer not null default 0,
    created_at real not null,
    decided_at real
);
create table if not exists inbox(
    id integer primary key autoincrement,
    run_id text not null,
    body text not null
);
create table if not exists memories(
    id integer primary key autoincrement,
    service text not null,
    text text not null,
    created_at real not null,
    expires_at real not null
);
"""

TERMINAL = {"completed", "needs_review", "failed", "cancelled"}


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=wal")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def _row(self, row):
        return dict(row) if row else None

    def create_run(self, scenario: str, goal: str, world: dict) -> dict:
        run_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self.db:
            self.db.execute(
                "insert into runs(id,scenario,goal,status,world,created_at,updated_at) values(?,?,?,?,?,?,?)",
                (run_id, scenario, goal, "running", json.dumps(world), now, now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict | None:
        run = self._row(self.db.execute("select * from runs where id=?", (run_id,)).fetchone())
        if run:
            run["world"] = json.loads(run["world"])
            run["result"] = json.loads(run["result"]) if run["result"] else None
        return run

    def list_runs(self) -> list[dict]:
        rows = self.db.execute(
            "select id,scenario,goal,status,step,prompt_tokens,completion_tokens,created_at from runs order by created_at desc"
        ).fetchall()
        return [dict(r) for r in rows]

    def runs_with_status(self, status: str) -> list[str]:
        rows = self.db.execute("select id from runs where status=?", (status,)).fetchall()
        return [r["id"] for r in rows]

    def update_run(self, run_id: str, **fields):
        for key in ("world", "result"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.db:
            self.db.execute(f"update runs set {cols} where id=?", (*fields.values(), run_id))

    def add_usage(self, run_id: str, prompt: int, completion: int, ms: int):
        with self.db:
            self.db.execute(
                "update runs set prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?, "
                "llm_ms=llm_ms+?, updated_at=? where id=?",
                (prompt, completion, ms, time.time(), run_id),
            )

    def messages(self, run_id: str) -> list[dict]:
        rows = self.db.execute("select body from messages where run_id=? order by seq", (run_id,)).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def append_messages(self, run_id: str, msgs: list[dict]):
        with self.db:
            start = self.db.execute(
                "select coalesce(max(seq),-1)+1 from messages where run_id=?", (run_id,)
            ).fetchone()[0]
            self.db.executemany(
                "insert into messages(run_id,seq,body) values(?,?,?)",
                [(run_id, start + i, json.dumps(m)) for i, m in enumerate(msgs)],
            )

    def replace_messages(self, run_id: str, msgs: list[dict]):
        with self.db:
            self.db.execute("delete from messages where run_id=?", (run_id,))
            self.db.executemany(
                "insert into messages(run_id,seq,body) values(?,?,?)",
                [(run_id, i, json.dumps(m)) for i, m in enumerate(msgs)],
            )

    def add_event(self, run_id: str, type_: str, data: dict) -> int:
        with self.db:
            cur = self.db.execute(
                "insert into events(run_id,ts,type,data) values(?,?,?,?)",
                (run_id, time.time(), type_, json.dumps(data)),
            )
        return cur.lastrowid

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        rows = self.db.execute(
            "select id,ts,type,data from events where run_id=? and id>? order by id", (run_id, after)
        ).fetchall()
        return [{"id": r["id"], "ts": r["ts"], "type": r["type"], "data": json.loads(r["data"])} for r in rows]

    def create_approval(self, run_id: str, call_id: str, tool: str, args: dict) -> dict:
        approval_id = uuid.uuid4().hex[:10]
        with self.db:
            self.db.execute(
                "insert into approvals(id,run_id,call_id,tool,args,status,created_at) values(?,?,?,?,?,?,?)",
                (approval_id, run_id, call_id, tool, json.dumps(args), "pending", time.time()),
            )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> dict | None:
        row = self._row(self.db.execute("select * from approvals where id=?", (approval_id,)).fetchone())
        if row:
            row["args"] = json.loads(row["args"])
        return row

    def approvals(self, run_id: str, status: str | None = None, unapplied: bool = False) -> list[dict]:
        sql = "select id from approvals where run_id=?"
        params: list = [run_id]
        if status:
            sql += " and status=?"
            params.append(status)
        if unapplied:
            sql += " and applied=0"
        rows = self.db.execute(sql + " order by created_at", params).fetchall()
        return [self.get_approval(r["id"]) for r in rows]

    def decide_approval(self, approval_id: str, approved: bool, reason: str | None):
        with self.db:
            self.db.execute(
                "update approvals set status=?, reason=?, decided_at=? where id=? and status='pending'",
                ("approved" if approved else "rejected", reason, time.time(), approval_id),
            )

    def mark_applied(self, approval_id: str):
        with self.db:
            self.db.execute("update approvals set applied=1 where id=?", (approval_id,))

    def push_inbox(self, run_id: str, body: str):
        with self.db:
            self.db.execute("insert into inbox(run_id,body) values(?,?)", (run_id, body))

    def drain_inbox(self, run_id: str) -> list[str]:
        with self.db:
            rows = self.db.execute("select id,body from inbox where run_id=? order by id", (run_id,)).fetchall()
            self.db.execute("delete from inbox where run_id=?", (run_id,))
        return [r["body"] for r in rows]

    def add_memory(self, service: str, text: str, ttl_days: int):
        now = time.time()
        with self.db:
            self.db.execute(
                "insert into memories(service,text,created_at,expires_at) values(?,?,?,?)",
                (service, text, now, now + ttl_days * 86400),
            )

    def recall(self, service: str, limit: int = 3) -> list[dict]:
        rows = self.db.execute(
            "select service,text,created_at from memories where service=? and expires_at>? "
            "order by created_at desc limit ?",
            (service, time.time(), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_memories(self) -> list[dict]:
        rows = self.db.execute(
            "select service,text,created_at,expires_at from memories where expires_at>? order by created_at desc",
            (time.time(),),
        ).fetchall()
        return [dict(r) for r in rows]
