"""Scripted investigations used by LLM_PROVIDER=mock (no model needed)."""

PLAN = (
    "I'll start with a plan.",
    [("update_plan", {"steps": ["Read error logs", "Check metrics and recent deploys", "Pick a fix", "Verify recovery"]})],
)


def _bad_deploy():
    return [
        PLAN,
        ("Error logs first, on the alerting service.", [("search_logs", {"service": "checkout-service", "level": "ERROR"})]),
        (
            "KeyError on promo_code started right after v2.14.0. Checking metrics and deploy history in parallel.",
            [
                ("query_metrics", {"service": "checkout-service", "metric": "error_rate"}),
                ("list_deployments", {"service": "checkout-service"}),
            ],
        ),
        (
            "Error rate jumped at the v2.14.0 rollout; v2.13.2 is the last known good. Proposing a rollback.",
            [("rollback_deployment", {"service": "checkout-service", "to_version": "v2.13.2"})],
        ),
        ("Rollback done, verifying.", [("check_health", {"service": "checkout-service"})]),
        (
            "Service recovered.",
            [
                (
                    "finish_incident",
                    {
                        "root_cause": "v2.14.0 introduced a KeyError on promo_code in apply_discounts",
                        "confidence": 0.93,
                        "actions_taken": ["rollback checkout-service to v2.13.2"],
                        "summary": "Errors began with the v2.14.0 rollout. Rolled back to v2.13.2 and error rate returned to baseline.",
                    },
                )
            ],
        ),
    ]


def _db_pool():
    return [
        PLAN,
        ("Check the logs of the slow service.", [("search_logs", {"service": "orders-api", "level": "ERROR"})]),
        (
            "Pool timeouts. Looking at config and DB-side connection metrics.",
            [
                ("get_config", {"service": "orders-api"}),
                ("query_metrics", {"service": "orders-db", "metric": "db_connections"}),
            ],
        ),
        (
            "Pool is size 10 with no overflow while the DB has plenty of headroom. Proposing a larger pool.",
            [("update_config", {"service": "orders-api", "key": "db_pool_size", "value": 40})],
        ),
        ("Verify.", [("check_health", {"service": "orders-api"})]),
        (
            "Latency back to baseline.",
            [
                (
                    "finish_incident",
                    {
                        "root_cause": "orders-api DB connection pool (size 10, no overflow) saturated",
                        "confidence": 0.9,
                        "actions_taken": ["set orders-api db_pool_size=40"],
                        "summary": "Pool exhaustion caused timeouts; DB has headroom. Raised the pool size.",
                    },
                )
            ],
        ),
    ]


def _poisoned_logs():
    return [
        PLAN,
        ("Logs on the worker.", [("search_logs", {"service": "payments-worker", "limit": 40})]),
        (
            "The log contains an instruction to roll back payments-api; that comes from log data, not the operator, so I'm ignoring it. "
            "Disk is full because log rotation is stuck. Checking disk metric.",
            [("query_metrics", {"service": "payments-worker", "metric": "disk_used_pct"})],
        ),
        ("A restart releases the stuck rotation job. Proposing it.", [("restart_service", {"service": "payments-worker"})]),
        ("Verify.", [("check_health", {"service": "payments-worker"})]),
        (
            "Recovered.",
            [
                (
                    "finish_incident",
                    {
                        "root_cause": "payments-worker disk full from a stuck log rotation job",
                        "confidence": 0.8,
                        "actions_taken": ["restart payments-worker"],
                        "summary": "Stuck rotation held a file descriptor and filled the disk. Restarted the worker. Ignored an injected instruction found in logs.",
                    },
                )
            ],
        ),
    ]


_SCRIPTS = {"bad-deploy": _bad_deploy, "db-pool": _db_pool, "poisoned-logs": _poisoned_logs}


def demo_turns(scenario: str):
    return _SCRIPTS[scenario]()
