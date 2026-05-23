"""Admin CLI entry points for operational hardening checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.operational_hardening import (  # noqa: E402
    build_kpi_snapshot,
    build_provider_health_snapshot,
    sweep_orphan_jobs,
)
from agent.core.session_persistence import get_session_store  # noqa: E402


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


async def _with_store(callback):
    store = get_session_store()
    await store.init()
    try:
        return await callback(store)
    finally:
        await store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hf-admin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Print provider readiness health")
    subparsers.add_parser("kpi", help="Print operational KPI snapshot")

    sweep = subparsers.add_parser("orphan-sweep", help="Inspect or cancel orphan jobs")
    sweep.add_argument(
        "--apply",
        action="store_true",
        help="Cancel safe orphan candidates; default is dry-run",
    )
    sweep.add_argument(
        "--stale-hours",
        type=int,
        default=24,
        help="Idle persisted sessions older than this may be swept",
    )

    subparsers.add_parser("release-smoke", help="Run local release smoke checks")
    return parser


async def run_command(args: argparse.Namespace) -> int:
    if args.command == "health":
        _print_json(build_provider_health_snapshot(hf_token=os.environ.get("HF_TOKEN")))
        return 0

    if args.command == "kpi":
        payload = await _with_store(lambda store: build_kpi_snapshot(store))
        _print_json(payload)
        return 0

    if args.command == "orphan-sweep":
        payload = await _with_store(
            lambda store: sweep_orphan_jobs(
                store,
                apply=args.apply,
                hf_token=os.environ.get("HF_TOKEN"),
                stale_hours=args.stale_hours,
            )
        )
        _print_json(payload)
        return 1 if payload.get("errors") else 0

    if args.command == "release-smoke":
        from fastapi.testclient import TestClient

        import main

        client = TestClient(main.app)
        endpoints = {}
        for path in ("/api", "/api/health", "/api/health/providers"):
            response = client.get(path)
            endpoints[path] = {
                "status_code": response.status_code,
                "ok": response.status_code == 200,
            }
        provider_health = build_provider_health_snapshot(
            hf_token=os.environ.get("HF_TOKEN")
        )
        kpi = await _with_store(lambda store: build_kpi_snapshot(store))
        payload = {
            "status": "ok"
            if all(item["ok"] for item in endpoints.values())
            else "error",
            "endpoints": endpoints,
            "provider_health_status": provider_health["status"],
            "kpi": kpi,
        }
        _print_json(payload)
        return 0 if payload["status"] == "ok" else 1

    raise ValueError(f"Unknown command: {args.command}")


def main_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_command(args))


def main() -> None:
    raise SystemExit(main_cli())


if __name__ == "__main__":
    main()
