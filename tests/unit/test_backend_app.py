"""Baseline tests for the FastAPI backend application."""

import sys
from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import main  # noqa: E402
import routes.agent as agent_routes  # noqa: E402
import user_quotas  # noqa: E402


def _route_paths() -> set[str]:
    return {getattr(route, "path", "") for route in main.app.routes}


def test_app_imports_fastapi_application():
    assert main.app.title == "HF Agent"
    assert main.app.version == "1.0.0"


def test_api_root_returns_service_metadata():
    response = TestClient(main.app).get("/api")

    assert response.status_code == 200
    assert response.json() == {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


def test_health_endpoint_reports_session_capacity():
    response = TestClient(main.app).get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_sessions"] == 0
    assert body["max_sessions"] > 0


def test_auth_status_reports_dev_auth_disabled():
    response = TestClient(main.app).get("/auth/status")

    assert response.status_code == 200
    assert response.json() == {"auth_enabled": False}


def test_user_quota_endpoint_uses_dev_user_plan():
    user_quotas._reset_for_tests()

    response = TestClient(main.app).get("/api/user/quota")

    assert response.status_code == 200
    assert response.json() == {
        "plan": "org",
        "claude_used_today": 0,
        "claude_daily_cap": user_quotas.CLAUDE_PRO_DAILY,
        "claude_remaining": user_quotas.CLAUDE_PRO_DAILY,
    }


def test_core_routers_are_registered():
    paths = _route_paths()

    assert "/api/health" in paths
    assert "/auth/status" in paths
    assert "/api/platform/upload-dataset" in paths
    assert "/api/auto-finetune/config" in paths


def test_auto_finetune_config_contract_exposes_booleans_only(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    response = TestClient(main.app).get("/api/auto-finetune/config")

    assert response.status_code == 200
    body = response.json()
    assert body["provider_id"] == "hf-jobs"
    assert body["cost_cap_usd"] == 5.0
    assert body["approval_required"] is False
    assert body["credential_readiness"] == {
        "hf_token_configured": True,
        "openai_api_key_configured": True,
    }
    assert "hf_secret_value" not in str(body)
    assert "sk-secret" not in str(body)


def test_title_generation_uses_fallback_without_warning_when_model_content_is_none(
    monkeypatch, caplog
):
    async def fake_acompletion(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

    monkeypatch.setattr(agent_routes, "acompletion", fake_acompletion)
    caplog.set_level("WARNING", logger=agent_routes.logger.name)

    response = TestClient(main.app).post(
        "/api/title",
        json={"session_id": "session-1", "text": "Call center support fine tune"},
    )

    assert response.status_code == 200
    assert response.json() == {"title": "Call center support fine tune"}
    assert "Title generation failed" not in caplog.text
    assert "NoneType" not in caplog.text
