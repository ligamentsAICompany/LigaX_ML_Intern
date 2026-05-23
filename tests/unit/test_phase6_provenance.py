"""Phase 6 Trackio and Hub provenance readiness tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.core.dataset_inspection import build_dataset_profile
from agent.core.provenance import (
    build_artifact_card,
    build_training_job_metadata,
    build_training_provenance,
)
from agent.core.telemetry import record_hf_job_submit
from agent.tools.jobs_tool import HfJobsTool


def _income_tax_profile() -> dict:
    return build_dataset_profile(
        rows=[
            {
                "Section": "80C",
                "Description": "Deduction for eligible investments",
                "Limit": "150000",
                "Applicability": "Individuals and HUF",
            },
            {
                "Section": "80D",
                "Description": "Medical insurance deduction",
                "Limit": "25000",
                "Applicability": "Individuals",
            },
        ],
        source={
            "type": "local_file",
            "path": "data/Income_Tax_Master.xlsx",
            "format": "xlsx",
            "repo_id": "tax-org/income-tax-reference",
        },
    )


def test_income_tax_provenance_includes_quality_intelligence_and_limitations():
    payload = build_training_provenance(
        base_model="google/gemma-2-2b-it",
        dataset_profile=_income_tax_profile(),
        training_method="rag_hybrid_baseline",
        cost={"estimated_cost_usd": 0.72},
        hardware={"flavor": "t4-small"},
        timeout="4h",
        limitations=["Reference table requires grounded retrieval before SFT."],
    )

    assert payload["base_model"] == "google/gemma-2-2b-it"
    assert payload["dataset"]["repo_id"] == "tax-org/income-tax-reference"
    assert payload["dataset"]["path"] == "data/Income_Tax_Master.xlsx"
    assert payload["dataset"]["row_count"] == 2
    assert payload["trainability"]["risk_level"] == "high"
    assert payload["strategy"]["strategy"] in {"rag", "hybrid"}
    assert payload["golden_eval"]["case_count"] >= 1
    assert "quality_constraints" in payload["golden_eval"]
    assert payload["limitations"] == [
        "Reference table requires grounded retrieval before SFT."
    ]
    assert payload["training"]["method"] == "rag_hybrid_baseline"
    assert payload["training"]["timeout"] == "4h"


def test_training_job_metadata_includes_trackio_fields_or_warning():
    with_trackio = build_training_job_metadata(
        {
            "dependencies": ["transformers", "trl", "trackio"],
            "trackio": {
                "project": "income-tax-sft",
                "space_id": "tax-org/trackio-dashboard",
            },
        }
    )
    without_trackio = build_training_job_metadata({"dependencies": ["trl"]})

    assert with_trackio["trackio"]["enabled"] is True
    assert with_trackio["trackio"]["project"] == "income-tax-sft"
    assert with_trackio["trackio"]["space_id"] == "tax-org/trackio-dashboard"
    assert with_trackio["warnings"] == []
    assert without_trackio["trackio"]["enabled"] is False
    assert "missing_trackio" in without_trackio["warnings"]


def test_training_job_metadata_detects_common_trackio_dependency_specs():
    for dependency in (
        "trackio>=0.2.0",
        " TrackIO ~= 0.3 ",
        "trackio[spaces]>=0.4",
        "trackio ; python_version >= '3.11'",
    ):
        metadata = build_training_job_metadata({"dependencies": ["trl", dependency]})

        assert metadata["trackio"]["enabled"] is True
        assert metadata["trackio"]["dependency"] is True
        assert metadata["warnings"] == []


def test_training_job_metadata_scrubs_nested_trackio_and_provenance_secrets():
    metadata = build_training_job_metadata(
        {
            "dependencies": ["trackio"],
            "trackio": {
                "project": "income-tax-sft",
                "nested": {"api_key": "sk-" + "a" * 40},
            },
            "provenance": {
                "dataset": {
                    "repo_id": "tax/private",
                    "auth": {"HF_TOKEN": "hf_" + "b" * 30},
                }
            },
        }
    )
    serialized = json.dumps(metadata)

    assert "sk-" + "a" * 40 not in serialized
    assert "hf_" + "b" * 30 not in serialized
    assert "[REDACTED_SECRET]" in serialized


def test_artifact_card_handles_missing_and_empty_provenance_fields():
    card = build_artifact_card({}, artifact_type="model")

    assert "Base model: Not specified" in card
    assert "Dataset: Not specified" in card
    assert "Method: Not specified" in card
    assert "Not specified." in card


def test_artifact_card_mentions_model_dataset_method_limitations_and_eval_summary():
    card = build_artifact_card(
        build_training_provenance(
            base_model="google/gemma-2-2b-it",
            dataset_profile=_income_tax_profile(),
            training_method="sft_lora",
            limitations=[
                "Only two source rows were available for this readiness check."
            ],
        ),
        artifact_type="model",
    )

    assert "google/gemma-2-2b-it" in card
    assert "tax-org/income-tax-reference" in card
    assert "sft_lora" in card
    assert "Only two source rows" in card
    assert "Golden Eval" in card
    assert "Trainability Risk" in card


def test_provenance_and_cards_redact_hf_aws_github_and_bearer_tokens():
    payload = build_training_provenance(
        base_model="org/model",
        dataset_profile={
            **_income_tax_profile(),
            "source": {
                "type": "local_file",
                "path": "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890",
                "repo_id": "repo-with-ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcd",
            },
        },
        training_method="sft",
        cost={"note": "Bearer abcdefghijklmnopqrstuvwxyz123456"},
        hardware={"aws": "AKIAABCDEFGHIJKLMNOP"},
        limitations=["github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_extra"],
    )
    card = build_artifact_card(payload, artifact_type="dataset")
    serialized = json.dumps({"payload": payload, "card": card})

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "AKIAABCDEFGHIJKLMNOP" not in serialized
    assert "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_extra" not in serialized
    assert "Bearer abcdefghijklmnopqrstuvwxyz123456" not in serialized
    assert "[REDACTED_HF_TOKEN]" in serialized
    assert "[REDACTED_AWS_KEY_ID]" in serialized
    assert "[REDACTED_GITHUB_TOKEN]" in serialized
    assert "Bearer [REDACTED]" in serialized


@pytest.mark.asyncio
async def test_hf_job_submit_telemetry_includes_scrubbed_provenance_fields():
    class FakeSession:
        def __init__(self):
            self.events = []

        async def send_event(self, event):
            self.events.append(event)

    session = FakeSession()
    provenance = build_training_provenance(
        base_model="google/gemma-2-2b-it",
        dataset_profile=_income_tax_profile(),
        training_method="sft_lora",
        limitations=["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890"],
    )

    await record_hf_job_submit(
        session,
        SimpleNamespace(id="job1", url="https://hf.co/jobs/job1"),
        {
            "hardware_flavor": "t4-small",
            "timeout": "4h",
            "script": "SFTConfig(push_to_hub=True, hub_model_id='tax/model')",
            "namespace": "tax-org",
            "provenance": provenance,
            "trackio": {"project": "income-tax-sft"},
        },
        image="ghcr.io/astral-sh/uv:python3.12-bookworm",
        job_type="Python",
    )

    event = session.events[0]
    serialized = json.dumps(event.data)
    assert event.event_type == "hf_job_submit"
    assert event.data["provenance"]["base_model"] == "google/gemma-2-2b-it"
    assert event.data["provenance"]["dataset"]["row_count"] == 2
    assert event.data["provenance"]["strategy"]["strategy"] in {"rag", "hybrid"}
    assert event.data["trackio"]["project"] == "income-tax-sft"
    assert event.data["push_to_hub"] is True
    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized


@pytest.mark.asyncio
async def test_hf_job_submit_does_not_infer_persistence_from_hub_model_id_only():
    class FakeSession:
        def __init__(self):
            self.events = []

        async def send_event(self, event):
            self.events.append(event)

    session = FakeSession()

    await record_hf_job_submit(
        session,
        SimpleNamespace(id="job1", url="https://hf.co/jobs/job1"),
        {
            "script": "SFTConfig(push_to_hub=False, hub_model_id='tax/model')",
        },
        image="ghcr.io/astral-sh/uv:python3.12-bookworm",
        job_type="Python",
    )

    assert session.events[0].data["push_to_hub"] is False


@pytest.mark.asyncio
async def test_hf_jobs_returned_error_strings_are_scrubbed():
    class FakeApi:
        def fetch_job_logs(self, **_kwargs):
            raise RuntimeError("fetch failed with HF_TOKEN=hf_" + "c" * 30)

    tool = HfJobsTool()
    tool.api = FakeApi()

    result = await tool._get_logs({"job_id": "job-1"})

    assert result["isError"] is True
    assert "hf_" + "c" * 30 not in result["formatted"]
    assert "[REDACTED_HF_TOKEN]" in result["formatted"]
