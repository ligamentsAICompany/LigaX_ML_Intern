"""Cloud provider abstractions for executable infrastructure backends."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from huggingface_hub import HfApi

from agent.core.cost_estimation import CostEstimate, parse_timeout_hours
from agent.core.cost_estimation import hf_jobs_price_catalog as _hf_jobs_price_catalog
from agent.core.redact import scrub
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


async def hf_jobs_price_catalog() -> dict[str, float]:
    """Compatibility hook for HF Jobs price lookup in provider tests."""
    return await _hf_jobs_price_catalog()


@dataclass(frozen=True)
class ProviderContext:
    """Execution context shared with cloud provider adapters."""

    hf_token: str | None = None
    namespace: str | None = None
    log_callback: Callable[[str], Awaitable[None]] | None = None
    session: Any = None
    tool_call_id: str | None = None
    tool_name: str = "hf_jobs"
    provider_id: str = "hf-jobs"


class CloudProvider(Protocol):
    """Provider contract for cloud-backed job execution."""

    provider_id: str
    display_name: str
    enabled: bool
    executable: bool
    disabled_reason: str | None

    async def execute_tool(
        self, arguments: dict[str, Any], context: ProviderContext
    ) -> ToolResult:
        """Execute a tool operation through this provider."""

    async def estimate_cost(self, arguments: dict[str, Any]) -> CostEstimate:
        """Estimate billable cost for an operation."""

    async def cancel_jobs(self, job_ids: list[str], context: ProviderContext) -> None:
        """Cancel provider jobs tracked by an agent session."""


@dataclass(frozen=True)
class DisabledCloudProvider:
    """Placeholder for future providers that must not execute yet."""

    provider_id: str
    display_name: str
    disabled_reason: str
    enabled: bool = False
    executable: bool = False

    async def execute_tool(
        self, arguments: dict[str, Any], context: ProviderContext
    ) -> ToolResult:
        raise RuntimeError(f"{self.display_name} is disabled: {self.disabled_reason}")

    async def estimate_cost(self, arguments: dict[str, Any]) -> CostEstimate:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"{self.display_name} is disabled: {self.disabled_reason}",
            label=self.provider_id,
        )

    async def cancel_jobs(self, job_ids: list[str], context: ProviderContext) -> None:
        return None


@dataclass(frozen=True)
class HfJobsProvider:
    """Executable provider adapter for Hugging Face Jobs."""

    provider_id: str = "hf-jobs"
    display_name: str = "Hugging Face Jobs"
    enabled: bool = True
    executable: bool = True
    disabled_reason: str | None = None

    async def execute_tool(
        self, arguments: dict[str, Any], context: ProviderContext
    ) -> ToolResult:
        # Lazy import avoids a cycle: jobs_tool owns the public schema and the
        # legacy HfJobsTool implementation, while this adapter owns routing.
        from agent.tools.jobs_tool import HfJobsTool

        tool = HfJobsTool(
            namespace=context.namespace,
            hf_token=context.hf_token,
            log_callback=context.log_callback,
            session=context.session,
            tool_call_id=context.tool_call_id,
        )
        return await tool.execute(arguments)

    async def estimate_cost(self, arguments: dict[str, Any]) -> CostEstimate:
        flavor = str(
            arguments.get("hardware_flavor")
            or arguments.get("flavor")
            or arguments.get("hardware")
            or "cpu-basic"
        )
        timeout_hours = parse_timeout_hours(arguments.get("timeout"))
        if timeout_hours is None:
            return CostEstimate(
                estimated_cost_usd=None,
                billable=True,
                block_reason=f"Could not parse HF job timeout: {arguments.get('timeout')!r}.",
                label=flavor,
            )

        prices = await hf_jobs_price_catalog()
        price = prices.get(flavor)
        if price is None:
            return CostEstimate(
                estimated_cost_usd=None,
                billable=True,
                block_reason=f"No price is available for HF job hardware '{flavor}'.",
                label=flavor,
            )

        return CostEstimate(
            estimated_cost_usd=round(price * timeout_hours, 4),
            billable=price > 0,
            label=flavor,
        )

    async def cancel_jobs(self, job_ids: list[str], context: ProviderContext) -> None:
        if not job_ids:
            return

        api = HfApi(token=context.hf_token)
        for job_id in job_ids:
            try:
                await asyncio.to_thread(
                    api.cancel_job,
                    job_id=job_id,
                    namespace=context.namespace,
                )
                logger.info("Cancelled HF job %s on interrupt", job_id)
            except Exception as e:
                logger.warning("Failed to cancel HF job %s: %s", job_id, e)


@dataclass(frozen=True)
class PlanOnlyCloudProvider:
    """Active dry-run provider for future cloud execution backends."""

    provider_id: str
    display_name: str
    credential_env_vars: tuple[str, ...]
    training_service: str
    storage_service: str
    accelerator_hint: str
    enabled: bool = True
    executable: bool = False
    disabled_reason: str | None = None

    async def execute_tool(
        self, arguments: dict[str, Any], context: ProviderContext
    ) -> ToolResult:
        operation = str(arguments.get("operation") or "plan").strip().lower()
        if operation in {"plan", "validate", "run", "uv", "scheduled run"}:
            return {
                "formatted": self._format_plan(arguments, operation),
                "totalResults": 1,
                "resultsShared": 1,
                "isError": False,
            }
        if operation == "submit":
            return {
                "formatted": (
                    f"Real {self.display_name} execution is not enabled for Phase 10.\n\n"
                    f"No {self.display_name} resources were created and no spend was launched. "
                    "Use `operation: plan` or `operation: validate` for the current "
                    "plan-only preview workflow."
                ),
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }
        if operation in {"status", "logs", "cancel", "ps", "inspect"}:
            return {
                "formatted": (
                    f"{self.display_name} is active in plan-only mode. There are no "
                    f"{self.display_name} jobs to {operation} because Phase 10 does not "
                    "create provider resources."
                ),
                "totalResults": 0,
                "resultsShared": 0,
                "isError": False,
            }
        return {
            "formatted": (
                f"Unknown {self.display_name} operation: {operation!r}. Supported "
                "plan-only operations are plan, validate, run, status, logs, cancel, "
                "and submit. Submit is blocked until real execution is explicitly enabled."
            ),
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    async def estimate_cost(self, arguments: dict[str, Any]) -> CostEstimate:
        operation = str(arguments.get("operation") or "").strip().lower()
        if operation == "submit":
            return CostEstimate(
                estimated_cost_usd=None,
                billable=True,
                block_reason=(
                    f"Real {self.display_name} execution is not enabled; submit cannot "
                    "be costed safely."
                ),
                label=f"{self.provider_id}-submit",
            )
        return CostEstimate(
            estimated_cost_usd=0.0,
            billable=False,
            label=f"{self.provider_id}-plan-only",
        )

    async def cancel_jobs(self, job_ids: list[str], context: ProviderContext) -> None:
        return None

    def _credential_readiness(self) -> list[str]:
        readiness = []
        for name in self.credential_env_vars:
            status = "configured" if os.environ.get(name) else "missing"
            readiness.append(f"- {name}: {status}")
        return readiness

    def _format_plan(self, arguments: dict[str, Any], operation: str) -> str:
        sanitized = scrub(arguments)
        mode = (
            "dry-run submit"
            if operation in {"run", "uv", "scheduled run"}
            else operation
        )
        script = sanitized.get("script")
        command = sanitized.get("command")
        hardware = (
            sanitized.get("hardware_flavor")
            or sanitized.get("flavor")
            or sanitized.get("hardware")
            or self.accelerator_hint
        )
        timeout = sanitized.get("timeout") or "provider default"
        workload = (
            "Python script"
            if script
            else "container command"
            if command
            else "training workload"
        )

        lines = [
            f"## {self.display_name} plan-only training plan",
            "",
            f"Requested operation: {mode}",
            f"Workload shape: {workload}",
            f"Target training service: {self.training_service}",
            f"Artifact/checkpoint storage: {self.storage_service}",
            f"Requested accelerator or flavor: {hardware}",
            f"Requested timeout: {timeout}",
            "",
            "Credential readiness:",
            *self._credential_readiness(),
            "",
            "Execution guardrails:",
            f"- No {self.display_name} resources were created.",
            "- No cloud job was submitted and no spend was launched.",
            "- Real execution remains disabled until explicit provider credentials, "
            "region/project settings, and approval policy enablement are added.",
            "",
            "Next implementation steps when execution is enabled:",
            "- Map the training script or container command into a provider job spec.",
            "- Attach a private artifact bucket/container and log sink.",
            "- Require manual approval with an estimated provider cost before submit.",
        ]
        return "\n".join(lines)


class CloudProviderRegistry:
    """Registry of supported cloud providers."""

    def __init__(self, providers: list[CloudProvider]):
        self._providers = {provider.provider_id: provider for provider in providers}

    def all(self) -> list[CloudProvider]:
        return list(self._providers.values())

    def require(self, provider_id: str) -> CloudProvider:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise KeyError(f"Unknown cloud provider: {provider_id}")
        return provider

    def require_executable(self, provider_id: str) -> CloudProvider:
        provider = self.require(provider_id)
        if not provider.enabled or not provider.executable:
            reason = provider.disabled_reason or "provider is not executable"
            raise RuntimeError(f"{provider.display_name} is not executable: {reason}")
        return provider


_REGISTRY = CloudProviderRegistry(
    [
        HfJobsProvider(),
        PlanOnlyCloudProvider(
            provider_id="aws",
            display_name="AWS",
            credential_env_vars=(
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION",
                "AWS_S3_BUCKET",
            ),
            training_service="SageMaker Training Jobs or AWS Batch",
            storage_service="S3",
            accelerator_hint="ml.g5 / ml.p4 class GPU instance",
        ),
        PlanOnlyCloudProvider(
            provider_id="azure",
            display_name="Azure",
            credential_env_vars=(
                "AZURE_CLIENT_ID",
                "AZURE_TENANT_ID",
                "AZURE_SUBSCRIPTION_ID",
                "AZURE_RESOURCE_GROUP",
                "AZURE_STORAGE_ACCOUNT",
            ),
            training_service="Azure Machine Learning Jobs",
            storage_service="Azure Blob Storage",
            accelerator_hint="NC/ND GPU compute cluster",
        ),
        PlanOnlyCloudProvider(
            provider_id="gcp",
            display_name="Google Cloud",
            credential_env_vars=(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GCP_PROJECT_ID",
                "GCP_REGION",
                "GCS_BUCKET",
            ),
            training_service="Vertex AI Custom Jobs",
            storage_service="Google Cloud Storage",
            accelerator_hint="A2/G2 GPU machine type",
        ),
    ]
)


def get_cloud_provider_registry() -> CloudProviderRegistry:
    return _REGISTRY
