"""Phase 6 stabilization tests for deterministic approval-before-spend flow."""

import sys
from pathlib import Path

import yaml

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import routes.agent as agent_routes  # noqa: E402
from agent.core import agent_loop  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _system_prompt() -> str:
    prompt_path = _REPO_ROOT / "agent" / "prompts" / "system_prompt_v3.yaml"
    return yaml.safe_load(prompt_path.read_text())["system_prompt"]


def test_system_prompt_has_phase6_approval_before_spend_fast_path():
    prompt = _system_prompt()

    assert "Phase 6 stabilization fast path" in prompt
    assert "approval required before spend" in prompt
    assert "Do not call `research`" in prompt
    assert "the next assistant turn must be an `hf_jobs` tool call" in prompt
    assert "Do not call bash, read, write, or edit before sandbox_create" in prompt


def test_call_center_context_adds_fast_approval_path_instructions():
    submitted = agent_routes._compose_submitted_text(
        "I want to fine-tune a small model for a Call Center support assistant.",
        (
            "Selected ML workflow context:\n"
            "- Vertical: Call Center\n"
            "- Compute provider: HF Jobs"
        ),
    )

    assert "Phase 6 approval-before-spend fast path" in submitted
    assert "bitext/Bitext-customer-support-llm-chatbot-training-dataset" in submitted
    assert "Stop when approval is required" in submitted


def test_call_center_fast_path_hf_jobs_tool_call_is_compact_and_valid():
    tool_call, args = agent_loop._call_center_fast_path_hf_jobs_tool_call()

    assert tool_call.function.name == "hf_jobs"
    assert tool_call.function.arguments == (
        '{"operation":"run","command":["python","-c","print(\\"approval smoke\\")"],'
        '"image":"python:3.12","hardware_flavor":"t4-small","timeout":"30m"}'
    )
    assert args == {
        "operation": "run",
        "command": ["python", "-c", 'print("approval smoke")'],
        "image": "python:3.12",
        "hardware_flavor": "t4-small",
        "timeout": "30m",
    }
