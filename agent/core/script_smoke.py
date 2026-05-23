"""Static Python training script smoke checks before billable jobs."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from agent.core.cost_estimation import parse_timeout_hours

IssueSeverity = Literal["error", "warning"]
SmokeStatus = Literal["pass", "warn", "fail"]

_TRAINING_CALL_NAMES = {
    "Trainer",
    "Seq2SeqTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "TrainingArguments",
    "Seq2SeqTrainingArguments",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
}
_TRAINING_CONFIG_CALL_NAMES = {
    "TrainingArguments",
    "Seq2SeqTrainingArguments",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
}
_TRACKIO_MARKERS = {"trackio", "trackio.init", "trackio.init_from_env"}
_TRAINING_TIMEOUT_MIN_HOURS = 2.0
_TRAINING_PATH_RE = re.compile(
    r"(^|[\\/\s_.-])(?:train|trainer|finetune|fine[-_]?tune)(?:\.py|[\\/\s_.-]|$)",
    re.I,
)


@dataclass(frozen=True)
class ScriptSmokeIssue:
    """One static smoke finding."""

    severity: IssueSeverity
    code: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptSmokeResult:
    """Structured script smoke result for approval and tool metadata."""

    status: SmokeStatus
    is_training_script: bool
    issues: list[ScriptSmokeIssue]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    @property
    def blocking_issues(self) -> list[ScriptSmokeIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ScriptSmokeIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "is_training_script": self.is_training_script,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def run_script_smoke(
    script: str,
    *,
    job_args: Mapping[str, Any] | None = None,
) -> ScriptSmokeResult:
    """Run deterministic static checks on inline Python script content."""

    job_args = job_args or {}
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return ScriptSmokeResult(
            status="fail",
            is_training_script=_looks_like_training_text(script, job_args),
            issues=[
                ScriptSmokeIssue(
                    severity="error",
                    code="python_syntax_error",
                    message=f"Python syntax error: {e.msg}",
                    line=e.lineno,
                )
            ],
        )

    detector = _TrainingScriptDetector()
    detector.visit(tree)
    is_training = detector.is_training_script or _looks_like_training_text(
        script, job_args
    )
    issues: list[ScriptSmokeIssue] = []

    if is_training:
        issues.extend(_hub_persistence_issues(tree))
        issues.extend(_trl_config_compatibility_issues(tree))
        issues.extend(_timeout_issues(job_args))
        issues.extend(_trackio_issues(tree, script, job_args))

    return ScriptSmokeResult(
        status=_status_for(issues),
        is_training_script=is_training,
        issues=issues,
    )


def is_inline_python_script(script: str) -> bool:
    """Return True when a script value is inline Python rather than a path or URL."""

    stripped = script.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(("http://", "https://", "/", "./", "../")):
        return False
    if re.search(r"(?:^|[\\/])[\w.-]+\.py$", stripped):
        return False
    return (
        "\n" in stripped
        or stripped.startswith(("import ", "from ", "def ", "class ", "print(", "#"))
        or "trainer.train" in lowered
        or "trainingarguments" in lowered
        or "sftconfig" in lowered
        or "sfttrainer" in lowered
    )


def coerce_script_smoke_result(value: Mapping[str, Any]) -> ScriptSmokeResult:
    """Coerce serialized smoke/preflight metadata back into a result object."""

    raw_issues = value.get("issues")
    issues = []
    if isinstance(raw_issues, list):
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, Mapping):
                continue
            severity = str(raw_issue.get("severity") or "").lower()
            if severity not in {"error", "warning"}:
                continue
            line = raw_issue.get("line")
            issues.append(
                ScriptSmokeIssue(
                    severity=severity,  # type: ignore[arg-type]
                    code=str(raw_issue.get("code") or "script_smoke_issue"),
                    message=str(raw_issue.get("message") or "Script smoke issue."),
                    line=line if isinstance(line, int) else None,
                )
            )

    status = str(value.get("status") or "").lower()
    derived_status = _status_for(issues)
    if derived_status == "fail":
        status = "fail"
    elif derived_status == "warn" and status == "pass":
        status = "warn"
    elif status not in {"pass", "warn", "fail"}:
        status = derived_status

    return ScriptSmokeResult(
        status=status,  # type: ignore[arg-type]
        is_training_script=bool(value.get("is_training_script")),
        issues=issues,
    )


def unverified_script_smoke_result(
    script: str,
    *,
    job_args: Mapping[str, Any] | None = None,
) -> ScriptSmokeResult | None:
    """Return a blocking result when a training script cannot be statically checked."""

    job_args = job_args or {}
    if not _looks_like_unresolved_training_script(script, job_args):
        return None
    return ScriptSmokeResult(
        status="fail",
        is_training_script=True,
        issues=[
            ScriptSmokeIssue(
                severity="error",
                code="script_smoke_unverified",
                message=(
                    "Training script content could not be resolved for static smoke "
                    "validation. Resolve the file/URL to inline content before spend "
                    "or require explicit manual approval for an unverified training run."
                ),
            )
        ],
    )


def format_script_smoke_result(result: ScriptSmokeResult) -> str:
    """Render a concise smoke summary for tool output and approval reasons."""

    label = "passed" if result.passed else "failed"
    lines = [
        f"Script smoke {label} ({result.status}); "
        f"training script: {'yes' if result.is_training_script else 'no'}."
    ]
    for issue in result.issues:
        location = f" line {issue.line}" if issue.line else ""
        lines.append(
            f"- {issue.severity.upper()} {issue.code}{location}: {issue.message}"
        )
    return "\n".join(lines)


def _status_for(issues: list[ScriptSmokeIssue]) -> SmokeStatus:
    if any(issue.severity == "error" for issue in issues):
        return "fail"
    if issues:
        return "warn"
    return "pass"


def _hub_persistence_issues(tree: ast.AST) -> list[ScriptSmokeIssue]:
    issues: list[ScriptSmokeIssue] = []
    config_calls = _training_config_calls(tree)
    if not config_calls:
        return [
            ScriptSmokeIssue(
                severity="error",
                code="hub_persistence_unverified",
                message=(
                    "Training is visible, but no TrainingArguments/SFTConfig-style "
                    "configuration call was found to verify push_to_hub=True and "
                    "hub_model_id before spend."
                ),
            )
        ]

    config_kwargs = _hub_config_kwargs(config_calls)
    if config_kwargs.unverified:
        return [
            ScriptSmokeIssue(
                severity="error",
                code="hub_persistence_unverified",
                message=(
                    "Training config uses dynamic kwargs, so push_to_hub=True and "
                    "hub_model_id cannot be verified statically before spend."
                ),
                line=config_kwargs.line,
            )
        ]

    push_value = config_kwargs.values.get("push_to_hub")
    hub_model_id = config_kwargs.values.get("hub_model_id")

    if push_value is not True:
        code = "missing_push_to_hub" if push_value is None else "push_to_hub_not_true"
        issues.append(
            ScriptSmokeIssue(
                severity="error",
                code=code,
                message=(
                    "Training jobs must set push_to_hub=True so artifacts survive "
                    "ephemeral HF Jobs storage."
                ),
                line=config_kwargs.lines.get("push_to_hub"),
            )
        )

    if not _non_empty_literal(hub_model_id):
        issues.append(
            ScriptSmokeIssue(
                severity="error",
                code="missing_hub_model_id",
                message="Training jobs must set a non-empty hub_model_id.",
                line=config_kwargs.lines.get("hub_model_id"),
            )
        )

    return issues


def _training_config_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node.func) in _TRAINING_CONFIG_CALL_NAMES
    ]


def _trl_config_compatibility_issues(tree: ast.AST) -> list[ScriptSmokeIssue]:
    issues: list[ScriptSmokeIssue] = []
    for call in _training_config_calls(tree):
        if _call_name(call.func) != "SFTConfig":
            continue
        for keyword in call.keywords:
            if keyword.arg == "max_seq_length":
                issues.append(_unsupported_sft_max_seq_length_issue(keyword.value))
            elif keyword.arg == "evaluation_strategy":
                issues.append(_unsupported_sft_evaluation_strategy_issue(keyword.value))
            elif keyword.arg is None:
                issues.extend(_literal_kwargs_compatibility_issues(keyword.value))
    issues.extend(_trl_trainer_compatibility_issues(tree))
    return issues


def _literal_kwargs_compatibility_issues(node: ast.AST) -> list[ScriptSmokeIssue]:
    try:
        kwargs = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(kwargs, Mapping):
        return []
    issues: list[ScriptSmokeIssue] = []
    if "max_seq_length" in kwargs:
        issues.append(_unsupported_sft_max_seq_length_issue(node))
    if "evaluation_strategy" in kwargs:
        issues.append(_unsupported_sft_evaluation_strategy_issue(node))
    return issues


def _trl_trainer_compatibility_issues(tree: ast.AST) -> list[ScriptSmokeIssue]:
    issues: list[ScriptSmokeIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "SFTTrainer":
            continue
        for keyword in node.keywords:
            if keyword.arg == "tokenizer":
                issues.append(_unsupported_sft_trainer_tokenizer_issue(keyword.value))
            elif keyword.arg is None:
                issues.extend(
                    _literal_trainer_kwargs_compatibility_issues(keyword.value)
                )
    return issues


def _literal_trainer_kwargs_compatibility_issues(
    node: ast.AST,
) -> list[ScriptSmokeIssue]:
    try:
        kwargs = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(kwargs, Mapping) or "tokenizer" not in kwargs:
        return []
    return [_unsupported_sft_trainer_tokenizer_issue(node)]


def _unsupported_sft_max_seq_length_issue(node: ast.AST) -> ScriptSmokeIssue:
    return ScriptSmokeIssue(
        severity="error",
        code="unsupported_sft_config_kwarg",
        message=(
            "SFTConfig no longer accepts max_seq_length; use max_length instead "
            "before submitting the HF Job."
        ),
        line=getattr(node, "lineno", None),
    )


def _unsupported_sft_evaluation_strategy_issue(node: ast.AST) -> ScriptSmokeIssue:
    return ScriptSmokeIssue(
        severity="error",
        code="unsupported_sft_config_kwarg",
        message=(
            "SFTConfig in current TRL accepts eval_strategy, not "
            "evaluation_strategy, before submitting the HF Job."
        ),
        line=getattr(node, "lineno", None),
    )


def _unsupported_sft_trainer_tokenizer_issue(node: ast.AST) -> ScriptSmokeIssue:
    return ScriptSmokeIssue(
        severity="error",
        code="unsupported_sft_trainer_kwarg",
        message=(
            "SFTTrainer no longer accepts tokenizer; use processing_class instead "
            "before submitting the HF Job."
        ),
        line=getattr(node, "lineno", None),
    )


def _timeout_issues(job_args: Mapping[str, Any]) -> list[ScriptSmokeIssue]:
    timeout_hours = parse_timeout_hours(job_args.get("timeout"))
    if timeout_hours is None:
        return [
            ScriptSmokeIssue(
                severity="warning",
                code="training_timeout_unparseable",
                message="Training timeout could not be parsed; use an explicit value like '8h'.",
            )
        ]
    if timeout_hours < _TRAINING_TIMEOUT_MIN_HOURS:
        return [
            ScriptSmokeIssue(
                severity="warning",
                code="training_timeout_too_short",
                message=(
                    "Training timeout is under 2h; HF Jobs default/short timeouts "
                    "commonly kill training mid-run."
                ),
            )
        ]
    return []


def _trackio_issues(
    tree: ast.AST,
    script: str,
    job_args: Mapping[str, Any],
) -> list[ScriptSmokeIssue]:
    issues = _trackio_compatibility_issues(tree)
    if issues:
        return issues
    if _reporting_disabled(tree):
        return []

    dependencies = job_args.get("dependencies")
    dep_text = " ".join(str(dep).lower() for dep in dependencies or [])
    text = script.lower()
    has_marker = any(marker in text for marker in _TRACKIO_MARKERS)
    has_dependency = "trackio" in dep_text
    if has_marker and has_dependency:
        return []
    return [
        ScriptSmokeIssue(
            severity="warning",
            code="missing_trackio",
            message=(
                "Training jobs should include Trackio dependency and initialization "
                "so paid runs expose monitoring."
            ),
        )
    ]


def _trackio_compatibility_issues(tree: ast.AST) -> list[ScriptSmokeIssue]:
    issues: list[ScriptSmokeIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_trackio_init_call(node.func):
            continue
        for keyword in node.keywords:
            if keyword.arg in {"project_name", "run_name"}:
                issues.append(
                    _unsupported_trackio_init_kwarg_issue(keyword.arg, keyword.value)
                )
            elif keyword.arg is None:
                issues.extend(
                    _literal_trackio_kwargs_compatibility_issues(keyword.value)
                )
    return issues


def _reporting_disabled(tree: ast.AST) -> bool:
    for call in _training_config_calls(tree):
        for keyword in call.keywords:
            if keyword.arg == "report_to":
                return _is_reporting_disabled_value(_literal_or_dynamic(keyword.value))
            if keyword.arg is None:
                try:
                    kwargs = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(kwargs, Mapping) and "report_to" in kwargs:
                    return _is_reporting_disabled_value(kwargs["report_to"])
    return False


def _is_reporting_disabled_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"none", ""}
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _is_trackio_init_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "init"
        and isinstance(node.value, ast.Name)
        and node.value.id == "trackio"
    )


def _literal_trackio_kwargs_compatibility_issues(
    node: ast.AST,
) -> list[ScriptSmokeIssue]:
    try:
        kwargs = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(kwargs, Mapping):
        return []
    return [
        _unsupported_trackio_init_kwarg_issue(name, node)
        for name in ("project_name", "run_name")
        if name in kwargs
    ]


def _unsupported_trackio_init_kwarg_issue(
    name: str,
    node: ast.AST,
) -> ScriptSmokeIssue:
    replacement = "project" if name == "project_name" else "name"
    return ScriptSmokeIssue(
        severity="error",
        code="unsupported_trackio_init_kwarg",
        message=(
            f"trackio.init no longer accepts {name}; use {replacement} "
            "before submitting the HF Job."
        ),
        line=getattr(node, "lineno", None),
    )


@dataclass(frozen=True)
class _ConfigKwargs:
    values: dict[str, Any]
    lines: dict[str, int | None]
    unverified: bool = False
    line: int | None = None


def _hub_config_kwargs(calls: list[ast.Call]) -> _ConfigKwargs:
    values: dict[str, Any] = {}
    lines: dict[str, int | None] = {}
    for node in calls:
        for keyword in node.keywords:
            if keyword.arg is None:
                try:
                    kwargs = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    return _ConfigKwargs(
                        values=values,
                        lines=lines,
                        unverified=True,
                        line=getattr(
                            keyword.value, "lineno", getattr(node, "lineno", None)
                        ),
                    )
                if not isinstance(kwargs, Mapping):
                    return _ConfigKwargs(
                        values=values,
                        lines=lines,
                        unverified=True,
                        line=getattr(
                            keyword.value, "lineno", getattr(node, "lineno", None)
                        ),
                    )
                for name in ("push_to_hub", "hub_model_id"):
                    if name in kwargs and name not in values:
                        values[name] = kwargs[name]
                        lines[name] = getattr(keyword.value, "lineno", None)
                continue
            if (
                keyword.arg in {"push_to_hub", "hub_model_id"}
                and keyword.arg not in values
            ):
                values[keyword.arg] = _literal_or_dynamic(keyword.value)
                lines[keyword.arg] = getattr(keyword.value, "lineno", None)
    return _ConfigKwargs(values=values, lines=lines)


def _literal_or_dynamic(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<dynamic>"


def _keyword_literal(calls: list[ast.Call], name: str) -> Any:
    for node in calls:
        for keyword in node.keywords:
            if keyword.arg == name:
                return _literal_or_dynamic(keyword.value)
    return None


def _keyword_line(calls: list[ast.Call], name: str) -> int | None:
    for node in calls:
        for keyword in node.keywords:
            if keyword.arg == name:
                return getattr(keyword.value, "lineno", None)
    return None


def _non_empty_literal(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_like_training_text(
    script: str,
    job_args: Mapping[str, Any],
) -> bool:
    text_parts = [script]
    dependencies = job_args.get("dependencies")
    if isinstance(dependencies, list):
        text_parts.extend(str(item) for item in dependencies)
    text = " ".join(text_parts).lower()
    return any(
        marker in text
        for marker in (
            "sfttrainer",
            "dpotrainer",
            "grpotrainer",
            "trainingarguments",
            "sftconfig",
            "dpoconfig",
            "grpoconfig",
            "trainer.train",
            "accelerate launch",
            "unsloth",
        )
    )


def _looks_like_unresolved_training_script(
    script: str,
    job_args: Mapping[str, Any],
) -> bool:
    if _explicit_training_script_metadata(job_args):
        return True

    text_parts = [script]
    dependencies = job_args.get("dependencies")
    if isinstance(dependencies, list):
        text_parts.extend(str(item) for item in dependencies)
    text = " ".join(text_parts).lower()
    return _looks_like_training_text(script, job_args) or bool(
        _TRAINING_PATH_RE.search(text)
    )


def _explicit_training_script_metadata(job_args: Mapping[str, Any]) -> bool:
    for key in ("script_smoke", "preflight"):
        explicit = job_args.get(key)
        if isinstance(explicit, Mapping):
            return coerce_script_smoke_result(explicit).is_training_script
    return False


class _TrainingScriptDetector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.is_training_script = False

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = _call_name(node.func)
        if call_name in _TRAINING_CALL_NAMES:
            self.is_training_script = True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
            self.is_training_script = True
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
