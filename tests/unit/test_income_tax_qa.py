from argparse import Namespace
import importlib.util
from pathlib import Path

from agent.tools.income_tax_qa import build_income_tax_qa_examples


def _load_income_tax_launcher():
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "launch_income_tax_sft_job.py"
    )
    spec = importlib.util.spec_from_file_location(
        "launch_income_tax_sft_job", script_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_income_tax_qa_examples_are_chat_sft_records():
    examples = build_income_tax_qa_examples(
        [
            {
                "Form No": "ITR-1 (Sahaj)",
                "Category": "ITR",
                "Purpose": "Simple income return",
                "Applicable Sections": "Sec 139(1)",
                "Filing Frequency": "Annual",
                "Due Dates": "31 July",
                "User Categories": "Salaried Individuals",
                "Old/New Regime Applicability": "Both",
                "API / E-Filing Mapping Possibilities": "ITR Filing API",
            }
        ]
    )

    assert len(examples) == 6
    assert examples[0]["messages"][0]["role"] == "system"
    assert examples[0]["messages"][1]["role"] == "user"
    assert examples[0]["messages"][2]["role"] == "assistant"
    assert examples[0]["form_no"] == "ITR-1 (Sahaj)"
    assert examples[0]["qa_type"] == "purpose"
    assert "Simple income return" in examples[0]["messages"][2]["content"]


def test_income_tax_qa_marks_missing_values_without_hallucinating():
    examples = build_income_tax_qa_examples(
        [
            {
                "Form No": "Form 10E",
                "Category": "Relief",
                "Purpose": "Relief claim",
                "Applicable Sections": "Sec 89",
                "Filing Frequency": "As needed",
                "Due Dates": None,
                "User Categories": "Individuals",
                "Old/New Regime Applicability": "",
                "API / E-Filing Mapping Possibilities": "E-Filing portal",
            }
        ]
    )

    timeline = next(item for item in examples if item["qa_type"] == "filing_timeline")
    regime = next(item for item in examples if item["qa_type"] == "regime")

    assert "Not specified in the source table" in timeline["messages"][2]["content"]
    assert "Not specified in the source table" in regime["messages"][2]["content"]


def test_income_tax_launcher_uses_current_sft_and_trackio_patterns():
    script = _load_income_tax_launcher().build_training_script(
        Namespace(
            dataset_repo="owner/dataset",
            model_repo="owner/model",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            max_steps=2,
        )
    )

    assert "trackio.init(project=" in script
    assert "name=RUN_NAME" in script
    assert "project_name=" not in script
    assert "trackio.init(run_name=" not in script
    assert "max_length=512" in script
    assert "max_seq_length" not in script
    assert "processing_class=tokenizer" in script
    assert "tokenizer=" not in script
