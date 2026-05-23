"""Phase 2 Dataset Inspection Upgrade tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from agent.core.dataset_inspection import inspect_local_dataset
from agent.tools.dataset_tools import _build_hub_dataset_profile, _format_samples


def _write_minimal_xlsx(path: Path, rows: list[list[object]]) -> None:
    """Create a small XLSX workbook using only the standard library."""

    def cell_ref(row_index: int, column_index: int) -> str:
        column = chr(ord("A") + column_index)
        return f"{column}{row_index}"

    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            escaped = (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            cells.append(
                f'<c r="{cell_ref(row_index, column_index)}" t="inlineStr">'
                f"<is><t>{escaped}</t></is></c>"
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/worksheets/sheet1.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="xl/workbook.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "xl/workbook.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Tax Rules" sheetId="1" r:id="rId1"/></sheets>'
                "</workbook>"
            ),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                'Target="worksheets/sheet1.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{''.join(sheet_rows)}</sheetData>"
                "</worksheet>"
            ),
        )


def test_xlsx_reference_table_profiles_trainability_and_missing_values(tmp_path):
    path = tmp_path / "Income_Tax_Master.xlsx"
    _write_minimal_xlsx(
        path,
        [
            ["Section", "Description", "Limit", "Applicability"],
            ["80C", "Deduction for eligible investments", "150000", "Individuals"],
            ["80D", "Medical insurance deduction", "", "Individuals"],
        ],
    )

    profile = inspect_local_dataset(path, sheet="Tax Rules")

    assert profile["source"] == {
        "type": "local_file",
        "path": str(path),
        "format": "xlsx",
        "sheet": "Tax Rules",
    }
    assert profile["row_count"] == 2
    assert profile["columns"] == ["Section", "Description", "Limit", "Applicability"]
    assert profile["inferred_shape"] == "structured_reference_table"
    assert profile["missing_summary"]["Limit"] == 1
    assert profile["trainability"]["risk_level"] == "high"
    assert profile["trainability"]["recommendation"] in {"rag", "hybrid"}


def test_csv_reference_table_profile_counts_duplicates(tmp_path):
    path = tmp_path / "tax.csv"
    path.write_text(
        "Section,Description,Limit,Applicability\n"
        "80C,Deduction for eligible investments,150000,Individuals\n"
        "80C,Deduction for eligible investments,150000,Individuals\n",
        encoding="utf-8",
    )

    profile = inspect_local_dataset(path)

    assert profile["row_count"] == 2
    assert profile["duplicate_count"] == 1
    assert profile["duplicate_fraction"] == 0.5
    assert profile["inferred_shape"] == "structured_reference_table"
    assert profile["trainability"]["risk_level"] == "high"


def test_jsonl_messages_sft_dataset_profiles_as_fine_tunable(tmp_path):
    path = tmp_path / "sft.jsonl"
    base_rows = [
        {
            "messages": [
                {"role": "system", "content": "Answer as a tax assistant."},
                {"role": "user", "content": "What is section 80C?"},
                {"role": "assistant", "content": "Section 80C covers deductions."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is section 80D?"},
                {"role": "assistant", "content": "It covers medical insurance."},
            ]
        },
    ]
    rows = base_rows * 600
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    profile = inspect_local_dataset(path)

    assert profile["row_count"] == 1200
    assert profile["columns"] == ["messages"]
    assert profile["inferred_shape"] == "sft_messages"
    assert profile["trainability"]["recommendation"] == "fine_tune"
    assert profile["trainability"]["risk_level"] == "low"


def test_json_prompt_completion_dataset_profiles_shape(tmp_path):
    path = tmp_path / "prompt_completion.json"
    path.write_text(
        json.dumps(
            [
                {
                    "prompt": "Explain section 80C.",
                    "completion": "Section 80C covers deductions.",
                },
                {
                    "prompt": "Explain section 80D.",
                    "completion": "Section 80D covers medical insurance.",
                },
            ]
            * 600
        ),
        encoding="utf-8",
    )

    profile = inspect_local_dataset(path)

    assert profile["row_count"] == 1200
    assert profile["columns"] == ["prompt", "completion"]
    assert profile["inferred_shape"] == "prompt_completion"
    assert profile["trainability"]["recommendation"] == "fine_tune"
    assert profile["trainability"]["risk_level"] == "low"


def test_empty_dirty_dataset_records_missingness_and_data_needed(tmp_path):
    path = tmp_path / "dirty.csv"
    path.write_text("prompt,completion\n,\n,\n", encoding="utf-8")

    profile = inspect_local_dataset(path)

    assert profile["row_count"] == 2
    assert profile["missing_summary"] == {"prompt": 2, "completion": 2}
    assert profile["missing_fraction"] == 1.0
    assert profile["duplicate_count"] == 1
    assert profile["inferred_shape"] == "unknown"
    assert profile["trainability"]["recommendation"] == "data_needed"


def test_unsupported_extension_raises_clear_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not,a,dataset\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset extension"):
        inspect_local_dataset(path)


def test_sample_rows_are_redacted_and_truncated(tmp_path):
    path = tmp_path / "secrets.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Bearer hf_abcdefghijklmnopqrstuvwxyz1234567890",
                "completion": "x" * 300,
            }
        ),
        encoding="utf-8",
    )

    profile = inspect_local_dataset(path)
    sample_text = json.dumps(profile["sample_rows"])

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in sample_text
    assert "[REDACTED_" in sample_text
    assert "x" * 200 not in sample_text


def test_hub_dataset_profile_normalizes_existing_api_payloads():
    profile = _build_hub_dataset_profile(
        dataset="org/tax-sft",
        config="default",
        split="train",
        splits_data={
            "splits": [
                {
                    "config": "default",
                    "split": "train",
                    "num_examples": 1200,
                }
            ]
        },
        info_data={
            "dataset_info": {
                "features": {
                    "messages": {"dtype": "list"},
                }
            }
        },
        rows_data={
            "rows": [
                {
                    "row": {
                        "messages": [
                            {"role": "user", "content": "What is 80C?"},
                            {"role": "assistant", "content": "A deduction section."},
                        ]
                    }
                }
            ]
        },
        file_format="parquet",
    )

    assert profile["source"] == {
        "type": "hf_hub",
        "repo": "org/tax-sft",
        "config": "default",
        "split": "train",
        "format": "parquet",
    }
    assert profile["row_count"] == 1200
    assert profile["columns"] == ["messages"]
    assert profile["inferred_shape"] == "sft_messages"
    assert profile["trainability"]["recommendation"] == "fine_tune"


def test_hub_sample_rows_are_redacted_before_formatting():
    formatted = _format_samples(
        {
            "rows": [
                {
                    "row": {
                        "prompt": "Use Bearer hf_abcdefghijklmnopqrstuvwxyz1234567890",
                        "completion": "ok",
                    }
                }
            ]
        },
        config="default",
        split="train",
        limit=1,
    )

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in formatted
    assert "Bearer [REDACTED]" in formatted or "[REDACTED_HF_TOKEN]" in formatted


def test_hub_messages_example_is_redacted_before_formatting():
    formatted = _format_samples(
        {
            "rows": [
                {
                    "row": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "token=hf_abcdefghijklmnopqrstuvwxyz1234567890",
                                "authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456",
                            },
                            {"role": "assistant", "content": "ok"},
                        ]
                    }
                }
            ]
        },
        config="default",
        split="train",
        limit=1,
    )

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in formatted
    assert "Bearer abcdefghijklmnopqrstuvwxyz123456" not in formatted
    assert "[REDACTED_" in formatted


def test_hub_sample_metrics_use_sample_basis_when_split_row_count_is_full_dataset():
    profile = _build_hub_dataset_profile(
        dataset="org/dirty-sft",
        config="default",
        split="train",
        splits_data={
            "splits": [
                {
                    "config": "default",
                    "split": "train",
                    "num_examples": 1200,
                }
            ]
        },
        info_data={
            "dataset_info": {
                "features": {
                    "prompt": {"dtype": "string"},
                    "completion": {"dtype": "string"},
                }
            }
        },
        rows_data={
            "rows": [
                {"row": {"prompt": "", "completion": "ok"}},
                {"row": {"prompt": "", "completion": "ok"}},
            ]
        },
        file_format="parquet",
    )

    assert profile["row_count"] == 1200
    assert profile["profiled_row_count"] == 2
    assert profile["statistics_basis"] == "sample_rows"
    assert profile["missing_summary"] == {"prompt": 2, "completion": 0}
    assert profile["missing_fraction"] == 0.5
    assert profile["duplicate_count"] == 1
    assert profile["duplicate_fraction"] == 0.5
    assert profile["trainability"]["recommendation"] == "data_needed"
