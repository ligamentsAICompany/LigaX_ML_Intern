"""Normalized dataset profiling for local files and Hub inspection payloads."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from agent.core.golden_eval import QUALITY_CONSTRAINTS, generate_golden_eval_cases
from agent.core.redact import scrub
from agent.core.reference_lookup import build_reference_lookup_summary
from agent.core.strategy_selector import select_ml_strategy
from agent.core.trainability import assess_trainability

MAX_SAMPLE_ROWS = 3
MAX_SAMPLE_VALUE_LEN = 150
SUPPORTED_LOCAL_EXTENSIONS = {
    ".csv",
    ".docx",
    ".json",
    ".jsonl",
    ".parquet",
    ".pdf",
    ".xlsx",
}
DOCUMENT_CHUNK_CHAR_LIMIT = 1200

_XLSX_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_RELS_NS = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
)
_DOCX_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_PDF_LITERAL_TEXT_RE = re.compile(rb"\(([^()]*)\)\s*Tj")


def inspect_local_dataset(
    path: str | Path,
    *,
    sheet: str | None = None,
    sample_rows: int = MAX_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Read a supported local dataset file and return a normalized profile."""

    dataset_path = Path(path)
    extension = dataset_path.suffix.lower()
    if extension not in SUPPORTED_LOCAL_EXTENSIONS:
        raise ValueError(
            "Unsupported dataset extension "
            f"{extension!r}. Supported: {', '.join(sorted(SUPPORTED_LOCAL_EXTENSIONS))}."
        )

    rows = _load_local_rows(dataset_path, extension=extension, sheet=sheet)
    source: dict[str, Any] = {
        "type": "local_file",
        "path": str(dataset_path),
        "format": extension.lstrip("."),
    }
    if extension == ".xlsx":
        source["sheet"] = sheet or _first_xlsx_sheet_name(dataset_path)

    return build_dataset_profile(
        rows=rows,
        source=source,
        sample_rows=sample_rows,
    )


def extract_local_dataset_rows(
    path: str | Path,
    *,
    source_filename: str | None = None,
    sheet: str | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Extract rows and source metadata for upload-time dataset conversion."""

    dataset_path = Path(path)
    extension = dataset_path.suffix.lower()
    if extension not in SUPPORTED_LOCAL_EXTENSIONS:
        raise ValueError(
            "Unsupported dataset extension "
            f"{extension!r}. Supported: {', '.join(sorted(SUPPORTED_LOCAL_EXTENSIONS))}."
        )

    source_name = source_filename or dataset_path.name
    source_format = extension.lstrip(".")
    rows = [
        _with_source_metadata(
            row,
            source_file=source_name,
            source_format=source_format,
        )
        for row in _load_local_rows(dataset_path, extension=extension, sheet=sheet)
    ]
    source: dict[str, Any] = {
        "type": "local_file",
        "path": source_name,
        "format": source_format,
    }
    if extension == ".xlsx":
        source["sheet"] = sheet or _first_xlsx_sheet_name(dataset_path)
    return rows, source


def build_dataset_profile(
    *,
    rows: list[Mapping[str, Any]],
    source: Mapping[str, Any],
    sample_rows: int = MAX_SAMPLE_ROWS,
    columns: list[str] | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Build a normalized dataset profile from already-loaded row mappings."""

    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    columns = columns or _collect_columns(normalized_rows)
    row_count = len(normalized_rows) if row_count is None else max(0, row_count)
    profiled_row_count = len(normalized_rows)
    statistics_basis = (
        "full_dataset" if profiled_row_count == row_count else "sample_rows"
    )
    missing_summary = _missing_summary(normalized_rows, columns)
    missing_cells = sum(missing_summary.values())
    profiled_cells = profiled_row_count * len(columns)
    duplicate_count = _duplicate_count(normalized_rows, columns)
    sample = [
        _truncate_sample_value(scrub(row))
        for row in normalized_rows[: max(0, min(sample_rows, 10))]
    ]
    profile: dict[str, Any] = {
        "source": dict(source),
        "format": source.get("format"),
        "row_count": row_count,
        "profiled_row_count": profiled_row_count,
        "statistics_basis": statistics_basis,
        "columns": columns,
        "inferred_shape": infer_dataset_shape(
            columns=columns,
            sample_rows=normalized_rows[: max(0, min(sample_rows, 10))],
            source_format=str(source.get("format") or ""),
        ),
        "missing_summary": missing_summary,
        "missing_fraction": round(missing_cells / profiled_cells, 6)
        if profiled_cells
        else 0.0,
        "duplicate_count": duplicate_count,
        "duplicate_fraction": round(duplicate_count / profiled_row_count, 6)
        if profiled_row_count
        else 0.0,
        "sample_rows": sample,
    }

    trainability_input = {
        "row_count": profile["row_count"],
        "columns": profile["columns"],
        "format": source.get("format"),
        "sample_rows": sample,
        "missing_fraction": profile["missing_fraction"],
        "duplicate_fraction": profile["duplicate_fraction"],
        "statistics_basis": profile["statistics_basis"],
    }
    trainability = assess_trainability(trainability_input).to_dict()
    profile["trainability"] = trainability
    profile["strategy"] = select_ml_strategy(profile, trainability).to_dict()
    golden_eval_cases = generate_golden_eval_cases(
        rows=normalized_rows,
        columns=columns,
        dataset_shape=str(profile["inferred_shape"]),
        source=source,
    )
    profile["golden_eval"] = {
        "case_count": len(golden_eval_cases),
        "quality_constraints": list(QUALITY_CONSTRAINTS),
        "cases": [case.to_dict() for case in golden_eval_cases],
    }
    profile["reference_lookup"] = build_reference_lookup_summary(
        rows=normalized_rows,
        columns=columns,
        source=source,
    )
    if profile["inferred_shape"] == "document_corpus":
        profile["reference_lookup"] = {
            "ready": bool(normalized_rows),
            "row_count": row_count,
            "key_columns": ["source_file", "chunk_index"],
            "answer_columns": ["text"],
            "status": "ready" if normalized_rows else "empty_document_corpus",
        }
    return profile


def infer_dataset_shape(
    *,
    columns: list[str],
    sample_rows: list[Mapping[str, Any]],
    source_format: str | None = None,
) -> str:
    """Infer the likely ML/data shape from schema and a few rows."""

    normalized_columns = {column.lower(): column for column in columns}
    column_set = set(normalized_columns)
    if "messages" in column_set and _messages_rows_have_user_and_assistant(sample_rows):
        return "sft_messages"
    if {"prompt", "chosen", "rejected"}.issubset(
        column_set
    ) and _rows_have_non_empty_fields(sample_rows, {"prompt", "chosen", "rejected"}):
        return "dpo"
    if {"prompt", "completion"}.issubset(column_set) and _rows_have_non_empty_fields(
        sample_rows, {"prompt", "completion"}
    ):
        return "prompt_completion"
    if {"instruction", "output"}.issubset(column_set) or {
        "instruction",
        "response",
    }.issubset(column_set):
        required = (
            {"instruction", "output"}
            if "output" in column_set
            else {
                "instruction",
                "response",
            }
        )
        if _rows_have_non_empty_fields(sample_rows, required):
            return "prompt_completion"
    if column_set == {"prompt"} and _rows_have_non_empty_fields(
        sample_rows, {"prompt"}
    ):
        return "grpo_prompt_only"
    if {"question", "answer"}.issubset(column_set) or {
        "question",
        "context",
        "answer",
    }.issubset(column_set):
        return "qa"
    if _looks_like_document_corpus(columns=column_set, sample_rows=sample_rows):
        return "document_corpus"
    if "label" in column_set and (
        "text" in column_set or "sentence" in column_set or "content" in column_set
    ):
        return "classification"
    if _looks_like_structured_reference(
        columns=column_set,
        sample_rows=sample_rows,
        source_format=source_format,
    ):
        return "structured_reference_table"
    return "unknown"


def _load_local_rows(
    path: Path, *, extension: str, sheet: str | None
) -> list[Mapping[str, Any]]:
    if extension == ".csv":
        return _load_csv_rows(path)
    if extension == ".json":
        return _load_json_rows(path)
    if extension == ".jsonl":
        return _load_jsonl_rows(path)
    if extension == ".parquet":
        return _load_parquet_rows(path)
    if extension == ".xlsx":
        return _load_xlsx_rows(path, sheet=sheet)
    if extension == ".pdf":
        return _load_pdf_rows(path)
    if extension == ".docx":
        return _load_docx_rows(path)
    raise ValueError(f"Unsupported dataset extension {extension!r}.")


def _load_csv_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _load_json_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "data", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
        return [payload]
    raise ValueError("JSON dataset must be an object or a list of objects.")


def _load_jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object.")
            rows.append(payload)
    return rows


def _load_parquet_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet inspection requires optional dependency pandas with a parquet "
            "engine such as pyarrow or fastparquet."
        ) from exc
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet inspection requires a parquet engine such as pyarrow or fastparquet."
        ) from exc
    return frame.to_dict(orient="records")


def _load_xlsx_rows(path: Path, *, sheet: str | None) -> list[Mapping[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        return _load_xlsx_rows_stdlib(path, sheet=sheet)
    try:
        frame = pd.read_excel(path, sheet_name=sheet or 0)
    except ImportError:
        return _load_xlsx_rows_stdlib(path, sheet=sheet)
    return frame.where(frame.notna(), "").to_dict(orient="records")


def _load_xlsx_rows_stdlib(path: Path, *, sheet: str | None) -> list[Mapping[str, Any]]:
    with zipfile.ZipFile(path) as workbook:
        sheet_path = _xlsx_sheet_path(workbook, sheet)
        shared_strings = _xlsx_shared_strings(workbook)
        sheet_xml = ElementTree.fromstring(workbook.read(sheet_path))

    table: list[list[Any]] = []
    for row in sheet_xml.findall(f".//{_XLSX_MAIN_NS}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{_XLSX_MAIN_NS}c"):
            column_index = _xlsx_column_index(cell.attrib.get("r", ""))
            values[column_index] = _xlsx_cell_value(cell, shared_strings)
        if values:
            width = max(values) + 1
            table.append([values.get(index, "") for index in range(width)])

    if not table:
        return []
    headers = [str(value).strip() for value in table[0]]
    return [
        {
            headers[index]: row[index] if index < len(row) else ""
            for index in range(len(headers))
        }
        for row in table[1:]
        if any(not _is_missing(value) for value in row)
    ]


def _load_pdf_rows(path: Path) -> list[Mapping[str, Any]]:
    text_by_page: list[tuple[int | None, str]] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                text_by_page.append((page_number, text))
    except Exception:
        text = _extract_pdf_literal_text(path.read_bytes())
        if text:
            text_by_page.append((None, text))

    rows: list[Mapping[str, Any]] = []
    chunk_index = 0
    for page_number, text in text_by_page:
        for chunk in _chunk_text(text):
            row: dict[str, Any] = {
                "text": chunk,
                "chunk_index": chunk_index,
            }
            if page_number is not None:
                row["page_number"] = page_number
            rows.append(row)
            chunk_index += 1
    return rows


def _load_docx_rows(path: Path) -> list[Mapping[str, Any]]:
    with zipfile.ZipFile(path) as document:
        document_xml = ElementTree.fromstring(document.read("word/document.xml"))

    chunks: list[str] = []
    for paragraph in document_xml.findall(f".//{_DOCX_WORD_NS}p"):
        text = _docx_text(paragraph)
        if text:
            chunks.append(text)
    for table in document_xml.findall(f".//{_DOCX_WORD_NS}tbl"):
        for row in table.findall(f"{_DOCX_WORD_NS}tr"):
            cells = [_docx_text(cell) for cell in row.findall(f"{_DOCX_WORD_NS}tc")]
            text = " | ".join(cell for cell in cells if cell)
            if text:
                chunks.append(text)

    rows: list[Mapping[str, Any]] = []
    chunk_index = 0
    for chunk in _chunk_text("\n".join(chunks)):
        rows.append({"text": chunk, "chunk_index": chunk_index})
        chunk_index += 1
    return rows


def _docx_text(node: ElementTree.Element) -> str:
    return " ".join(
        text_node.text or ""
        for text_node in node.findall(f".//{_DOCX_WORD_NS}t")
        if text_node.text
    ).strip()


def _extract_pdf_literal_text(contents: bytes) -> str:
    values: list[str] = []
    for raw_value in _PDF_LITERAL_TEXT_RE.findall(contents):
        try:
            values.append(raw_value.decode("utf-8"))
        except UnicodeDecodeError:
            values.append(raw_value.decode("latin-1", errors="ignore"))
    return "\n".join(value.strip() for value in values if value.strip())


def _chunk_text(text: str, *, limit: int = DOCUMENT_CHUNK_CHAR_LIMIT) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in normalized.split("\n"):
        if not current:
            current = paragraph
        elif len(current) + 1 + len(paragraph) <= limit:
            current = f"{current}\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
        while len(current) > limit:
            chunks.append(current[:limit])
            current = current[limit:].lstrip()
    if current:
        chunks.append(current)
    return chunks


def _with_source_metadata(
    row: Mapping[str, Any], *, source_file: str, source_format: str
) -> Mapping[str, Any]:
    enriched = dict(row)
    enriched.setdefault("source_file", source_file)
    enriched.setdefault("source_format", source_format)
    return enriched


def _first_xlsx_sheet_name(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as workbook:
            workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
            sheet = workbook_xml.find(f".//{_XLSX_MAIN_NS}sheet")
            return sheet.attrib.get("name") if sheet is not None else None
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError):
        return None


def _xlsx_sheet_path(workbook: zipfile.ZipFile, sheet: str | None) -> str:
    workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    rels_xml = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_xml.findall(f"{_RELS_NS}Relationship")
    }
    selected_rel_id: str | None = None
    available_sheets: list[str] = []
    for sheet_node in workbook_xml.findall(f".//{_XLSX_MAIN_NS}sheet"):
        sheet_name = sheet_node.attrib.get("name", "")
        available_sheets.append(sheet_name)
        if sheet is None or sheet_name == sheet:
            selected_rel_id = sheet_node.attrib.get(f"{_OFFICE_RELS_NS}id")
            break
    if selected_rel_id is None:
        raise ValueError(
            f"Sheet {sheet!r} not found. Available sheets: {', '.join(available_sheets)}."
        )
    target = rel_targets[selected_rel_id]
    return f"xl/{target}" if not target.startswith("xl/") else target


def _xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        shared_xml = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in shared_xml.findall(f"{_XLSX_MAIN_NS}si"):
        parts = [node.text or "" for node in item.findall(f".//{_XLSX_MAIN_NS}t")]
        strings.append("".join(parts))
    return strings


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(f".//{_XLSX_MAIN_NS}t"))
    value_node = cell.find(f"{_XLSX_MAIN_NS}v")
    value = value_node.text if value_node is not None else ""
    if cell_type == "s" and value:
        return shared_strings[int(value)]
    return value


def _xlsx_column_index(cell_reference: str) -> int:
    letters = "".join(char for char in cell_reference if char.isalpha()).upper()
    if not letters:
        return 0
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _collect_columns(rows: list[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            column_name = str(column)
            if column_name not in seen:
                columns.append(column_name)
                seen.add(column_name)
    return columns


def _missing_summary(
    rows: list[Mapping[str, Any]], columns: list[str]
) -> dict[str, int]:
    return {
        column: sum(1 for row in rows if _is_missing(row.get(column)))
        for column in columns
    }


def _duplicate_count(rows: list[Mapping[str, Any]], columns: list[str]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        fingerprint = json.dumps(
            [_json_safe(row.get(column)) for column in columns],
            sort_keys=True,
            default=str,
        )
        if fingerprint in seen:
            duplicates += 1
        else:
            seen.add(fingerprint)
    return duplicates


def _truncate_sample_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_SAMPLE_VALUE_LEN:
            return value[:MAX_SAMPLE_VALUE_LEN] + "..."
        return value
    if isinstance(value, Mapping):
        return {key: _truncate_sample_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_sample_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_truncate_sample_value(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _messages_rows_have_user_and_assistant(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                continue
        if not isinstance(messages, list):
            continue
        roles = {
            message.get("role")
            for message in messages
            if isinstance(message, Mapping) and not _is_missing(message.get("content"))
        }
        if {"user", "assistant"}.issubset(roles):
            return True
    return False


def _rows_have_non_empty_fields(
    rows: list[Mapping[str, Any]], required: set[str]
) -> bool:
    for row in rows:
        normalized_row = {str(key).lower(): value for key, value in row.items()}
        if all(not _is_missing(normalized_row.get(column)) for column in required):
            return True
    return False


def _looks_like_structured_reference(
    *,
    columns: set[str],
    sample_rows: list[Mapping[str, Any]],
    source_format: str | None,
) -> bool:
    reference_hints = {
        "form no",
        "purpose",
        "section",
        "applicable sections",
        "description",
        "limit",
        "applicability",
        "rule",
        "rate",
        "category",
        "deduction",
        "due dates",
        "filing frequency",
        "user categories",
        "old/new regime applicability",
    }
    if len(columns) < 3 or len(columns & reference_hints) < 2:
        return False
    scalar_rows = bool(sample_rows) and all(
        not isinstance(value, (list, dict))
        for row in sample_rows
        for value in row.values()
    )
    return scalar_rows or (source_format or "").lower() in {"csv", "xlsx", "xls"}


def _looks_like_document_corpus(
    *, columns: set[str], sample_rows: list[Mapping[str, Any]]
) -> bool:
    if "text" not in columns:
        return False
    if "source_format" in columns:
        formats = {
            str(row.get("source_format") or "").lower()
            for row in sample_rows
            if isinstance(row, Mapping)
        }
        if formats & {"pdf", "docx"}:
            return True
    return {"source_file", "chunk_index"}.issubset(columns)
