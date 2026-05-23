"""Phase 2 tests for platform service extraction and route hardening."""

import io
import os
import sys
import zipfile
from pathlib import Path

import pytest
import httpx
from fastapi.testclient import TestClient
from huggingface_hub.utils import HfHubHTTPError

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import main  # noqa: E402
from services import dataset_service, model_chat_service  # noqa: E402


class _FakeHfApi:
    instances = []
    fail_create = None
    fail_upload = None
    failed_upload_path = None

    def __init__(self, token=None):
        self.token = token
        self.created_repos = []
        self.uploads = []
        _FakeHfApi.instances.append(self)

    def whoami(self):
        if self.token == "invalid":
            raise RuntimeError("401 Unauthorized")
        return {"name": "tester"}

    def create_repo(self, **kwargs):
        if self.fail_create:
            raise self.fail_create
        self.created_repos.append(kwargs)

    def upload_file(self, **kwargs):
        path = kwargs["path_or_fileobj"]
        if self.fail_upload:
            _FakeHfApi.failed_upload_path = path
            raise self.fail_upload
        assert os.path.exists(path)
        self.uploads.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_hf_api(monkeypatch):
    _FakeHfApi.instances = []
    _FakeHfApi.fail_create = None
    _FakeHfApi.fail_upload = None
    _FakeHfApi.failed_upload_path = None
    monkeypatch.setattr(dataset_service, "HfApi", _FakeHfApi)
    monkeypatch.setattr(model_chat_service, "HfApi", _FakeHfApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_upload_dataset_success_uses_hub_and_preserves_response_fields(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"prompt,response\nhi,there\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_profile"]["row_count"] == 1
    assert payload["dataset_profile"]["columns"] == ["prompt", "response"]
    assert payload["dataset_profile"]["source"]["type"] == "local_file"
    assert payload["dataset_profile"]["trainability"]["risk_level"] == "high"
    assert {key: payload[key] for key in ("dataset_id", "filename", "url")} == {
        "dataset_id": "tester/dataset",
        "filename": "train.csv",
        "url": "https://huggingface.co/datasets/tester/dataset",
    }
    api = _FakeHfApi.instances[-1]
    assert api.created_repos == [
        {
            "repo_id": "tester/dataset",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    upload = api.uploads[0]
    assert upload["path_in_repo"] == "train.csv"
    assert upload["repo_id"] == "tester/dataset"
    assert upload["repo_type"] == "dataset"
    assert not os.path.exists(upload["path_or_fileobj"])


def test_upload_dataset_rewrites_dev_namespace_to_hf_token_owner(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "dev/generic-session-1234"},
        files={"file": ("train.csv", b"prompt,response\nhi,there\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "tester/generic-session-1234"
    assert (
        payload["url"] == "https://huggingface.co/datasets/tester/generic-session-1234"
    )
    api = _FakeHfApi.instances[-1]
    assert api.created_repos[0]["repo_id"] == "tester/generic-session-1234"
    assert api.uploads[0]["repo_id"] == "tester/generic-session-1234"


def test_upload_dataset_accepts_multiple_files_and_profiles_combined_rows(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "dev/multi-upload"},
        files=[
            (
                "files",
                ("train.csv", b"instruction,output\nSay hi,Hi there\n", "text/csv"),
            ),
            (
                "files",
                (
                    "reference.xlsx",
                    _xlsx_bytes([["instruction", "output"], ["Say bye", "Bye there"]]),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "tester/multi-upload"
    assert payload["filename"] == "train.jsonl"
    assert [item["filename"] for item in payload["files"]] == [
        "train.csv",
        "reference.xlsx",
    ]
    assert payload["dataset_profile"]["row_count"] == 2
    assert payload["dataset_profile"]["source"]["type"] == "uploaded_files"
    assert payload["dataset_profile"]["strategy"]["strategy"] == "fine_tune"
    uploads = _FakeHfApi.instances[-1].uploads
    assert [upload["path_in_repo"] for upload in uploads] == [
        "raw/train.csv",
        "raw/reference.xlsx",
        "train.jsonl",
    ]
    assert all(upload["repo_id"] == "tester/multi-upload" for upload in uploads)


def test_upload_dataset_converts_pdf_and_docx_to_reference_rows(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/docs"},
        files=[
            (
                "files",
                (
                    "guide.pdf",
                    _pdf_bytes("Refund rules require Form 16."),
                    "application/pdf",
                ),
            ),
            (
                "files",
                (
                    "notes.docx",
                    _docx_bytes(
                        paragraphs=["Use ITR-1 for salary income."],
                        table_rows=[["Form", "Purpose"], ["ITR-1", "Salary filing"]],
                    ),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            ),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    profile = payload["dataset_profile"]
    assert profile["source"]["formats"] == ["docx", "pdf"]
    assert profile["inferred_shape"] == "document_corpus"
    assert profile["strategy"]["strategy"] == "rag"
    assert profile["reference_lookup"]["ready"] is True
    sample_text = " ".join(str(row.get("text", "")) for row in profile["sample_rows"])
    assert "Refund rules" in sample_text
    assert "ITR-1" in sample_text
    assert _FakeHfApi.instances[-1].uploads[-1]["path_in_repo"] == "train.jsonl"


def test_upload_dataset_rejects_any_invalid_file_in_multi_upload(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files=[
            ("files", ("train.csv", b"instruction,output\nhi,there\n", "text/csv")),
            ("files", ("notes.txt", b"not supported", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert "Only .pdf, .docx, .csv, .xlsx" in response.json()["detail"]
    assert _FakeHfApi.instances[-1].created_repos == []
    assert _FakeHfApi.instances[-1].uploads == []


def test_upload_dataset_skips_profile_above_profile_threshold(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")
    monkeypatch.setattr(dataset_service, "MAX_PROFILE_UPLOAD_BYTES", 3)

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "dataset_profile" not in payload
    assert "Dataset profiling skipped" in payload["dataset_profile_error"]
    assert "profiling limit" in payload["dataset_profile_error"]
    assert _FakeHfApi.instances[-1].uploads[0]["path_in_repo"] == "train.csv"


def test_upload_dataset_rejects_empty_file(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded dataset file is empty."
    assert _FakeHfApi.instances[-1].created_repos == []
    assert _FakeHfApi.instances[-1].uploads == []


def test_upload_dataset_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("notes.txt", b"not,a,dataset\n", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .pdf, .docx, .csv, .xlsx" in response.json()["detail"]
    assert _FakeHfApi.instances[-1].created_repos == []
    assert _FakeHfApi.instances[-1].uploads == []


def test_upload_dataset_rejects_oversize_file(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")
    monkeypatch.setattr(dataset_service, "MAX_UPLOAD_BYTES", 3)

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("too-big.csv", b"1234", "text/csv")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Dataset upload exceeds the 100 MB limit."
    assert _FakeHfApi.instances[-1].created_repos == []
    assert _FakeHfApi.instances[-1].uploads == []


def test_upload_dataset_resolves_auth_before_file_validation(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("unsupported.txt", b"not,a,dataset\n", "text/plain")},
        headers={"Authorization": "Bearer invalid"},
    )

    assert response.status_code == 401
    assert "No valid HF auth token found" in response.json()["detail"]


class _SizedUploadFile:
    filename = "too-big.csv"

    def __init__(self, size):
        self.file = _SizedFile(size)

    async def read(self):
        raise AssertionError("oversize validation should not read the upload body")


class _SizedFile:
    def __init__(self, size):
        self.size = size
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == os.SEEK_SET:
            self.position = offset
        elif whence == os.SEEK_CUR:
            self.position += offset
        elif whence == os.SEEK_END:
            self.position = self.size + offset
        return self.position


@pytest.mark.asyncio
async def test_validate_dataset_file_rejects_oversize_without_reading(monkeypatch):
    monkeypatch.setattr(dataset_service, "MAX_UPLOAD_BYTES", 3)

    with pytest.raises(Exception) as exc_info:
        await dataset_service.validate_dataset_file(_SizedUploadFile(size=4))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "Dataset upload exceeds the 100 MB limit."


def test_upload_dataset_sanitizes_filename_for_hub_path(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("../bad name.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    upload = _FakeHfApi.instances[-1].uploads[0]
    assert upload["path_in_repo"] == "bad-name.csv"
    assert response.json()["filename"] == "bad-name.csv"


def test_upload_dataset_missing_or_invalid_token_returns_401(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    missing = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
    )
    invalid = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
        headers={"Authorization": "Bearer invalid"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_upload_dataset_maps_hub_permission_failure(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")
    _FakeHfApi.fail_create = RuntimeError("403 Forbidden: permission denied")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 403
    assert "Hub permission denied" in response.json()["detail"]


def _hf_http_error(status_code: int) -> HfHubHTTPError:
    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://huggingface.co/api/repos/create"),
    )
    return HfHubHTTPError(f"{status_code} hub error", response=response)


@pytest.mark.parametrize(
    ("status_code", "expected_detail"),
    [(401, "Hub authentication failed"), (403, "Hub permission denied")],
)
def test_upload_dataset_maps_hf_hub_http_auth_errors(
    monkeypatch, status_code, expected_detail
):
    monkeypatch.setenv("HF_TOKEN", "valid")
    _FakeHfApi.fail_create = _hf_http_error(status_code)

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == status_code
    assert expected_detail in response.json()["detail"]


def test_upload_dataset_removes_temp_file_when_upload_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "valid")
    _FakeHfApi.fail_upload = RuntimeError("network failed")

    response = TestClient(main.app).post(
        "/api/platform/upload-dataset",
        data={"repo_id": "tester/dataset"},
        files={"file": ("train.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 500
    assert "Upload failed" in response.json()["detail"]
    assert _FakeHfApi.failed_upload_path
    assert not os.path.exists(_FakeHfApi.failed_upload_path)


def _xlsx_bytes(rows):
    shared_strings = []
    shared_index = {}

    def shared(value):
        text = str(value)
        if text not in shared_index:
            shared_index[text] = len(shared_strings)
            shared_strings.append(text)
        return shared_index[text]

    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            cell_ref = f"{chr(ord('A') + column_index)}{row_index}"
            cells.append(f'<c r="{cell_ref}" t="s"><v>{shared(value)}</v></c>')
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""",
        )
        workbook.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
            + "</sst>",
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(sheet_rows)
            + "</sheetData></worksheet>",
        )
    return buffer.getvalue()


def _docx_bytes(*, paragraphs, table_rows):
    paragraph_xml = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    table_xml = (
        "<w:tbl>"
        + "".join(
            "<w:tr>"
            + "".join(
                f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row
            )
            + "</w:tr>"
            for row in table_rows
        )
        + "</w:tbl>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as document:
        document.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + paragraph_xml
            + table_xml
            + "</w:body></w:document>",
        )
    return buffer.getvalue()


def _pdf_bytes(text):
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<<>>endobj\n"
        b"2 0 obj<< /Length 64 >>stream\nBT /F1 12 Tf 72 720 Td ("
        + text.encode("ascii")
        + b") Tj ET\nendstream\nendobj\n"
        b"trailer<< /Root 1 0 R >>\n%%EOF"
    )


class _FakeStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_text(self):
        yield "data: hello\n\n"

    async def aread(self):
        return b""


class _FakeAsyncClient:
    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, headers, json):
        assert method == "POST"
        assert url.endswith("/tester/model/v1/chat/completions")
        assert headers == {"Authorization": "Bearer valid"}
        assert json["model"] == "tester/model"
        assert json["stream"] is True
        return _FakeStreamResponse()


@pytest.mark.asyncio
async def test_model_chat_service_streams_happy_path_chunks():
    chunks = [
        chunk
        async for chunk in model_chat_service.stream_model_chat(
            token="valid",
            model_id="tester/model",
            messages=[{"role": "user", "content": "hi"}],
            async_client_factory=_FakeAsyncClient,
        )
    ]

    assert chunks == ["data: hello\n\n"]


def test_model_chat_route_auth_failure_returns_401(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    response = TestClient(main.app).post(
        "/api/platform/model-chat",
        json={
            "model_id": "tester/model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 401
