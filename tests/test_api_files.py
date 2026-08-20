"""Phase 2 gate at the HTTP boundary: the flow the frontend actually drives."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ml import indexing as _indexing
from app.ml import pipeline as _ml_pipeline
from tests import fixtures

# Captured before any test patches them, so a test that wants the real
# behaviour can restore it precisely instead of undoing every patch at once.
REAL_PROCESS_FILE = _ml_pipeline.process_file
REAL_INDEX_FILE = _indexing.index_file


@pytest.fixture
def client(
    _migrated_db: None, db_session: object, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client whose background queue runs jobs inline, with ML stubbed out.

    Two deliberate substitutions:

    * Jobs run synchronously. In production they go to a worker thread; here an
      assertion must never race the worker. The endpoint contract is what is
      under test, not the threading.
    * `ml.pipeline.process_file` is replaced. These tests cover the HTTP layer,
      and running the real pipeline would spend ~390s per image captioning on
      CPU. Real inference is covered by `test_ml_inference.py`.
    """
    from app.db.models import File, ProcessingStatus
    from app.main import create_app
    from app.ml import pipeline as ml_pipeline
    from app.tasks.queue import Job, task_queue

    def run_now(
        name: str, handler, file_id: int | None = None, **kwargs
    ) -> Job:  # noqa: ANN001, ANN003
        # **kwargs so this stub keeps working as `submit` gains parameters
        # (priority, and whatever comes next) -- ordering is irrelevant here
        # because everything runs inline.
        job = Job(id=f"inline-{file_id}", name=name, file_id=file_id)
        handler(job)
        return job

    def fake_process(session, file_id: int):  # noqa: ANN001, ANN202
        file_row = session.get(File, file_id)
        if file_row is not None:
            file_row.status = ProcessingStatus.READY
            session.commit()
        return ml_pipeline.ProcessResult(file_id, ProcessingStatus.READY, 0, 0, 0)

    def fake_index(session, file_id: int):  # noqa: ANN001, ANN202
        from app.ml import indexing

        return indexing.IndexResult(file_id, 0, 0)

    # `app.tasks.process` and `app.tasks.index` hold references to these same
    # module objects, so patching the attributes here is what the worker calls.
    monkeypatch.setattr(ml_pipeline, "process_file", fake_process)
    monkeypatch.setattr(_indexing, "index_file", fake_index)

    original = task_queue.submit
    task_queue.submit = run_now  # type: ignore[method-assign]
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        task_queue.submit = original  # type: ignore[method-assign]


def upload(client: TestClient, path: Path) -> dict:
    with path.open("rb") as handle:
        response = client.post("/files", files={"files": (path.name, io.BytesIO(handle.read()))})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_extract_list_and_open(client: TestClient, tmp_path: Path) -> None:
    path = fixtures.make_text_pdf(tmp_path / "terns.pdf", pages=2)
    body = upload(client, path)

    assert body["accepted"] == 1 and body["rejected"] == 0
    file_id = body["results"][0]["file_id"]

    detail = client.get(f"/files/{file_id}").json()
    # A text-only PDF has no images, so ML processing has nothing to do and the
    # file goes straight through to ready.
    assert detail["status"] == "ready"
    assert detail["chunk_count"] > 0
    assert detail["chunks"][0]["text"]

    listing = client.get("/files").json()
    assert any(f["id"] == file_id for f in listing)

    # "Open file" in the results view fetches the original bytes back.
    content = client.get(f"/files/{file_id}/content")
    assert content.status_code == 200
    assert content.content.startswith(b"%PDF")


def test_mixed_batch_reports_per_file_outcomes(client: TestClient, tmp_path: Path) -> None:
    """Seventeen good files must not be lost because three are unsupported."""
    good = fixtures.make_text_pdf(tmp_path / "good.pdf", pages=1)
    bad = fixtures.make_unsupported(tmp_path / "bad.xyz")

    response = client.post(
        "/files",
        files=[
            ("files", (good.name, good.read_bytes())),
            ("files", (bad.name, bad.read_bytes())),
        ],
    )
    body = response.json()

    assert response.status_code == 200
    assert body["accepted"] == 1
    assert body["rejected"] == 1
    rejected = next(r for r in body["results"] if not r["accepted"])
    assert ".pdf" in rejected["reason"], "the reason must tell the user what is supported"


def test_file_with_images_defers_when_models_are_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing model must not destroy successfully extracted content.

    The document's text stays searchable and the file stays in `extracted`, so
    `requeue_unprocessed` picks it up once the model is installed. Marking it
    `failed` would hide working content behind a red state and would need a
    manual retry.

    The missing model is simulated rather than assumed: on a provisioned dev box
    the weights are present, and a test that only passes on an unprovisioned
    machine silently stops testing anything.
    """
    from app.ml import loaders
    from app.ml import pipeline as ml_pipeline

    def raise_missing(*args: object, **kwargs: object):  # noqa: ANN202
        raise loaders.ModelNotAvailableError("moondream2", Path("nowhere"))

    # Restore the REAL process_file so the actual deferral logic runs, while
    # leaving indexing stubbed -- `monkeypatch.undo()` would restore both and
    # send the test into real embedding, which needs gigabytes of models.
    monkeypatch.setattr(ml_pipeline, "process_file", REAL_PROCESS_FILE)
    monkeypatch.setattr(ml_pipeline.captioning, "caption_images", raise_missing)
    monkeypatch.setattr(ml_pipeline.ocr, "read_images", raise_missing)

    path = fixtures.make_pdf_with_image(tmp_path / "survey.pdf")
    file_id = upload(client, path)["results"][0]["file_id"]

    detail = client.get(f"/files/{file_id}").json()
    assert detail["status"] == "extracted", "a missing model is not a file failure"
    assert detail["error"] is None, "a deferred capability is not an error on the file"
    assert detail["chunk_count"] > 0, "the extracted text must survive"


def test_asset_image_is_served(client: TestClient, tmp_path: Path) -> None:
    path = fixtures.make_image(tmp_path / "harbour.png")
    file_id = upload(client, path)["results"][0]["file_id"]

    detail = client.get(f"/files/{file_id}").json()
    thumb = next(a for a in detail["assets"] if a["kind"] == "thumbnail")

    response = client.get(f"/files/{file_id}/assets/{thumb['id']}/image")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_stats_reflect_the_library(client: TestClient, tmp_path: Path) -> None:
    upload(client, fixtures.make_text_pdf(tmp_path / "one.pdf", pages=1))
    upload(client, fixtures.make_docx(tmp_path / "two.docx"))

    stats = client.get("/files/stats").json()
    assert stats["total_files"] == 2
    assert stats["by_kind"] == {"pdf": 1, "docx": 1}
    assert stats["total_chunks"] > 0
    assert "originals" in stats["disk_usage_mb"]


def test_delete_removes_the_file(client: TestClient, tmp_path: Path) -> None:
    file_id = upload(client, fixtures.make_text_pdf(tmp_path / "x.pdf", pages=1))["results"][0][
        "file_id"
    ]

    assert client.delete(f"/files/{file_id}").json() == {"deleted": True}
    assert client.get(f"/files/{file_id}").status_code == 404


def test_missing_file_is_a_404_not_a_crash(client: TestClient) -> None:
    assert client.get("/files/99999").status_code == 404
    assert client.get("/files/99999/content").status_code == 404
    assert client.delete("/files/99999").status_code == 404


def test_supported_types_are_advertised(client: TestClient) -> None:
    types = client.get("/files/supported-types").json()
    assert ".pdf" in types and ".docx" in types and ".png" in types


def test_upload_chains_all_the_way_to_indexing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the extract -> process -> index chain must reach the end.

    The index step was silently never submitted, so files reached `ready` with
    everything extracted and captioned while the vector store stayed empty --
    the whole library was unsearchable and nothing reported an error. Nothing
    caught it because no test followed the chain past `process`.
    """
    from app.ml import indexing

    indexed: list[int] = []

    def record(session, file_id: int):  # noqa: ANN001, ANN202
        indexed.append(file_id)
        return indexing.IndexResult(file_id, 0, 0)

    monkeypatch.setattr(_indexing, "index_file", record)

    file_id = upload(client, fixtures.make_text_pdf(tmp_path / "chain.pdf", pages=1))["results"][0][
        "file_id"
    ]

    assert indexed == [file_id], "processing finished without queueing indexing"
