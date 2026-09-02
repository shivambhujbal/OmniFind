"""Which images get a CLIP vector, and which are left to OCR.

Only standalone image files are embedded. Pictures *inside* a document are not:
they are already reachable through the document's own text and through OCR of
the picture itself, so a CLIP vector duplicates a route that already works --
and it is far from free. One 52-page slide deck in the dev library extracts to
2054 embedded images, which was 75 minutes of CPU and 95% of the entire
indexing backlog, for a file OCR had already made searchable.

`ml.pipeline.CLIP_INDEXABLE_KINDS` is the rule; these tests pin both halves of
it, and the cleanup that removes vectors written before it narrowed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AssetKind, ProcessingStatus
from app.ingestion import pipeline
from app.ml import indexing, maintenance
from app.ml.pipeline import CLIP_INDEXABLE_KINDS, PROCESSABLE_KINDS
from tests import fixtures


@pytest.fixture
def images_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_image_search", True)
    monkeypatch.setattr(settings, "enable_captioning", False)


@pytest.fixture
def stub_clip(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record what reaches the vector store, without loading 2.9GB of CLIP."""
    upserted: list[list[int]] = []
    monkeypatch.setattr(
        indexing.embeddings_clip,
        "embed_images",
        lambda paths: (np.ones((len(paths), 4), dtype=np.float32), list(paths)),
    )
    monkeypatch.setattr(indexing.embeddings_clip, "embedding_dim", lambda: 4)
    monkeypatch.setattr(indexing.store, "ensure_collections", lambda **kw: None)
    monkeypatch.setattr(
        indexing.store, "upsert_images", lambda ids, vectors, payloads: upserted.append(list(ids))
    )
    return upserted


def ingest(session: Session, path: Path) -> object:
    file_row = pipeline.register_upload(session, path, path.name)
    session.commit()
    pipeline.extract_file(session, file_row.id)
    return file_row


def test_ocr_still_covers_images_inside_documents() -> None:
    """Narrowing the CLIP set must not narrow what OCR reads.

    OCR over an embedded figure is what makes a screenshotted table or a
    scanned page findable at all -- the opposite of redundant.
    """
    assert AssetKind.EMBEDDED_IMAGE in PROCESSABLE_KINDS
    assert AssetKind.PAGE_IMAGE in PROCESSABLE_KINDS
    assert AssetKind.EMBEDDED_IMAGE not in CLIP_INDEXABLE_KINDS
    assert AssetKind.PAGE_IMAGE not in CLIP_INDEXABLE_KINDS


def test_standalone_image_is_embedded(
    db_session: Session, tmp_path: Path, images_on: None, stub_clip: list[list[int]]
) -> None:
    """A picture the user actually added is the thing image search is for."""
    file_row = ingest(db_session, fixtures.make_image(tmp_path / "harbour.png"))
    written = indexing._index_images(db_session, file_row)

    assert written > 0
    assert stub_clip, "the vector must reach the store"
    sources = [a for a in file_row.assets if a.kind is AssetKind.SOURCE_IMAGE]
    assert sources and all(a.clip_embedded for a in sources)


def test_images_inside_a_document_are_not_embedded(
    db_session: Session, tmp_path: Path, images_on: None, stub_clip: list[list[int]]
) -> None:
    """The 75-minute case. OCR already reaches these; CLIP would duplicate it."""
    file_row = ingest(db_session, fixtures.make_pdf_with_image(tmp_path / "notes.pdf"))
    embedded = [a for a in file_row.assets if a.kind is AssetKind.EMBEDDED_IMAGE]
    assert embedded, "precondition: the PDF really does carry a picture"

    written = indexing._index_images(db_session, file_row)

    assert written == 0, "document images must not consume CLIP time"
    assert not stub_clip, "nothing should have reached the image collection"
    assert all(not a.clip_embedded for a in embedded)


def test_requeue_does_not_loop_on_unindexable_assets(
    db_session: Session, tmp_path: Path, images_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the kind filter, requeue is an infinite treadmill.

    Every document holding a picture would be queued on each start, indexed to
    zero because the kind is excluded, and queued again on the next one.
    """
    from app.tasks import index as index_tasks

    file_row = ingest(db_session, fixtures.make_pdf_with_image(tmp_path / "notes.pdf"))
    file_row.status = ProcessingStatus.READY
    for chunk in file_row.chunks:
        chunk.text_embedded = True
    db_session.commit()

    submitted: list[int] = []
    monkeypatch.setattr(index_tasks, "submit_indexing", lambda fid, name: submitted.append(fid))

    index_tasks.requeue_unindexed()

    assert file_row.id not in submitted, "a file with only document images must not be re-queued"


def test_prune_removes_vectors_written_before_the_rule_narrowed(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the SQLite flag is not enough -- Qdrant is a separate database.

    Left behind, these keep matching queries and crowd out the standalone
    photographs the collection exists for.
    """
    deleted: list[list[int]] = []
    monkeypatch.setattr(
        maintenance.store, "delete_points", lambda collection, ids: deleted.append(list(ids))
    )

    file_row = ingest(db_session, fixtures.make_pdf_with_image(tmp_path / "legacy.pdf"))
    embedded = [a for a in file_row.assets if a.kind is AssetKind.EMBEDDED_IMAGE]
    for asset in embedded:
        asset.clip_embedded = True  # as the old rule would have left it
    db_session.commit()

    pruned = maintenance.prune_image_vectors(db_session)

    assert pruned == len(embedded)
    assert deleted and set(deleted[0]) == {a.id for a in embedded}
    assert all(not a.clip_embedded for a in embedded), "flag must match reality"


def test_prune_leaves_standalone_images_alone(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotent, and not overzealous -- it runs at every startup."""
    monkeypatch.setattr(
        maintenance.store,
        "delete_points",
        lambda collection, ids: pytest.fail("deleted a source image"),
    )

    file_row = ingest(db_session, fixtures.make_image(tmp_path / "harbour.png"))
    for asset in file_row.assets:
        if asset.kind is AssetKind.SOURCE_IMAGE:
            asset.clip_embedded = True
    db_session.commit()

    assert maintenance.prune_image_vectors(db_session) == 0
