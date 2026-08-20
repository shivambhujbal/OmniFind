"""Low-information passages must stay out of the search index.

From a real failure: searching a library for "EEIM notes" returned, at
essentially the same score as the correct document, a chunk from a patent PDF
reading "US 11,321,312 B2 Page 2 (56) 2014/0004027 2016/0012044 ..." -- 400
characters containing five words.

Such passages sit near the centre of the embedding space, which in high
dimensions makes them a near neighbour of *every* query.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.ingestion.quality import assess, is_searchable

# Verbatim from the library that exposed the problem.
PATENT_REFERENCES = (
    "US 11,321,312 B2 Page 2 (56) 2014/0004027 2016/0012044 2016/0012045 2016/0012057 "
    "2016/0012058 2016/0012092 2016/0012106 2016/0012119 2016/0012122 2016/0012125 "
    "2016/0012126 2016/0012336 2016/0093050 2016/0336006 2018/0004752 2018/0203848"
)
FIGURE_CALLOUTS = "U.S. Patent May 3 , 2022 Sheet 8 of 8 US 11,321,312 B2 322 322 328 300 324 326"
DIAGRAM_OCR = "Mircroservice Clients Mircroservice Mircroservice Mircroservice Mircroservice"
DIAGNOSTIC_DUMP = "ft09-0d8bbf25 3 gave_up 47.62 4/6 82 72276 tokens=108882 g50t-5849a774 0"

REAL_PROSE = (
    "Quality Management: Implementing quality control and quality assurance processes to "
    "maintain high standards in production. Application in Engineering Economics and "
    "Industrial Management: these concepts are used to make informed decisions."
)
SHORT_OCR = "SCANNED INVOICE 2024-118\nTotal due: 4820.00"
DENSE_TECHNICAL = (
    "U.S. Patent May 3, 2022 Sheet 7 of 8 US 11,321,312 B2 190 240 ENCODED SENTENCE "
    "VECTOR PRODUCED BY ENCODER MODEL TRAINED ON SENTENCE SEQUENCES"
)


@pytest.mark.parametrize(
    ("text", "expected_reason"),
    [
        (PATENT_REFERENCES, "actual text"),
        (FIGURE_CALLOUTS, "actual text"),
        (DIAGNOSTIC_DUMP, "actual text"),
        (DIAGRAM_OCR, "repeated"),
        ("", "empty"),
        ("   \n  ", "empty"),
        ("Page 12", "distinct word"),
    ],
)
def test_content_free_passages_are_rejected(text: str, expected_reason: str) -> None:
    verdict = assess(text)
    assert not verdict.searchable, f"should have been rejected: {text[:60]!r}"
    assert expected_reason in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "text",
    [
        REAL_PROSE,
        # Short but genuinely useful -- this is what makes a scanned invoice
        # findable, so an over-eager word-count rule would break the feature
        # that OCR exists for.
        SHORT_OCR,
        # Dense technical text with lots of numbers is still real content.
        DENSE_TECHNICAL,
        "Bioluminescence in deep water is produced by symbiotic bacteria.",
    ],
)
def test_real_content_is_kept(text: str) -> None:
    assert is_searchable(text), f"should have been kept: {text[:60]!r}"


def test_the_offending_chunk_and_the_correct_one_are_told_apart() -> None:
    """The exact pair that scored 0.60 and 0.60 against "EEIM notes"."""
    assert is_searchable(REAL_PROSE)
    assert not is_searchable(PATENT_REFERENCES)


# --- wiring ------------------------------------------------------------


def test_junk_chunks_are_stored_but_not_indexed(db_session: Session, tmp_path: Path) -> None:
    """They remain part of the document; they just are not search targets.

    Dropping them entirely would quietly remove content from the file view,
    where a user may legitimately want to see the page as extracted.
    """
    from app.db.models import Chunk, ChunkKind, File, FileKind, ProcessingStatus

    file_row = File(
        sha256="a" * 64,
        original_name="patent.pdf",
        stored_path="aa/x.pdf",
        kind=FileKind.PDF,
        mime_type="application/pdf",
        size_bytes=1,
        status=ProcessingStatus.EXTRACTED,
    )
    db_session.add(file_row)
    db_session.flush()

    from app.ingestion import quality

    for ordinal, text in enumerate([REAL_PROSE, PATENT_REFERENCES]):
        db_session.add(
            Chunk(
                file_id=file_row.id,
                kind=ChunkKind.TEXT,
                ordinal=ordinal,
                text=text,
                char_count=len(text),
                searchable=quality.is_searchable(text),
            )
        )
    db_session.commit()

    stored = db_session.query(Chunk).order_by(Chunk.ordinal).all()
    assert len(stored) == 2, "both chunks belong to the document"
    assert stored[0].searchable is True
    assert stored[1].searchable is False


def test_reclassification_removes_stale_junk_from_the_index(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing library must be cleaned without re-importing.

    Re-running OCR and captioning to apply a text rule would cost hours on CPU.
    """
    from app.db.models import Chunk, ChunkKind, File, FileKind, ProcessingStatus
    from app.ml import maintenance

    file_row = File(
        sha256="b" * 64,
        original_name="patent.pdf",
        stored_path="bb/x.pdf",
        kind=FileKind.PDF,
        mime_type="application/pdf",
        size_bytes=1,
        status=ProcessingStatus.READY,
    )
    db_session.add(file_row)
    db_session.flush()

    # Classified under the OLD rules: junk marked searchable and indexed.
    junk = Chunk(
        file_id=file_row.id,
        kind=ChunkKind.TEXT,
        ordinal=0,
        text=PATENT_REFERENCES,
        char_count=len(PATENT_REFERENCES),
        searchable=True,
        text_embedded=True,
    )
    good = Chunk(
        file_id=file_row.id,
        kind=ChunkKind.TEXT,
        ordinal=1,
        text=REAL_PROSE,
        char_count=len(REAL_PROSE),
        searchable=True,
        text_embedded=True,
    )
    db_session.add_all([junk, good])
    db_session.commit()

    deleted: list[list[int]] = []
    monkeypatch.setattr(
        maintenance.store, "delete_points", lambda collection, ids: deleted.append(ids)
    )

    result = maintenance.reclassify_chunks(db_session)

    assert result.demoted == 1
    assert deleted == [[junk.id]], "the junk chunk's vector must be deleted"
    db_session.refresh(junk)
    db_session.refresh(good)
    assert junk.searchable is False
    assert good.searchable is True


def test_reclassification_is_idempotent(db_session: Session) -> None:
    """It runs at every startup, so a second pass must change nothing."""
    from app.ml import maintenance

    first = maintenance.reclassify_chunks(db_session)
    second = maintenance.reclassify_chunks(db_session)

    assert second.demoted == 0
    assert second.promoted == 0
    assert second.checked == first.checked
