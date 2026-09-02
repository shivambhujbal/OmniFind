"""Exact identifier lookup.

Dense embeddings cannot do this. Measured on a real library, searching
"Outward : 42595078122" scored the chunk that literally contains that number at
0.45 while unrelated lab reports scored 0.57 -- the right answer ranked *below*
the noise, and no threshold can fix an inversion. FTS5 finds it as one token.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.search import lexical
from app.search.ranking import drop_weak_tail, fuse
from app.vectors.store import VectorHit

REFERENCE = "42595078122"
CERTIFICATE = (
    "FORM - 8 [Rule No. 5] Form of Caste Certificate for De-Notified Tribe. "
    "Documents Verified: Copy of Aadhar Card, School Leaving Certificate. "
    f"CASTE CERTIFICATE Outward : {REFERENCE} Date : 10/09/2018 "
    f"Rev Case No : MRC : {REFERENCE} This is to certify that..."
)


# --- recognising an identifier -----------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Outward : 42595078122",
        "42595078122",
        "MRC : 42595078122",
        "invoice 2024-118",
        "part no A4C-9910",
    ],
)
def test_identifier_queries_are_recognised(query: str) -> None:
    assert lexical.identifier_tokens(query), f"no identifier found in {query!r}"


@pytest.mark.parametrize(
    "query",
    [
        "how far do sea birds travel",
        "recipe of smoked paprika",
        "devops notes",
        # Too short to be a reference number; page 5 is not an identifier.
        "page 5",
    ],
)
def test_ordinary_queries_are_not_treated_as_identifiers(query: str) -> None:
    assert not lexical.identifier_tokens(query), f"{query!r} wrongly looks like an ID"


def test_punctuation_does_not_break_the_query() -> None:
    """FTS5 reads a bare colon as a column filter, not as text.

    An unquoted "Outward : 42595078122" is a syntax error rather than a search,
    which would silently return nothing.
    """
    assert lexical.tokenize("Outward : 42595078122") == ["Outward", "42595078122"]
    assert lexical._match_expression(["Outward", "42595078122"], require_all=True) == (
        '"Outward" AND "42595078122"'
    )


# --- the index ---------------------------------------------------------


def test_an_exact_reference_number_is_found(db_session: Session) -> None:
    """The case that started this."""
    _add_chunk(db_session, 1, CERTIFICATE, "caste certificate.pdf")
    _add_chunk(db_session, 2, "Binary search runs in logarithmic time.", "DAA_exp_2B.docx")

    hits = lexical.search(db_session, "Outward : 42595078122")

    assert hits, "the reference number was not found"
    assert hits[0].payload["file_name"] == "caste certificate.pdf"
    assert hits[0].is_exact


def test_the_number_alone_finds_it(db_session: Session) -> None:
    _add_chunk(db_session, 1, CERTIFICATE, "caste certificate.pdf")
    hits = lexical.search(db_session, REFERENCE)
    assert hits and hits[0].chunk_id == 1


def test_a_different_number_does_not_match(db_session: Session) -> None:
    _add_chunk(db_session, 1, CERTIFICATE, "caste certificate.pdf")
    assert lexical.search(db_session, "42595078999") == []


def test_deleting_a_chunk_removes_it_from_the_keyword_index(db_session: Session) -> None:
    """External-content FTS5 keeps stale rows unless the trigger hands back the
    old text, and search would then return ids that no longer exist."""
    from app.db.models import Chunk

    _add_chunk(db_session, 1, CERTIFICATE, "caste certificate.pdf")
    assert lexical.search(db_session, REFERENCE)

    db_session.query(Chunk).filter(Chunk.id == 1).delete()
    db_session.commit()

    assert lexical.search(db_session, REFERENCE) == []


def test_non_searchable_chunks_are_still_findable_by_exact_text(db_session: Session) -> None:
    """A page of reference numbers is excluded from *semantic* search because
    its embedding is meaningless. Its literal text is still valid to look up."""
    _add_chunk(
        db_session,
        1,
        f"US 11,321,312 B2 Page 2 (56) {REFERENCE} 2016/0012044",
        "patent.pdf",
        searchable=False,
    )
    hits = lexical.search(db_session, REFERENCE)
    assert hits, "an exact lookup should reach chunks excluded from semantic search"


# --- ranking -----------------------------------------------------------


def hit(point_id: int, score: float, file_id: int, exact: bool = False) -> VectorHit:
    return VectorHit(
        point_id=point_id,
        score=score,
        collection="text_chunks",
        payload={
            "file_id": file_id,
            "chunk_id": point_id,
            "kind": "text",
            "file_name": f"file{file_id}.pdf",
            "file_kind": "pdf",
            "snippet": "...",
        },
        exact=exact,
    )


def test_an_exact_match_outranks_a_higher_scoring_guess() -> None:
    """The measured numbers: the correct document scores 0.46, the noise 0.57.

    Without an explicit rule these merely tie -- a keyword hit at rank 1 scores
    1/(60+1), exactly what the top dense hit scores.
    """
    results = fuse(
        text_hits=[hit(1, 0.572, file_id=1), hit(2, 0.463, file_id=2, exact=True)],
        image_hits=[],
    )

    assert results[0].file_id == 2, "the exact match should lead"
    assert results[0].exact is True


def test_semantic_noise_is_dropped_beneath_an_exact_match() -> None:
    """A document that does not contain the number is not a worse answer -
    it is not an answer."""
    results = fuse(
        text_hits=[
            hit(1, 0.463, file_id=1, exact=True),
            hit(2, 0.572, file_id=2),
            hit(3, 0.570, file_id=3),
        ],
        image_hits=[],
    )

    assert [r.file_id for r in results] == [1], f"noise survived: {[r.file_id for r in results]}"


def test_a_confident_semantic_hit_survives_alongside_an_exact_one() -> None:
    """Suppression is for baseline noise, not for genuinely strong matches."""
    results = fuse(
        text_hits=[hit(1, 0.463, file_id=1, exact=True), hit(2, 0.780, file_id=2)],
        image_hits=[],
    )
    assert {r.file_id for r in results} == {1, 2}


def test_ordinary_queries_keep_the_normal_tail_rule() -> None:
    """With no exact match the existing relative cutoff still applies."""
    results = fuse(text_hits=[hit(1, 0.668, file_id=1), hit(2, 0.551, file_id=2)], image_hits=[])
    assert [r.file_id for r in results] == [1]
    assert drop_weak_tail([]) == []


def _add_chunk(
    session: Session, chunk_id: int, text: str, file_name: str, searchable: bool = True
) -> None:
    from app.db.models import Chunk, ChunkKind, File, FileKind, ProcessingStatus

    file_row = session.get(File, chunk_id)
    if file_row is None:
        file_row = File(
            id=chunk_id,
            sha256=f"{chunk_id:064d}",
            original_name=file_name,
            stored_path=f"xx/{chunk_id}.pdf",
            kind=FileKind.PDF,
            mime_type="application/pdf",
            size_bytes=1,
            status=ProcessingStatus.READY,
        )
        session.add(file_row)
        session.flush()

    session.add(
        Chunk(
            id=chunk_id,
            file_id=file_row.id,
            kind=ChunkKind.TEXT,
            ordinal=chunk_id,
            text=text,
            char_count=len(text),
            searchable=searchable,
        )
    )
    session.commit()
