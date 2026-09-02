"""Literal keyword search, to sit alongside the semantic one.

Dense embeddings cannot do exact lookup. Measured on a real library, searching
``Outward : 42595078122`` scored the chunk that literally contains that number
at **0.45**, while unrelated lab reports scored **0.57** -- the right answer
ranked *below* the noise. bge tokenises a long digit string into subword
fragments that carry no identity, so no threshold or reranking fixes it.

SQLite's FTS5 does exactly what is needed and is already in the database: it
indexes ``42595078122`` as one token and finds it instantly. Results are fused
with the semantic ones by rank, the same way text and image results already
are.

**Non-searchable chunks are still searched here.** A passage is excluded from
the *semantic* index when its embedding would be meaningless (a page of patent
numbers, a repeated OCR label -- see ``ingestion/quality.py``). That says
nothing about its literal text: if someone types an exact reference number that
appears in such a page, finding it is correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.logging_conf import get_logger

log = get_logger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9]+")

# A token worth treating as an identifier: it contains a digit and is long
# enough not to be a page number or a year fragment. Reference numbers, invoice
# numbers, case numbers, part codes.
_MIN_IDENTIFIER_LENGTH = 4

SNIPPET_CHARS = 400


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: int
    file_id: int
    rank: int
    # True when the hit matched an identifier-like token rather than ordinary
    # words. Those are exempt from the similarity floor: an exact match on a
    # reference number is evidence of a different kind from a cosine score.
    is_exact: bool
    payload: dict[str, object]


def tokenize(query: str) -> list[str]:
    return _TOKEN.findall(query)


def identifier_tokens(query: str) -> list[str]:
    """Tokens that look like a reference number, code or ID."""
    return [
        token
        for token in tokenize(query)
        if len(token) >= _MIN_IDENTIFIER_LENGTH and any(c.isdigit() for c in token)
    ]


def _match_expression(tokens: list[str], require_all: bool) -> str:
    """Build an FTS5 MATCH expression.

    Every token is double-quoted. FTS5 treats bare ``:``, ``-``, ``*`` and
    ``NEAR`` as operators, so an unquoted query like ``Outward : 42595078122``
    is a syntax error rather than a search.
    """
    quoted = [f'"{token}"' for token in tokens]
    return (" AND " if require_all else " OR ").join(quoted)


def search(session: Session, query: str, limit: int = 50) -> list[LexicalHit]:
    """Keyword search over chunk text, best first.

    Identifier-like tokens are required (AND) rather than merely preferred: if
    someone types a reference number they want *that* document, not everything
    sharing the word next to it.
    """
    tokens = tokenize(query)
    if not tokens:
        return []

    identifiers = identifier_tokens(query)
    if identifiers:
        expression = _match_expression(identifiers, require_all=True)
        is_exact = True
    else:
        expression = _match_expression(tokens, require_all=False)
        is_exact = False

    try:
        rows = session.execute(
            sql(
                """
                SELECT c.id, c.file_id, c.kind, c.page_number, c.text,
                       f.original_name, f.kind AS file_kind, f.title
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN files  f ON f.id = c.file_id
                WHERE chunks_fts MATCH :expression
                ORDER BY bm25(chunks_fts)
                LIMIT :limit
                """
            ),
            {"expression": expression, "limit": limit},
        ).all()
    except Exception as exc:  # noqa: BLE001 - a malformed query must not 500
        log.warning("keyword search failed for %r (%s): %s", query, expression, exc)
        return []

    hits = []
    for rank, row in enumerate(rows, start=1):
        hits.append(
            LexicalHit(
                chunk_id=row.id,
                file_id=row.file_id,
                rank=rank,
                is_exact=is_exact,
                payload={
                    "file_id": row.file_id,
                    "chunk_id": row.id,
                    "kind": row.kind.value if hasattr(row.kind, "value") else str(row.kind),
                    "file_name": row.original_name,
                    "file_kind": (
                        row.file_kind.value
                        if hasattr(row.file_kind, "value")
                        else str(row.file_kind)
                    ),
                    "title": row.title,
                    "page_number": row.page_number,
                    "snippet": row.text[:SNIPPET_CHARS],
                },
            )
        )

    if hits:
        log.info(
            "keyword search %r matched %d chunk(s)%s",
            query,
            len(hits),
            " on an identifier" if is_exact else "",
        )
    return hits
