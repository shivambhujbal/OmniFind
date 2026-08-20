"""Fuse and deduplicate results from two different embedding spaces.

The hard part of multimodal search is not retrieval, it is combining the two
result lists. bge cosine scores for a good match sit around 0.6-0.8; CLIP's sit
around 0.2-0.35 for an equally good match. Comparing or averaging those raw
numbers would let text results dominate every query regardless of relevance --
the numbers are not on the same scale and never will be.

**Reciprocal Rank Fusion** avoids the problem by discarding the magnitudes and
using only each item's *rank within its own list*:

    score(item) = sum over lists of  1 / (k + rank)

Each list votes with its ordering, which is the part that is meaningful across
models. ``k`` (60, from the original paper) damps the advantage of rank 1 so a
single list cannot monopolise the results. An item found near the top of *both*
lists beats an item at the top of one, which is exactly the behaviour wanted
from a query like "invoice" that matches OCR text and a scanned page image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.ingestion.titles import clean_title
from app.vectors.store import VectorHit


@dataclass
class SearchResult:
    """One result, after fusion and deduplication."""

    file_id: int
    file_name: str
    file_kind: str
    title: str | None
    # Fusion score, used for ORDERING only. It is a rank artefact
    # (1/(k+rank)), so its magnitude means nothing: rank 1 is always 0.0164 and
    # rank 2 always 0.0161, whether the match was excellent or barely passed the
    # threshold. Never show this to a user.
    score: float
    # The actual cosine similarity from the model that found this hit. This is
    # the number that means something -- 0.78 is a strong match, 0.56 a weak
    # one -- and it is what the UI displays.
    similarity: float = 0.0
    # Where the match came from: "text", "ocr", "caption", or an image kind.
    match_kind: str = "text"
    page_number: int | None = None
    snippet: str | None = None
    chunk_id: int | None = None
    asset_id: int | None = None
    # Which collection this hit came from. Not exposed in the API -- it exists
    # so per-file scoring can tell "matched in both spaces" from "matched twice
    # in one".
    source_collection: str = ""
    # Every passage that matched in this file, best first. The UI shows the
    # first and can expand the rest.
    supporting: list[SearchResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_kind": self.file_kind,
            "title": self.title,
            "score": round(self.score, 5),
            "similarity": round(self.similarity, 4),
            "match_kind": self.match_kind,
            "page_number": self.page_number,
            "snippet": self.snippet,
            "chunk_id": self.chunk_id,
            "asset_id": self.asset_id,
            "supporting": [s.as_dict() for s in self.supporting],
        }


def _to_result(hit: VectorHit, score: float) -> SearchResult:
    payload = hit.payload
    file_name = str(payload.get("file_name", ""))
    return SearchResult(
        source_collection=hit.collection,
        file_id=int(payload.get("file_id", 0)),
        file_name=file_name,
        file_kind=str(payload.get("file_kind", "")),
        # Re-cleaned here, not just at ingest: the title is denormalised into
        # the vector payload, so files indexed before the cleaner improved
        # would otherwise keep showing "(anonymous)" until re-imported.
        title=clean_title(payload.get("title"), file_name),
        score=score,
        similarity=hit.score,
        match_kind=str(payload.get("kind", "text")),
        page_number=payload.get("page_number"),
        snippet=payload.get("snippet") or payload.get("caption"),
        chunk_id=payload.get("chunk_id"),
        asset_id=payload.get("asset_id"),
    )


def reciprocal_rank_fusion(
    result_lists: list[list[VectorHit]], k: int | None = None
) -> list[SearchResult]:
    """Merge ranked lists from different models into one ordering."""
    k = k if k is not None else settings.rrf_k

    scores: dict[tuple[str, int], float] = {}
    best_hit: dict[tuple[str, int], VectorHit] = {}

    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.collection, hit.point_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # A point appears once per list, so first sighting is enough.
            best_hit.setdefault(key, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [_to_result(best_hit[key], score) for key, score in ordered]


def deduplicate_by_file(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Collapse multiple passages from one file into a single result.

    Without this, a long document whose every chunk mentions the query fills the
    entire first page and hides every other file. The user is looking for *a
    file*; the additional passages are useful as supporting evidence, not as
    separate results, so they are attached rather than dropped.

    **Scoring a file: sum the best hit from each space, not all hits.**

    Taking the plain maximum would waste the most useful signal there is --
    a file whose OCR text *and* whose page image both match is corroborated by
    two independent retrievers, and should outrank a file that matched once.
    Summing every passage instead would just re-introduce the long-document
    problem, since a 200-page report accumulates hits by sheer length.

    Taking the best per collection and adding those does exactly what is wanted:
    matching in both spaces helps, matching forty times in one does not. It also
    needs no tuning constant -- it is the same damped-sum idea as RRF itself,
    applied one level up.
    """
    by_file: dict[int, SearchResult] = {}
    best_per_space: dict[int, dict[str, float]] = {}

    for result in results:
        space = result.source_collection
        spaces = best_per_space.setdefault(result.file_id, {})
        spaces[space] = max(spaces.get(space, 0.0), result.score)

        existing = by_file.get(result.file_id)
        if existing is None:
            by_file[result.file_id] = result
        elif result.score > existing.score:
            # Strongest passage becomes the headline; the rest sit behind it.
            result.supporting = [existing, *existing.supporting]
            by_file[result.file_id] = result
        else:
            existing.supporting.append(result)

    for file_id, result in by_file.items():
        result.score = sum(best_per_space[file_id].values())
        # Similarity stays the headline passage's own value. Summing it would
        # be meaningless -- similarities are not additive.
        result.similarity = max([result.similarity, *(s.similarity for s in result.supporting)])

    ranked = sorted(by_file.values(), key=lambda r: r.score, reverse=True)
    for result in ranked:
        result.supporting = sorted(result.supporting, key=lambda r: r.score, reverse=True)[:5]
    return ranked[:limit]


def drop_weak_tail(results: list[SearchResult], ratio: float | None = None) -> list[SearchResult]:
    """Remove results far weaker than the best one.

    The absolute floor (``min_text_score``) removes results that match nothing.
    This removes the ones that merely match *less* -- the flat tail sitting at
    the library's baseline similarity, which is what made a resume appear under
    a recipe search.

    Expressed as a fraction of the best hit rather than a fixed margin, so it
    adapts to the query: a vague query whose best hit only reaches 0.60 gets a
    proportionally lower cutoff than a sharp one reaching 0.79. The top result
    is never dropped.
    """
    if not results:
        return results

    ratio = settings.relative_score_floor if ratio is None else ratio
    cutoff = results[0].similarity * ratio
    return [results[0], *(r for r in results[1:] if r.similarity >= cutoff)]


def fuse(
    text_hits: list[VectorHit],
    image_hits: list[VectorHit],
    limit: int | None = None,
    group_by_file: bool = True,
) -> list[SearchResult]:
    """Full ranking pass: fuse the two spaces, collapse per file, trim the tail."""
    limit = limit or settings.search_limit
    fused = reciprocal_rank_fusion([text_hits, image_hits])
    grouped = fused[:limit] if not group_by_file else deduplicate_by_file(fused, limit)
    return drop_weak_tail(grouped)
