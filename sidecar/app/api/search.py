"""Search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from app.logging_conf import get_logger
from app.ml import loaders
from app.search import service
from app.vectors import store

log = get_logger(__name__)

router = APIRouter(tags=["search"])


class SearchResultOut(BaseModel):
    file_id: int
    file_name: str
    file_kind: str
    title: str | None
    # Ordering artefact; not meaningful to a human. See ranking.SearchResult.
    score: float
    # The real cosine similarity -- this is what to show.
    similarity: float
    match_kind: str
    page_number: int | None
    snippet: str | None
    chunk_id: int | None
    asset_id: int | None
    # True when the passage was found by literal keyword match on an identifier
    # rather than by similarity. The UI labels these differently because the
    # cosine is not what makes them right.
    exact: bool = False
    supporting: list[SearchResultOut] = []


class SearchResponseOut(BaseModel):
    query: str
    results: list[SearchResultOut]
    text_candidates: int
    image_candidates: int
    # False when the image index could not be searched -- the results are
    # text-only, and the UI should say so rather than imply completeness.
    searched_images: bool
    # False when even the best hit is barely above the library's baseline. The
    # results are the closest available, not answers.
    confident: bool
    # Paging: `total` counts every match, `results` holds only this page.
    total: int
    page: int
    page_size: int
    total_pages: int


class IndexStatusOut(BaseModel):
    collections: dict[str, int]
    text_dim: int
    image_dim: int
    image_search: bool


@router.get("/search", response_model=SearchResponseOut)
def run_search(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    limit: int = Query(default=0, ge=0, le=200),
    include_images: bool = True,
    group_by_file: bool = True,
    kinds: service.KindFilter = Query(
        default=service.KindFilter.ALL,
        description="Restrict to documents (pdf, docx) or images",
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=0, ge=0, le=100),
) -> SearchResponseOut:
    try:
        response = service.search(
            q,
            limit=limit or settings.search_limit,
            include_images=include_images,
            group_by_file=group_by_file,
            kinds=kinds,
            page=page,
            page_size=page_size or settings.search_page_size,
        )
    except loaders.ModelBusyError as exc:
        # 503 with a different meaning: nothing is wrong, the machine is busy.
        # Retrying later genuinely works, which a 500 would not suggest.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except loaders.ModelNotAvailableError as exc:
        # 503, not 500: the service is fine, a capability is missing, and the
        # message says exactly how to provide it.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponseOut(
        query=response.query,
        results=[SearchResultOut(**r.as_dict()) for r in response.results],
        text_candidates=response.text_candidates,
        image_candidates=response.image_candidates,
        searched_images=response.searched_images,
        confident=response.confident,
        total=response.total,
        page=response.page,
        page_size=response.page_size,
        total_pages=max(1, -(-response.total // response.page_size)),
    )


@router.get("/search/status", response_model=IndexStatusOut)
def index_status() -> IndexStatusOut:
    """How much is indexed. Deliberately does not load the models."""
    return IndexStatusOut(
        collections=store.counts(),
        text_dim=settings.text_embedding_dim,
        image_dim=settings.clip_embedding_dim,
        image_search=settings.enable_image_search,
    )
