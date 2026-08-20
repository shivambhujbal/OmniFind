"""Ranking and deduplication -- pure logic, no models, no Qdrant.

This is where multimodal search is actually won or lost, so it is tested
independently of whether any weights are downloaded.
"""

from __future__ import annotations

from app.search.ranking import deduplicate_by_file, fuse, reciprocal_rank_fusion
from app.vectors.store import VectorHit


def text_hit(point_id: int, score: float, file_id: int = 1, **payload: object) -> VectorHit:
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
            "snippet": f"chunk {point_id}",
            **payload,
        },
    )


def image_hit(point_id: int, score: float, file_id: int = 2, **payload: object) -> VectorHit:
    return VectorHit(
        point_id=point_id,
        score=score,
        collection="images",
        payload={
            "file_id": file_id,
            "asset_id": point_id,
            "kind": "source_image",
            "file_name": f"photo{file_id}.png",
            "file_kind": "image",
            "caption": f"a picture {point_id}",
            **payload,
        },
    )


# --- fusion ------------------------------------------------------------


def test_raw_scores_do_not_decide_the_order() -> None:
    """The central reason RRF is used at all.

    bge scores a good text match around 0.7; CLIP scores an equally good image
    match around 0.25. Comparing those numbers directly would rank every text
    result above every image result regardless of relevance. Fusing by rank
    means the top image beats the second-place text hit.
    """
    text = [text_hit(1, 0.71), text_hit(2, 0.68)]
    images = [image_hit(10, 0.25)]

    fused = reciprocal_rank_fusion([text, images])
    order = [(r.chunk_id, r.asset_id) for r in fused]

    assert order[0] == (1, None), "the best text hit should still lead"
    assert order[1] == (None, 10), "the top image must outrank the second text hit"


def test_appearing_in_both_lists_wins() -> None:
    """An item ranked well in both spaces should beat a one-list leader."""
    shared = {"file_id": 7, "file_name": "invoice.pdf", "file_kind": "pdf"}
    in_both_text = VectorHit(99, 0.6, "text_chunks", {**shared, "chunk_id": 99, "kind": "ocr"})
    in_both_image = VectorHit(99, 0.3, "images", {**shared, "asset_id": 99, "kind": "page_image"})

    # A different item that tops the text list alone.
    solo = text_hit(1, 0.62, file_id=1)

    fused = fuse(
        text_hits=[solo, in_both_text],
        image_hits=[in_both_image],
        group_by_file=True,
    )
    assert fused[0].file_id == 7, "the file matching in both spaces should rank first"


def test_fusion_of_empty_lists_is_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_a_single_list_keeps_its_own_order() -> None:
    hits = [text_hit(1, 0.9), text_hit(2, 0.8), text_hit(3, 0.7)]
    fused = reciprocal_rank_fusion([hits, []])
    assert [r.chunk_id for r in fused] == [1, 2, 3]


def test_rrf_k_damps_the_top_rank_advantage() -> None:
    """Small k makes rank 1 dominate; large k flattens the curve."""
    text = [text_hit(1, 0.9)]
    images = [image_hit(10, 0.2), image_hit(11, 0.19)]

    sharp = reciprocal_rank_fusion([text, images], k=1)
    flat = reciprocal_rank_fusion([text, images], k=1000)

    gap_sharp = sharp[0].score - sharp[-1].score
    gap_flat = flat[0].score - flat[-1].score
    assert gap_sharp > gap_flat


# --- deduplication -----------------------------------------------------


def test_one_document_cannot_fill_the_whole_page() -> None:
    """A long report matching on every chunk must not hide every other file.

    Scores are kept close together so the weak-tail cutoff does not remove the
    second file for a different reason -- what is under test here is grouping.
    """
    text = [text_hit(i, 0.90 - i * 0.002, file_id=1) for i in range(1, 11)]
    text.append(text_hit(99, 0.88, file_id=2))

    results = fuse(text_hits=text, image_hits=[], limit=10)

    file_ids = [r.file_id for r in results]
    assert file_ids.count(1) == 1, "the long document should appear once"
    assert 2 in file_ids, "the other file must still be reachable"


def test_extra_passages_are_kept_as_supporting_evidence() -> None:
    """Collapsing a file must not throw away where else it matched."""
    text = [text_hit(1, 0.9, file_id=1), text_hit(2, 0.8, file_id=1), text_hit(3, 0.7, file_id=1)]

    results = fuse(text_hits=text, image_hits=[], limit=10)

    assert len(results) == 1
    assert len(results[0].supporting) == 2
    # Best first, and the headline is the strongest of the three.
    assert results[0].chunk_id == 1
    assert [s.chunk_id for s in results[0].supporting] == [2, 3]


def test_supporting_passages_are_capped() -> None:
    """A 500-chunk book should not attach 499 passages to one result."""
    text = [text_hit(i, 1.0 - i * 0.001, file_id=1) for i in range(1, 60)]
    results = deduplicate_by_file(reciprocal_rank_fusion([text, []]), limit=10)
    assert len(results[0].supporting) <= 5


def test_limit_is_respected() -> None:
    text = [text_hit(i, 0.9, file_id=i) for i in range(1, 30)]
    assert len(fuse(text_hits=text, image_hits=[], limit=5)) == 5


def test_grouping_can_be_turned_off() -> None:
    """Passage-level results, for a UI that wants every hit."""
    text = [text_hit(1, 0.90, file_id=1), text_hit(2, 0.89, file_id=1)]
    results = fuse(text_hits=text, image_hits=[], group_by_file=False)
    assert len(results) == 2


def test_result_payload_survives_into_the_response() -> None:
    """Every result must carry enough to open the source file at the right page."""
    hit = text_hit(5, 0.8, file_id=3, page_number=12, title="Coastal Survey")
    result = fuse(text_hits=[hit], image_hits=[])[0]

    assert result.file_id == 3
    assert result.page_number == 12
    assert result.title == "Coastal Survey"
    assert result.snippet
    assert result.match_kind == "text"

    payload = result.as_dict()
    assert payload["file_id"] == 3 and payload["page_number"] == 12


def test_image_results_use_the_caption_as_the_snippet() -> None:
    """Images have no text, so the caption is what the UI shows."""
    result = fuse(text_hits=[], image_hits=[image_hit(10, 0.3)])[0]
    assert result.snippet == "a picture 10"
    assert result.asset_id == 10
