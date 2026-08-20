"""Relevance filtering and title cleaning.

Both come from the same real failure: a query for "recipe of smoked peparika"
returned all 13 files in the library, the second of them a resume, with scores
that looked identical (0.0164 vs 0.0161) and a headline reading "(anonymous)".
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.ingestion.titles import clean_title
from app.search.ranking import fuse
from app.vectors.store import VectorHit


def hit(point_id: int, score: float, file_id: int = 1, **payload: object) -> VectorHit:
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
            "snippet": "…",
            **payload,
        },
    )


# --- the score the user sees -------------------------------------------


def test_results_carry_the_real_similarity_not_the_fusion_score() -> None:
    """The fusion score is a rank artefact and must never be shown.

    Rank 1 is 1/(60+1)=0.0164 and rank 2 is 1/(60+2)=0.0161 *whatever* the
    underlying match quality was, so displaying it told the user nothing and
    made an excellent match look identical to a barely-passing one.
    """
    # Two close scores, so the weak-tail cutoff leaves both -- this test is
    # about which number is carried through, not about trimming.
    results = fuse(text_hits=[hit(1, 0.785), hit(2, 0.771, file_id=2)], image_hits=[])

    assert results[0].similarity == pytest.approx(0.785)
    assert results[1].similarity == pytest.approx(0.771)
    # And the ordering artefact is still there for ranking, but distinct.
    assert results[0].score != results[0].similarity


def test_a_grouped_file_reports_its_best_passage_similarity() -> None:
    """Similarities are not additive; summing them would be meaningless."""
    results = fuse(
        text_hits=[hit(1, 0.62, file_id=1), hit(2, 0.78, file_id=1), hit(3, 0.59, file_id=1)],
        image_hits=[],
    )

    assert len(results) == 1
    assert results[0].similarity == pytest.approx(0.78)


def test_similarity_is_exposed_over_the_api() -> None:
    from app.api.search import SearchResultOut

    result = fuse(text_hits=[hit(1, 0.7)], image_hits=[])[0]
    payload = SearchResultOut(**result.as_dict())
    assert payload.similarity == pytest.approx(0.7)


# --- the relevance floor -----------------------------------------------


def test_the_floor_sits_between_noise_and_real_matches() -> None:
    """Measured on a real mixed library; see config.min_text_score.

    bge does not spread its vectors over [0,1]: unrelated text still scores
    0.42-0.52 against any query. Without a floor the store returns its nearest
    neighbours however far away they are, so every document came back for
    every query.
    """
    noise_ceiling = 0.53  # highest observed for a query nothing could answer
    weakest_real_match = 0.62  # lowest observed for a correct answer

    assert noise_ceiling < settings.min_text_score < weakest_real_match


def test_the_floor_is_passed_to_the_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtering must happen in the engine, not after the fact.

    Post-filtering would silently shrink the candidate pool: the store would
    still return its N nearest, most of which get discarded, so genuinely
    relevant results beyond position N never appear at all.
    """
    import numpy as np

    from app.vectors import store

    seen: dict[str, object] = {}

    class FakeClient:
        def query_points(self, **kwargs: object):  # noqa: ANN202
            seen.update(kwargs)

            class Empty:
                points: list[object] = []

            return Empty()

    monkeypatch.setattr(store, "get_client", lambda: FakeClient())

    store.search_text(np.zeros(4, dtype=np.float32))
    assert seen["score_threshold"] == settings.min_text_score

    store.search_images(np.zeros(4, dtype=np.float32))
    assert seen["score_threshold"] == settings.min_image_score


def test_image_and_text_floors_are_separate() -> None:
    """CLIP and bge score on completely different scales.

    A single shared threshold would either flood the results with weak image
    matches or suppress every good one.
    """
    assert settings.min_image_score < settings.min_text_score


# --- title cleaning ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "(anonymous)",
        "anonymous",
        "Untitled",
        "document",
        "Slide 1",
        "PowerPoint Presentation",
        "  ",
        "ab",
        None,
        # Bare identifiers: a patent number and a build artefact, both seen in
        # a real library.
        "US00000011321312B220220503",
        "anim-20260807-anim",
        # The producing tool echoing the source filename.
        "Microsoft Word - Household_Budget.docx",
        "report.pdf",
    ],
)
def test_junk_metadata_titles_are_rejected(raw: str | None) -> None:
    """A result headed "(anonymous)" is worse than one headed with a filename."""
    assert clean_title(raw, "some_file.pdf") is None


@pytest.mark.parametrize(
    "raw",
    [
        "Arctic Tern Migration",
        "Keeping a Sourdough Starter",
        "Shivam Bhujbal Resume",
        "Bioluminescence",
        "2024 Annual Report",
    ],
)
def test_real_titles_are_kept(raw: str) -> None:
    assert clean_title(raw, "whatever.pdf") == raw


def test_a_title_that_merely_repeats_the_filename_is_dropped() -> None:
    """It adds nothing the reader is not already being shown."""
    assert (
        clean_title("Field Notes Migratory Songbirds", "Field_Notes_Migratory_Songbirds.pdf")
        is None
    )
    assert clean_title("quarterly notes", "Quarterly-Notes.docx") is None


def test_titles_are_cleaned_at_display_time_too() -> None:
    """Already-indexed files must improve without a re-import.

    The title is denormalised into the vector payload, so a library indexed
    before the cleaner improved would otherwise keep showing "(anonymous)".
    """
    result = fuse(
        text_hits=[hit(1, 0.8, title="(anonymous)", file_name="Recipe_Smoked_Paprika.pdf")],
        image_hits=[],
    )[0]

    assert result.title is None
    assert result.file_name == "Recipe_Smoked_Paprika.pdf"


# --- the weak tail -----------------------------------------------------


def test_the_flat_tail_below_the_best_hit_is_dropped() -> None:
    """Real numbers from "notes of devops subject" on a real library.

    The absolute floor removes results that match nothing. This removes the
    ones that merely match *less*: 0.55 here is the library's baseline
    similarity against any query, not an answer.
    """
    results = fuse(
        text_hits=[
            hit(1, 0.668, file_id=1),
            hit(2, 0.551, file_id=2),
            hit(3, 0.550, file_id=3),
            hit(4, 0.550, file_id=4),
        ],
        image_hits=[],
    )

    assert [r.file_id for r in results] == [
        1
    ], f"tail not trimmed: {[r.similarity for r in results]}"


def test_a_genuinely_close_second_survives() -> None:
    """Trimming must not amputate real alternatives.

    From "what programming languages does this person know?", where two
    resumes and a patent about language models all legitimately match.
    """
    results = fuse(
        text_hits=[
            hit(1, 0.621, file_id=1),
            hit(2, 0.616, file_id=2),
            hit(3, 0.608, file_id=3),
        ],
        image_hits=[],
    )

    assert len(results) == 3


def test_the_top_result_is_never_dropped() -> None:
    """Even a lone weak match is the best answer available; show it."""
    results = fuse(text_hits=[hit(1, 0.56, file_id=1)], image_hits=[])
    assert len(results) == 1


def test_the_cutoff_scales_with_the_query_not_a_fixed_margin() -> None:
    """A vague query gets a proportionally lower cutoff than a sharp one.

    With a fixed margin, the same absolute gap would be treated identically
    whether the best hit was 0.60 or 0.95, which is wrong: a 0.04 gap below a
    weak 0.60 is noise, the same gap below 0.95 is a real alternative.
    """
    from app.search.ranking import drop_weak_tail

    sharp = fuse(text_hits=[hit(1, 0.95, file_id=1), hit(2, 0.91, file_id=2)], image_hits=[])
    vague = fuse(text_hits=[hit(1, 0.60, file_id=1), hit(2, 0.56, file_id=2)], image_hits=[])

    assert len(sharp) == 2, "0.91 is 96% of 0.95 and should survive"
    assert len(vague) == 1, "0.56 is 93% of 0.60 and is baseline noise"
    assert drop_weak_tail([]) == []


# --- confidence --------------------------------------------------------


def test_a_weak_best_hit_is_reported_as_not_confident(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no threshold that cleanly separates "no answer" from "weak".

    Measured over 20 queries on a real library: the best score for a question
    the library could NOT answer reached 0.572, while the worst for one it
    could answer was 0.581. Rather than pretend to a precision the model does
    not have, the app returns the closest match and flags that it is unsure.
    """
    import numpy as np

    from app.search import service
    from app.vectors.store import VectorHit

    def weak_hits(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return [
            VectorHit(
                1,
                0.57,
                "text_chunks",
                {"file_id": 1, "chunk_id": 1, "file_name": "a.pdf", "file_kind": "pdf"},
            )
        ]

    monkeypatch.setattr(
        service.embeddings_text, "embed_query", lambda q: np.zeros(4, dtype=np.float32)
    )
    monkeypatch.setattr(service.store, "search_text", weak_hits)

    response = service.search("something the library cannot answer", include_images=False)

    assert response.results, "the closest match is still worth returning"
    assert response.confident is False


def test_a_strong_best_hit_is_reported_as_confident(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from app.search import service
    from app.vectors.store import VectorHit

    def strong_hits(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return [
            VectorHit(
                1,
                0.79,
                "text_chunks",
                {"file_id": 1, "chunk_id": 1, "file_name": "a.pdf", "file_kind": "pdf"},
            )
        ]

    monkeypatch.setattr(
        service.embeddings_text, "embed_query", lambda q: np.zeros(4, dtype=np.float32)
    )
    monkeypatch.setattr(service.store, "search_text", strong_hits)

    assert service.search("recipe of smoked paprika", include_images=False).confident is True


def test_an_empty_query_is_not_confident() -> None:
    from app.search import service

    response = service.search("   ")
    assert response.results == []
    assert response.confident is False
