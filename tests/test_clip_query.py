"""Query templating: bare nouns must not fall out of CLIP's distribution."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml import embeddings_clip


def test_query_is_encoded_under_several_caption_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLIP never saw bare nouns in training; it saw captions.

    Measured on a real library, "human" returned zero results and "person"
    returned one, while a full description worked. Templating the query is what
    closes that gap -- and it is the query side only, so no re-indexing.
    """
    seen: list[list[str]] = []

    def fake_embed_text(texts: list[str]):  # noqa: ANN202
        seen.append(list(texts))
        return np.eye(len(texts), 4, dtype=np.float32)

    monkeypatch.setattr(embeddings_clip, "embed_text", fake_embed_text)

    embeddings_clip.embed_query("person")

    assert len(seen) == 1, "one batched forward pass, not one per template"
    variants = seen[0]
    assert "person" in variants, "the raw query must stay in the mix"
    assert any(v.startswith("a photo of") for v in variants), "must add a caption form"
    assert len(variants) >= 2


def test_embedded_query_is_unit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cosine similarity is a dot product only if the query is normalised.

    Averaging unit vectors does not produce a unit vector, so the mean has to
    be renormalised or every image score is silently scaled down.
    """

    def fake_embed_text(texts: list[str]):  # noqa: ANN202
        rows = np.tile(np.array([[0.6, 0.8, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))
        rows[0] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return rows

    monkeypatch.setattr(embeddings_clip, "embed_text", fake_embed_text)

    vector = embeddings_clip.embed_query("anything")

    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
