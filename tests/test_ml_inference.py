"""Phase 3 inference against the real vendored weights.

Each test skips when its model is not present, so the suite still passes on a
clean checkout. On a provisioned machine these are the tests that would catch a
model loading but producing garbage -- a shape mismatch, an unnormalised vector,
a tokenizer silently truncating.

They are slow by nature (loading ViT-H/14 alone takes tens of seconds), so they
share module-scoped fixtures rather than reloading per test.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.ml import loaders
from tests import fixtures

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _one_heavy_model_at_a_time(request: pytest.FixtureRequest) -> None:
    """Keep only the model a test actually needs resident.

    bge (1.3GB) + ViT-H/14 (2.9GB) + moondream2 (4.1GB) is over 8GB in one
    process. On a 16GB machine that exceeds the Windows commit limit and the
    interpreter dies with an access violation mid-suite. Each test declares the
    model it needs with `@pytest.mark.model(...)`; everything else is unloaded
    first. Loaders are lru_cached, so consecutive tests on the same model still
    pay the load cost only once.
    """
    marker = request.node.get_closest_marker("model")
    wanted = marker.args[0] if marker else None

    loaders_by_name = {
        "text": loaders.load_text_encoder,
        "clip": loaders.load_clip,
        "caption": loaders.load_moondream,
    }
    for name, loader in loaders_by_name.items():
        if name != wanted:
            loader.cache_clear()

    # cache_clear() only drops the reference; the multi-gigabyte tensors are
    # not actually freed until the collector runs. Without this the next model
    # allocates on top of the previous one and the process dies.
    gc.collect()
    loaders.empty_device_cache()


def _available(name: str) -> bool:
    return loaders.available_models()[name]["available"]


requires_text = pytest.mark.skipif(
    not _available("text_embeddings"), reason="bge weights not downloaded"
)
requires_clip = pytest.mark.skipif(not _available("clip"), reason="OpenCLIP weights not downloaded")
requires_caption = pytest.mark.skipif(
    not _available("moondream2"), reason="moondream2 weights not downloaded"
)
requires_ocr = pytest.mark.skipif(
    not _available(settings.ocr_engine), reason="no OCR engine installed"
)


# --- text embeddings ---------------------------------------------------


@requires_text
@pytest.mark.model("text")
def test_text_embeddings_are_normalised_and_correctly_shaped() -> None:
    from app.ml import embeddings_text

    vectors = embeddings_text.embed_documents(["one", "two", "three"])

    assert vectors.shape == (3, settings.text_embedding_dim)
    assert vectors.dtype == np.float32
    # Normalised, so cosine == dot and Qdrant's distance metrics agree.
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@requires_text
@pytest.mark.model("text")
def test_configured_dim_matches_the_actual_model() -> None:
    """A mismatch here silently corrupts the Qdrant collection in phase 4."""
    from app.ml import embeddings_text

    assert embeddings_text.embedding_dim() == settings.text_embedding_dim


@requires_text
@pytest.mark.model("text")
def test_text_search_ranks_by_meaning_not_keywords() -> None:
    """The whole premise of the app: no shared words, still the top hit."""
    from app.ml import embeddings_text

    documents = [
        "Arctic terns migrate nearly forty thousand kilometres each year.",
        "The invoice total came to 4,820.00 including tax.",
        "Sourdough needs a starter fed twice daily at room temperature.",
    ]
    vectors = embeddings_text.embed_documents(documents)
    query = embeddings_text.embed_query("how far do sea birds travel when migrating?")

    scores = vectors @ query
    assert int(np.argmax(scores)) == 0
    assert scores[0] > scores[1] + 0.15, "the correct hit should win clearly"


@requires_text
@pytest.mark.model("text")
def test_empty_input_returns_an_empty_matrix_not_an_error() -> None:
    from app.ml import embeddings_text

    vectors = embeddings_text.embed_documents([])
    assert vectors.shape == (0, settings.text_embedding_dim)


# --- CLIP --------------------------------------------------------------


@requires_clip
@pytest.mark.model("clip")
def test_clip_image_embeddings_are_normalised(tmp_path: Path) -> None:
    from app.ml import embeddings_clip

    paths = [fixtures.make_image(tmp_path / f"img{i}.png") for i in range(2)]
    vectors, kept = embeddings_clip.embed_images(paths)

    assert len(kept) == 2
    assert vectors.shape == (2, settings.clip_embedding_dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


@requires_clip
@pytest.mark.model("clip")
def test_clip_skips_unreadable_images_without_shifting_rows(tmp_path: Path) -> None:
    """The reason embed_images returns the kept paths as well as the vectors.

    If a bad image silently dropped a row, every later vector would be attached
    to the wrong asset -- a corruption that no exception would reveal.
    """
    from app.ml import embeddings_clip

    good = fixtures.make_image(tmp_path / "good.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    vectors, kept = embeddings_clip.embed_images([bad, good])

    assert kept == [good]
    assert vectors.shape[0] == 1


@requires_clip
@pytest.mark.model("clip")
def test_clip_matches_text_to_the_right_image(tmp_path: Path) -> None:
    """Cross-modal retrieval: text query, image result, no OCR involved."""
    from PIL import Image, ImageDraw

    from app.ml import embeddings_clip

    red_circle = tmp_path / "red_circle.png"
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).ellipse([80, 80, 320, 320], fill="#cc0000")
    image.save(red_circle)

    blue_square = tmp_path / "blue_square.png"
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle([80, 80, 320, 320], fill="#0033cc")
    image.save(blue_square)

    vectors, kept = embeddings_clip.embed_images([red_circle, blue_square])
    assert len(kept) == 2

    scores = vectors @ embeddings_clip.embed_query("a red circle")
    assert int(np.argmax(scores)) == 0, f"expected the red circle to win, got {scores}"


@requires_clip
@pytest.mark.model("clip")
def test_clip_configured_dim_matches_the_model() -> None:
    from app.ml import embeddings_clip

    assert embeddings_clip.embedding_dim() == settings.clip_embedding_dim


# --- OCR ---------------------------------------------------------------


@requires_ocr
def test_ocr_reads_text_from_a_rendered_page(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from app.ml import ocr

    path = tmp_path / "notice.png"
    image = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 100), "INVOICE TOTAL 4820", fill="black", font=font)
    image.save(path)

    result = ocr.get_engine().read_safe(path)

    assert not result.is_empty, "OCR produced nothing from clearly rendered text"
    assert "4820" in result.text.replace(",", "").replace(" ", "")


@requires_ocr
def test_ocr_on_a_blank_page_yields_empty(tmp_path: Path) -> None:
    from PIL import Image

    from app.ml import ocr

    path = tmp_path / "blank.png"
    Image.new("RGB", (800, 600), "white").save(path)

    assert ocr.get_engine().read_safe(path).is_empty


# --- captioning --------------------------------------------------------


@requires_caption
@pytest.mark.model("caption")
def test_captioner_describes_an_image(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    from app.ml import captioning

    path = tmp_path / "scene.png"
    image = Image.new("RGB", (640, 480), "#87ceeb")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 320, 640, 480], fill="#3a7d44")
    draw.ellipse([500, 40, 600, 140], fill="#ffd700")
    image.save(path)

    caption = captioning.get_captioner().caption(path)

    assert caption is not None, "captioner returned nothing for a valid image"
    assert len(caption.text) > 5
    assert caption.engine == settings.captioner


# --- regressions: silently-wrong output --------------------------------


@requires_caption
@pytest.mark.model("caption")
def test_moondream_tokenizer_matches_the_model_vocabulary() -> None:
    """Regression: the tokenizer beside the weights is the WRONG one.

    moondream2's own code fetches its tokenizer from a separate repo
    (`moondream/starmie-v1`). The `tokenizer.json` shipped with the weights is
    an older revision with a 50295-token vocabulary while the model generates
    over 51200. It loads without complaint and every caption comes out as
    fluent nonsense -- worse than a crash, because the index fills with noise
    that looks like real text.
    """
    model, _tokenizer = loaders.load_moondream()

    tokenizer_vocab = model.model.tokenizer.get_vocab_size()
    model_vocab = model.model.config.text.vocab_size

    assert tokenizer_vocab >= model_vocab, (
        f"tokenizer vocab {tokenizer_vocab} < model vocab {model_vocab}: "
        "captions would decode to nonsense"
    )


@requires_caption
@pytest.mark.model("caption")
def test_caption_is_readable_english_not_token_salad() -> None:
    """A vocabulary mismatch still produces *a* caption -- just a garbage one.

    So assert on the shape of the text, not merely that something came back.
    """
    import tempfile

    from PIL import Image, ImageDraw

    from app.ml import captioning

    path = Path(tempfile.mkdtemp()) / "scene.png"
    image = Image.new("RGB", (640, 480), "#87ceeb")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 320, 640, 480], fill="#3a7d44")
    draw.ellipse([500, 40, 600, 140], fill="#ffd700")
    image.save(path)

    caption = captioning.get_captioner().caption(path)
    assert caption is not None

    words = caption.text.split()
    assert len(words) >= 4, f"suspiciously short caption: {caption.text!r}"
    # Real English prose is mostly lowercase alphabetic words. Token salad from
    # a vocab mismatch is full of fragments like "sotti", "allocatewikiA".
    alphabetic = [w for w in words if w.strip(".,:;!?").isalpha()]
    assert len(alphabetic) / len(words) > 0.7, f"does not read as prose: {caption.text!r}"


@requires_clip
@pytest.mark.model("clip")
def test_clip_computed_buffers_are_materialised() -> None:
    """Regression: meta-device loading left `attn_mask` with no storage.

    Buffers registered `persistent=False` are absent from the checkpoint by
    design. Left on the meta device the text tower returned confident, wrong
    embeddings -- "a red circle" retrieved the blue square -- with nothing
    raised or logged.
    """
    model, _preprocess, _tokenizer = loaders.load_clip()

    still_meta = [name for name, buf in model.named_buffers() if buf is not None and buf.is_meta]
    assert not still_meta, f"buffers left uninitialised: {still_meta}"

    mask = getattr(model, "attn_mask", None)
    assert mask is not None, "CLIP lost its causal attention mask"
    # Causal additive mask: strictly-upper triangle -inf, everything else 0.
    import torch

    assert bool(torch.isinf(mask[0, 1])), "mask is not masking future positions"
    assert float(mask[1, 0]) == 0.0, "mask is blocking past positions"
