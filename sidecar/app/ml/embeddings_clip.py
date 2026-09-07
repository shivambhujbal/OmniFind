"""Image and text embeddings with OpenCLIP ViT-H/14.

CLIP puts images and text in one shared space, which is what lets "a red bicycle
against a brick wall" retrieve a photo that has no text in it at all. This is a
*separate* space from the bge text embeddings -- the two are not comparable, so
phase 4 keeps them in different Qdrant collections and merges at ranking time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.config import settings
from app.ingestion.images import open_normalised
from app.logging_conf import get_logger
from app.ml import loaders

log = get_logger(__name__)

# ViT-H/14 is heavy; a small batch keeps peak memory predictable on CPU.
BATCH_SIZE = 8


def _normalise_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalise so cosine similarity is a dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against a zero vector, which would otherwise produce NaNs that
    # propagate silently into the index.
    norms[norms == 0] = 1.0
    normalised: NDArray[np.float32] = (vectors / norms).astype(np.float32)
    return normalised


def embed_images(paths: list[Path]) -> tuple[NDArray[np.float32], list[Path]]:
    """Embed images, skipping any that cannot be decoded.

    Returns the vectors *and* the paths that actually produced them: an image
    that fails to open must not silently shift every later vector onto the wrong
    row, which is what a plain list-in/list-out signature would allow.
    """
    if not paths:
        return np.zeros((0, settings.clip_embedding_dim), dtype=np.float32), []

    import torch

    model, preprocess, _ = loaders.load_clip()
    device = settings.ml_device

    tensors: list[torch.Tensor] = []
    kept: list[Path] = []
    for path in paths:
        try:
            with open_normalised(path) as image:
                tensors.append(preprocess(image))
            kept.append(path)
        except (ValueError, OSError) as exc:
            log.warning("skipping unreadable image %s: %s", path, exc)

    if not tensors:
        return np.zeros((0, settings.clip_embedding_dim), dtype=np.float32), []

    outputs: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[start : start + BATCH_SIZE]).to(device)
            features = model.encode_image(batch)
            outputs.append(features.cpu().numpy().astype(np.float32))

    loaders.empty_device_cache()
    return _normalise_rows(np.concatenate(outputs, axis=0)), kept


def embed_image(path: Path) -> NDArray[np.float32] | None:
    vectors, kept = embed_images([path])
    return vectors[0] if kept else None


def embed_pil_image(image: Image.Image) -> NDArray[np.float32]:
    """Embed an already-open image, for callers that have one in hand."""
    import torch

    model, preprocess, _ = loaders.load_clip()
    with torch.no_grad():
        batch = preprocess(image).unsqueeze(0).to(settings.ml_device)
        features = model.encode_image(batch)
    vector: NDArray[np.float32] = features.cpu().numpy().astype(np.float32)
    row: NDArray[np.float32] = _normalise_rows(vector)[0]
    return row


def embed_text(texts: list[str]) -> NDArray[np.float32]:
    """Embed text into CLIP's space, for text-to-image search.

    Note this is *not* interchangeable with ``embeddings_text.embed_query``:
    different model, different space. This one is only for querying the image
    collection.
    """
    if not texts:
        return np.zeros((0, settings.clip_embedding_dim), dtype=np.float32)

    import torch

    model, _, tokenizer = loaders.load_clip()
    with torch.no_grad():
        tokens = tokenizer(texts).to(settings.ml_device)
        features = model.encode_text(tokens)

    matrix: NDArray[np.float32] = features.cpu().numpy().astype(np.float32)
    return _normalise_rows(matrix)


# CLIP is trained on image-*caption* pairs -- "a photo of a person standing in
# a park", never the bare noun "person". A one-word query is therefore out of
# distribution and scores lower against *every* image, which a fixed relevance
# floor then removes wholesale. Measured on this library (794 images, floor
# 0.25), searching "human" returned nothing at all while a full description
# like "black guy in yellow tshirt" worked perfectly.
#
# Encoding the query under caption templates and averaging repairs it. Results
# above the floor:
#
#   query      raw   "a photo of {}."   this ensemble
#   person       1         46                27
#   human        0         21                15
#   people       0          7                 4
#   vehicle      4          1                 6
#   car          8         10                13
#   dog          3          6                 7
#   gibberish    1          7                 4
#
# A single template finds more people, but makes "vehicle" *worse* and pulls in
# twice the noise. The ensemble is the only form that improves every real query
# -- and it keeps the raw query in the mix, so a phrase that already reads like
# a caption is not distorted by being wrapped in another one.
_QUERY_TEMPLATES = ("{}", "a photo of {}.", "a photo of a {}.")


def embed_query(query: str) -> NDArray[np.float32]:
    """Embed a search query, averaged over caption templates.

    Only the *query* side is templated. Stored image vectors are untouched, so
    this changes retrieval immediately with no re-indexing.
    """
    rows = embed_text([template.format(query) for template in _QUERY_TEMPLATES])
    mean = rows.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        # Degenerate only if the variants cancel exactly, which normalised
        # embeddings of near-identical text will not do -- but a zero vector
        # would score 0 against everything and silently return nothing.
        first: NDArray[np.float32] = rows[0]
        return first
    averaged: NDArray[np.float32] = (mean / norm).astype(np.float32)
    return averaged


def embedding_dim() -> int:
    """Actual output width of the loaded CLIP model."""
    try:
        config_path = settings.models_dir / "openclip-vit-h14" / "open_clip_config.json"
        if config_path.exists():
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data["model_cfg"]["embed_dim"])
    except Exception:
        pass
    model, _, _ = loaders.load_clip()
    return int(model.visual.output_dim)
