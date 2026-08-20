"""Text embeddings with bge-large-en-v1.5.

Asymmetric retrieval: a short question and a long passage are embedded
differently. bge is trained with an instruction prefix on the *query* side only
(``settings.text_query_prefix``); documents go in bare. Applying the prefix to
both, or to neither, measurably hurts recall -- which is why queries and
documents have separate entry points here rather than one ``embed()``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.config import settings
from app.logging_conf import get_logger
from app.ml import loaders

log = get_logger(__name__)

# Batch size is a memory/throughput trade-off, not a correctness one. 16 keeps
# peak usage modest on a laptop; a CUDA box could go much higher.
BATCH_SIZE = 16


def embed_documents(texts: list[str]) -> NDArray[np.float32]:
    """Embed passages for storage. No query prefix -- see the module docstring.

    Returns L2-normalised vectors, so cosine similarity is a dot product and
    Qdrant's COSINE distance and DOT distance agree.
    """
    if not texts:
        return np.zeros((0, settings.text_embedding_dim), dtype=np.float32)

    model = loaders.load_text_encoder()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> NDArray[np.float32]:
    """Embed a search query, with the instruction prefix bge expects."""
    model = loaders.load_text_encoder()
    vector = model.encode(
        [settings.text_query_prefix + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    matrix: NDArray[np.float32] = np.asarray(vector, dtype=np.float32)
    row: NDArray[np.float32] = matrix[0]
    return row


def embedding_dim() -> int:
    """Actual dimensionality of the loaded model.

    Read from the model rather than trusting the config value: a mismatch
    between this and the Qdrant collection is a silent corruption, so phase 4
    checks them against each other at startup.
    """
    dim = loaders.load_text_encoder().get_sentence_embedding_dimension()
    if dim is None:
        # Happens when a model directory has no pooling module -- the folder is
        # a plain transformer, not a sentence-transformers bundle. Better to say
        # so than to crash on int(None) three frames away.
        raise RuntimeError(
            f"{settings.text_embedding_model} reports no embedding dimension; "
            "the model directory is probably missing its pooling module "
            "(1_Pooling/ and modules.json). Re-run scripts/download_models.py."
        )
    return int(dim)
