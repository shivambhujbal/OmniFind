"""Image captioning: moondream2 primary, BLIP-2 fallback, one interface.

Captions are what make an image findable by meaning without any text in it --
"the photo of a harbour at dusk" is a caption match, not an OCR match. They are
stored as ``CAPTION`` chunks so search ranks them alongside document text.

The two engines sit behind ``Captioner`` and are chosen by ``settings.captioner``.
Swapping them is a config change; nothing outside this module knows which one
ran. Neither constructs its own model -- both ask ``ml/loaders.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PIL import Image

from app.config import Captioner as CaptionerName
from app.config import settings
from app.ingestion.images import open_normalised
from app.logging_conf import get_logger
from app.ml import loaders

log = get_logger(__name__)

# Long enough for a useful sentence or two, short enough that a CPU run
# finishes this decade. Captions are for retrieval, not prose.
MAX_NEW_TOKENS = 64

PROMPT = "Describe this image."


@dataclass(frozen=True)
class Caption:
    text: str
    engine: str


class Captioner(ABC):
    """One interface, so the engine choice never leaks past this module."""

    name: str

    @abstractmethod
    def caption_image(self, image: Image.Image) -> str: ...

    def caption(self, path: Path) -> Caption | None:
        """Caption a file, returning None rather than raising on bad input.

        A single undecodable image must not fail the whole file's processing.
        A missing model is a different thing entirely and propagates -- see
        ``ocr.OcrEngine.read_safe`` for the same distinction.
        """
        # Decoding the image is the only step whose failure is genuinely about
        # *this* image. Everything after it -- a broken model install, a missing
        # vendored module, an out-of-memory forward pass -- applies to every
        # image equally and must propagate, or a broken install looks exactly
        # like "none of your pictures could be described" with nothing logged
        # above debug level. That is how a missing `lora.py` in the vendored
        # repo silently produced zero captions for a whole library.
        try:
            image = open_normalised(path)
        except (ValueError, OSError) as exc:
            log.warning("cannot decode %s: %s", path.name, exc)
            return None

        with image:
            text = self.caption_image(image)

        text = text.strip()
        if not text:
            return None
        return Caption(text=text, engine=self.name)


class MoondreamCaptioner(Captioner):
    name = "moondream2"

    def caption_image(self, image: Image.Image) -> str:
        model, _tokenizer = loaders.load_moondream()

        # moondream2 exposes a high-level `caption` API in recent revisions and
        # the older encode_image/answer_question pair before that. Support both
        # so a revision bump does not require a code change here.
        if hasattr(model, "caption"):
            result = model.caption(image, length="short")
            return str(result["caption"] if isinstance(result, dict) else result)

        encoded = model.encode_image(image)
        return str(model.answer_question(encoded, PROMPT, _tokenizer))


class Blip2Captioner(Captioner):
    name = "blip2"

    def caption_image(self, image: Image.Image) -> str:
        import torch

        model, processor = loaders.load_blip2()
        device = settings.ml_device

        inputs = processor(images=image, return_tensors="pt").to(
            device, loaders.resolve_dtype(device)
        )
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        return str(processor.batch_decode(tokens, skip_special_tokens=True)[0])


@cache
def get_captioner(name: CaptionerName | None = None) -> Captioner:
    """Return the configured captioner. Cached: these are stateless wrappers."""
    name = name or settings.captioner
    if name == "blip2":
        return Blip2Captioner()
    return MoondreamCaptioner()


def caption_images(paths: list[Path]) -> dict[Path, Caption]:
    """Caption a batch, skipping failures.

    Sequential rather than batched on purpose: these models generate
    autoregressively, so a batch does not amortise the way an encoder batch
    does, and one image at a time keeps peak memory flat.
    """
    captioner = get_captioner()
    results: dict[Path, Caption] = {}
    for path in paths:
        # Per image, not per batch: under low-memory mode this is the lock that
        # keeps a search query from loading CLIP on top of a resident
        # moondream2. Releasing between images lets a query in after ~390s
        # rather than after the whole document.
        with loaders.heavy_model_slot("captioner"):
            caption = captioner.caption(path)
        if caption is not None:
            results[path] = caption
    loaders.empty_device_cache()
    return results
