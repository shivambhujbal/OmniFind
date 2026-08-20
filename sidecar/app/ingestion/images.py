"""Image handling: validation, EXIF-correct orientation, thumbnails.

Standalone images have no text to extract in phase 2 -- their searchable content
comes from OCR and captioning in phase 3. What this module does now is make sure
the image is readable, record its dimensions, and produce a thumbnail for the
results view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.logging_conf import get_logger
from app.storage import files as storage

log = get_logger(__name__)

THUMBNAIL_SIZE = (400, 400)

# Pillow refuses very large images as a decompression-bomb guard. A 200 MP
# ceiling covers legitimate scans and panoramas while keeping the guard useful.
Image.MAX_IMAGE_PIXELS = 200_000_000


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str
    thumbnail_path: Path


def open_normalised(path: Path) -> Image.Image:
    """Open an image with EXIF rotation applied and a predictable mode.

    Phone photos are stored unrotated with an EXIF orientation tag. Skipping
    ``exif_transpose`` means OCR reads sideways text and captions describe a
    rotated scene -- both fail quietly, which is the worst way to fail.
    """
    try:
        opened = Image.open(path)
    except UnidentifiedImageError as exc:
        raise ValueError("This image file could not be decoded.") from exc
    except OSError as exc:
        raise ValueError(f"This image file could not be read: {exc}") from exc

    # exif_transpose returns None only for a None input, which cannot happen here.
    image: Image.Image = ImageOps.exif_transpose(opened) or opened
    if image.mode not in ("RGB", "L"):
        # CMYK, palette and RGBA all need converting before the vision models
        # see them; do it once here rather than in each consumer.
        image = image.convert("RGB")
    return image


def make_thumbnail(source: Path, sha256: str, name: str = "thumb.jpg") -> ImageInfo:
    """Write a thumbnail into the derived directory and report the source size."""
    with open_normalised(source) as image:
        width, height = image.size
        mode = image.mode

        thumb = image.copy()
        thumb.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")

        destination = storage.derived_path(sha256, "thumbs", name)
        thumb.save(destination, format="JPEG", quality=85, optimize=True)

    return ImageInfo(width=width, height=height, mode=mode, thumbnail_path=destination)


def extract(path: Path, sha256: str) -> ImageInfo:
    """Validate a standalone uploaded image and build its thumbnail."""
    info = make_thumbnail(path, sha256)
    log.info("image %s: %dx%d %s", path.name, info.width, info.height, info.mode)
    return info
