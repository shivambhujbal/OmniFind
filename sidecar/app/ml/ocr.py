"""OCR: PaddleOCR primary, Tesseract CPU fallback, one interface.

OCR is what makes a scanned PDF searchable at all -- without it a scan is a
picture of words and matches nothing. Engines sit behind ``OcrEngine`` and are
chosen by ``settings.ocr_engine``.

PaddleOCR is markedly more accurate on photographed and skewed text; Tesseract
needs no CUDA and no extra wheel. Neither constructs its own model -- both ask
``ml/loaders.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from app.config import OcrEngine as OcrEngineName
from app.config import settings
from app.logging_conf import get_logger
from app.ml import loaders

log = get_logger(__name__)

# Below this the "text" is almost always noise from compression artefacts or
# page furniture, and indexing it only pollutes search results.
MIN_CONFIDENCE = 0.5
MIN_TEXT_CHARS = 3


@dataclass
class OcrResult:
    text: str
    engine: str
    mean_confidence: float | None = None
    line_count: int = 0
    lines: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < MIN_TEXT_CHARS


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def read(self, path: Path) -> OcrResult: ...

    def read_safe(self, path: Path) -> OcrResult:
        """Read, converting a per-image failure into an empty result.

        One unreadable page must not fail an entire document: a 40-page scan
        where page 12 is corrupt should still yield 39 searchable pages.

        A missing *model* is deliberately NOT swallowed. That is not a bad
        page, it is "OCR is not installed" -- a condition that applies to every
        image equally and that the caller must be able to report rather than
        silently record as "this scan contains no text".
        """
        try:
            return self.read(path)
        except loaders.ModelNotAvailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - engines raise many unrelated types
            log.warning("OCR failed on %s: %s", path.name, exc)
            return OcrResult(text="", engine=self.name)


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def read(self, path: Path) -> OcrResult:
        engine = loaders.load_paddleocr()
        raw = engine.predict(str(path))

        lines: list[str] = []
        confidences: list[float] = []
        # PaddleOCR 3.x returns a list of per-image result dicts.
        for page in raw or []:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            for text, score in zip(texts, scores, strict=False):
                if score >= MIN_CONFIDENCE and text.strip():
                    lines.append(text.strip())
                    confidences.append(float(score))

        return OcrResult(
            text="\n".join(lines),
            engine=self.name,
            mean_confidence=sum(confidences) / len(confidences) if confidences else None,
            line_count=len(lines),
            lines=lines,
        )


class TesseractEngine(OcrEngine):
    name = "tesseract"

    def read(self, path: Path) -> OcrResult:
        pytesseract = loaders.load_tesseract()
        from app.ingestion.images import open_normalised

        with open_normalised(path) as image:
            # image_to_data rather than image_to_string: the per-word confidence
            # is what lets low-quality noise be dropped instead of indexed.
            data = pytesseract.image_to_data(
                image,
                lang=_tesseract_lang(settings.ocr_languages),
                output_type=pytesseract.Output.DICT,
            )

        lines: list[str] = []
        confidences: list[float] = []
        current: list[str] = []
        last_line_key: tuple[int, int, int] | None = None

        for index, word in enumerate(data["text"]):
            word = word.strip()
            confidence = float(data["conf"][index])
            # Tesseract marks non-word boxes with -1.
            if not word or confidence < 0:
                continue

            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            if last_line_key is not None and key != last_line_key and current:
                lines.append(" ".join(current))
                current = []
            last_line_key = key

            if confidence / 100.0 >= MIN_CONFIDENCE:
                current.append(word)
                confidences.append(confidence / 100.0)

        if current:
            lines.append(" ".join(current))

        lines = [line for line in lines if line.strip()]
        return OcrResult(
            text="\n".join(lines),
            engine=self.name,
            mean_confidence=sum(confidences) / len(confidences) if confidences else None,
            line_count=len(lines),
            lines=lines,
        )


def _tesseract_lang(languages: str) -> str:
    """Map our language setting onto Tesseract's traineddata names."""
    mapping = {"en": "eng", "ch": "chi_sim", "fr": "fra", "de": "deu", "es": "spa"}
    return "+".join(mapping.get(code.strip(), code.strip()) for code in languages.split(","))


@cache
def get_engine(name: OcrEngineName | None = None) -> OcrEngine:
    name = name or settings.ocr_engine
    if name == "tesseract":
        return TesseractEngine()
    return PaddleOcrEngine()


def read_images(paths: list[Path]) -> dict[Path, OcrResult]:
    """OCR a batch, returning only the images that produced usable text."""
    engine = get_engine()
    results: dict[Path, OcrResult] = {}
    for path in paths:
        with loaders.heavy_model_slot("ocr"):
            result = engine.read_safe(path)
        if not result.is_empty:
            results[path] = result
    loaders.empty_device_cache()
    return results
