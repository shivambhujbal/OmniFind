"""Phase 3 contracts that hold whether or not the weights are downloaded.

These run in CI and on a machine with an empty models directory. Tests that
need the real 9 GB of weights live in ``test_ml_inference.py`` and skip when the
files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, settings
from app.ml import captioning, loaders, ocr

# --- loader discipline -------------------------------------------------


def test_loaders_default_to_the_configured_device() -> None:
    """Every loader takes `device` with settings.ml_device as the default.

    This is the seam the CPU-fallback plan depends on: if a loader hardcodes a
    device or drops the parameter, changing one config value stops working.
    """
    import inspect

    device_taking = (
        loaders.load_text_encoder,
        loaders.load_clip,
        loaders.load_moondream,
        loaders.load_blip2,
        loaders.load_paddleocr,
    )
    for loader in device_taking:
        # lru_cache wraps the function; inspect the original.
        signature = inspect.signature(loader.__wrapped__)
        assert "device" in signature.parameters, f"{loader.__name__} has no device parameter"
        assert signature.parameters["device"].default is None, (
            f"{loader.__name__} should default device to None and resolve "
            "settings.ml_device at call time, so a config change takes effect"
        )

    # Tesseract is CPU-only by nature; a device parameter would be a lie.
    assert "device" not in inspect.signature(loaders.load_tesseract.__wrapped__).parameters


def test_dtype_follows_the_device() -> None:
    """float16 on CPU is slower than float32, not faster -- see loaders.resolve_dtype."""
    import torch

    assert loaders.resolve_dtype("cuda") is torch.float16
    assert loaders.resolve_dtype("cpu") is torch.float32


def test_empty_device_cache_is_a_noop_on_cpu() -> None:
    """Must not raise when there is no CUDA context to empty."""
    loaders.empty_device_cache("cpu")


def test_missing_model_raises_an_actionable_error(tmp_path: Path) -> None:
    """The error names the path and the command that fixes it.

    With networking disabled, the alternative is a library hanging on a hub
    request or failing with a stack trace about a missing config key.
    """
    with pytest.raises(loaders.ModelNotAvailableError) as exc:
        loaders._require_dir("Some model", tmp_path / "not-there")

    message = str(exc.value)
    assert "not-there" in message
    assert "download_models.py" in message


def test_missing_required_file_is_named(tmp_path: Path) -> None:
    (tmp_path / "present").mkdir()
    with pytest.raises(loaders.ModelNotAvailableError, match="tokenizer.json"):
        loaders._require_dir("Some model", tmp_path / "present", "tokenizer.json")


def test_available_models_does_not_load_anything() -> None:
    """Status polling must stay cheap -- no 4 GB read to answer a UI question."""
    loaders.unload_all()
    report = loaders.available_models()

    assert {"text_embeddings", "clip", "moondream2", "paddleocr"} <= set(report)
    for name, detail in report.items():
        assert "available" in detail, name
        assert "path" in detail, name

    # Nothing should have been constructed by asking.
    assert loaders.load_text_encoder.cache_info().currsize == 0
    assert loaders.load_clip.cache_info().currsize == 0


def test_empty_model_directory_does_not_count_as_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing-but-empty directory must report as NOT available.

    `ensure_dirs` and the download script both create these directories
    eagerly, so a plain `is_dir()` check reported PaddleOCR as installed when
    nothing had been fetched -- and the UI told the user OCR was ready.
    """
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "models" / "paddlex" / "official_models").mkdir(parents=True)

    report = loaders.available_models()

    assert report["paddleocr"]["available"] is False
    assert report["paddleocr"]["missing"], "an empty directory must say what is missing"


# --- engine selection --------------------------------------------------


def test_ocr_engine_follows_config() -> None:
    ocr.get_engine.cache_clear()
    assert ocr.get_engine("tesseract").name == "tesseract"
    assert ocr.get_engine("paddleocr").name == "paddleocr"


def test_captioner_follows_config() -> None:
    captioning.get_captioner.cache_clear()
    assert captioning.get_captioner("moondream2").name == "moondream2"
    assert captioning.get_captioner("blip2").name == "blip2"


def test_both_engines_share_one_interface() -> None:
    """The point of the abstraction: callers never branch on engine."""
    for engine in (ocr.TesseractEngine(), ocr.PaddleOcrEngine()):
        assert isinstance(engine, ocr.OcrEngine)
        assert hasattr(engine, "read_safe")
    for captioner in (captioning.MoondreamCaptioner(), captioning.Blip2Captioner()):
        assert isinstance(captioner, captioning.Captioner)
        assert hasattr(captioner, "caption")


def test_ocr_failure_yields_empty_not_an_exception() -> None:
    """One corrupt page must not fail a forty-page scan."""

    class ExplodingEngine(ocr.OcrEngine):
        name = "exploding"

        def read(self, path: Path) -> ocr.OcrResult:
            raise OSError("simulated decode failure on one page")

    result = ExplodingEngine().read_safe(Path("page-12.png"))

    assert result.is_empty
    assert result.text == ""


def test_missing_ocr_model_is_not_swallowed_as_empty_text() -> None:
    """ "OCR is not installed" and "this page has no text" are different facts.

    Swallowing the first as the second would silently record a scanned document
    as containing nothing, with no indication anything was wrong.
    """

    class UninstalledEngine(ocr.OcrEngine):
        name = "uninstalled"

        def read(self, path: Path) -> ocr.OcrResult:
            raise loaders.ModelNotAvailableError("Engine", Path("nowhere"))

    with pytest.raises(loaders.ModelNotAvailableError):
        UninstalledEngine().read_safe(Path("page.png"))


def test_caption_failure_yields_none(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.png"
    assert captioning.MoondreamCaptioner().caption(missing) is None


def test_ocr_result_treats_whitespace_as_empty() -> None:
    assert ocr.OcrResult(text="   \n ", engine="x").is_empty
    assert not ocr.OcrResult(text="Total due 4,820.00", engine="x").is_empty


def test_tesseract_language_codes_are_mapped() -> None:
    """Our config uses 'en'; Tesseract wants 'eng'."""
    assert ocr._tesseract_lang("en") == "eng"
    assert ocr._tesseract_lang("en,fr") == "eng+fra"


# --- config seam -------------------------------------------------------


def test_device_is_overridable_by_one_setting() -> None:
    """The whole CPU-fallback plan rests on this."""
    assert Settings(ml_device="cpu").ml_device == "cpu"
    assert Settings(ml_device="cuda").ml_device == "cuda"

    with pytest.raises(ValueError):
        Settings(ml_device="tpu")  # type: ignore[arg-type]


def test_query_prefix_is_configured_not_hardcoded() -> None:
    """bge is asymmetric: queries get an instruction prefix, documents do not."""
    assert settings.text_query_prefix
    assert "search" in settings.text_query_prefix.lower()
