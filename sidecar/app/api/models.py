"""Model provisioning status.

Lets the UI tell a user "captioning is unavailable because moondream2 is not
downloaded" up front, instead of after they wait through an ingest that then
fails. Deliberately does *not* load anything -- it inspects the filesystem, so
polling it is cheap and does not pull 4 GB into memory.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.ml import loaders

router = APIRouter(prefix="/models", tags=["models"])


class ModelStatus(BaseModel):
    path: str
    available: bool
    missing: list[str]


class ModelsResponse(BaseModel):
    device: str
    ocr_engine: str
    captioner: str
    models: dict[str, ModelStatus]
    ready_for_processing: bool
    # When false, captions and image vectors are skipped entirely. OCR still
    # runs, so scanned documents remain searchable.
    image_understanding: bool


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    report = loaders.available_models()
    models = {name: ModelStatus(**detail) for name, detail in report.items()}

    # "Ready" means the engines actually selected by config are present, not
    # that every model in the report is -- BLIP-2 missing is fine when the
    # captioner is moondream2.
    ocr_ready = models[settings.ocr_engine].available
    caption_ready = models[
        "moondream2" if settings.captioner == "moondream2" else "blip2"
    ].available

    # With image understanding off, the captioner's absence is irrelevant --
    # reporting "not ready" because of a model the app will never load would be
    # a false alarm.
    ready = ocr_ready and (caption_ready or not settings.enable_image_understanding)

    return ModelsResponse(
        device=settings.ml_device,
        ocr_engine=settings.ocr_engine,
        captioner=settings.captioner,
        models=models,
        ready_for_processing=ready,
        image_understanding=settings.enable_image_understanding,
    )
