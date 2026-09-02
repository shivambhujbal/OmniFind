"""FastAPI sidecar entrypoint.

Launched by the shell (Tauri in production, ``scripts/run_backend.ps1`` during
development) with a loopback port. Binds 127.0.0.1 only -- see
``Settings.validate_host``.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import files, health, jobs, models, search
from app.config import settings
from app.db import bootstrap
from app.db.session import session_scope
from app.logging_conf import configure_logging, get_logger
from app.ml import maintenance
from app.tasks import index, ingest, process
from app.tasks.queue import task_queue
from app.vectors import store as vector_store

log = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    log.info("sidecar starting: version=%s device=%s", settings.app_version, settings.ml_device)
    log.info("data dir: %s", settings.data_dir)

    bootstrap.upgrade_to_head()
    task_queue.start()
    # A crash or a forced quit leaves files half-processed; pick them back up
    # rather than stranding them mid-pipeline forever.
    # Re-apply the current text-quality rules to chunks classified under
    # older ones, and drop anything that no longer qualifies from the index.
    # Text analysis only -- no models, no file reads.
    with session_scope() as session:
        maintenance.reclassify_chunks(session)
        maintenance.prune_image_vectors(session)

    ingest.requeue_unfinished()
    process.requeue_unprocessed()
    process.requeue_uncaptioned()
    index.requeue_unindexed()

    yield

    log.info("sidecar shutting down")
    task_queue.stop()
    # Release the Qdrant storage lock so the next start can open it.
    vector_store.close_client()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local File Search sidecar",
        version=settings.app_version,
        lifespan=lifespan,
        # No public docs surface needed, but they cost nothing locally and are
        # genuinely useful while building.
        docs_url="/docs",
    )

    # The Streamlit dev client runs on a different loopback port, so the
    # browser treats it as a cross origin. Loopback only -- never a wildcard.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(files.router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(search.router)
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local File Search sidecar")
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Loopback port supplied by the shell process.",
    )
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args()

    settings.port = args.port
    settings.log_level = args.log_level
    configure_logging()
    settings.ensure_dirs()

    uvicorn.run(
        app,
        host=settings.host,  # validated loopback-only
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
