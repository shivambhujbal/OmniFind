"""File-search sidecar package.

Importing anything under ``app`` applies the offline ML environment first.
This is done at package level, not in ``main.py``, because ``huggingface_hub``
and ``transformers`` read ``HF_HOME`` / ``HF_HUB_OFFLINE`` once at *their*
import time -- setting them later has no effect. Any module that imports an ML
library therefore transitively guarantees the environment is already correct.

``Settings.apply_offline_env`` uses ``os.environ.setdefault``, so a caller that
legitimately needs the network (``scripts/download_models.py``, which runs on
the build machine only) can opt out by setting ``HF_HUB_OFFLINE=0`` before
importing this package.
"""

from app.config import settings as _settings

_settings.apply_offline_env()

__all__ = ["__version__"]
__version__ = _settings.app_version
