"""Logging helpers.

Nothing here imports Forge, so the module is safe to use from the unit tests.
"""

from __future__ import annotations

import hashlib
import logging
import sys

PREFIX = "[Krea2 RefBatchFix]"

logger = logging.getLogger("krea2_batchfix")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(f"{PREFIX} %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

logger.setLevel(logging.INFO)


def set_debug(enabled: bool) -> None:
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)


def banner(message: str) -> None:
    """Print to stdout even when logging is filtered, for install-time notices."""
    print(f"{PREFIX} {message}")


def short_image_hash(image, length: int = 10) -> str:
    """Short, stable digest of a PIL image.

    Only the digest is ever emitted; image contents are never logged.
    Returns ``"<none>"`` when no usable image is supplied.
    """
    if image is None:
        return "<none>"

    try:
        payload = image.tobytes()
    except Exception:
        try:
            payload = repr(image).encode("utf-8", "replace")
        except Exception:
            return "<unhashable>"

    return hashlib.sha1(payload).hexdigest()[:length]
