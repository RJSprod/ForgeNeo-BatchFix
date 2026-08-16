"""Interception and re-entry.

``Script.before_process()`` is not enough: Forge calls it once around
``process_images()``, while the Batch Count loop and the single ``p.init()``
that creates the Krea reference live further down inside
``process_images_inner()``. The extension therefore preserves the original
``modules.processing.process_images`` and installs an idempotent wrapper in
front of it.

``modules/img2img.py`` does ``from modules.processing import process_images``,
so rebinding only the ``modules.processing`` attribute would leave the actual
Img2Img caller pointing at the unpatched function; every module already holding
a direct reference to the original is rebound as well, by identity.

No Forge source file is ever written to.
"""

from __future__ import annotations

import functools
import sys

from . import VERSION
from .logs import banner, logger

WRAPPER_MARK = "__krea2_reference_batchfix__"

_original = None
_wrapper = None
_installed = False
_disabled_reason: str | None = None
_late_hooks_registered = False


def is_installed() -> bool:
    return _installed


def disabled_reason() -> str | None:
    return _disabled_reason


def original_process_images():
    return _original


def install() -> bool:
    """Install the wrapper. Safe (and cheap) to call repeatedly."""
    global _original, _wrapper, _installed, _disabled_reason

    if _installed:
        _rebind_direct_importers()
        return True

    from . import compat

    report = compat.check()
    if not report.ok:
        _disabled_reason = report.summary
        banner("DISABLED — this Forge build does not match the tested API surface:")
        for problem in report.problems:
            banner(f"  - {problem}")
        banner(f"  tested against Forge {compat.TESTED_FORGE_VERSION} @ {compat.TESTED_FORGE_COMMIT}")
        banner("  Krea 2 Img2Img batches will run unpatched; no unsafe patch was applied.")
        return False

    try:
        import modules.processing as processing
    except Exception as exc:
        _disabled_reason = f"cannot import Forge processing modules ({exc!r})"
        banner(f"DISABLED — {_disabled_reason}")
        return False

    current = processing.process_images

    if getattr(current, WRAPPER_MARK, False):
        # A previous install survived a script reload; adopt it.
        _wrapper = current
        _installed = True
        _rebind_direct_importers()
        logger.debug("wrapper already present; adopted it")
        return True

    _original = current
    _wrapper = _make_wrapper(_original)
    processing.process_images = _wrapper
    _installed = True
    _disabled_reason = None

    rebound = _rebind_direct_importers()
    banner(f"active (v{VERSION}) — {report.summary}; redirected {rebound} direct import(s)")
    return True


def uninstall() -> None:
    """Restore Forge's original function. Used by the tests."""
    global _original, _wrapper, _installed

    if not _installed:
        return

    try:
        import modules.processing as processing

        if processing.process_images is _wrapper:
            processing.process_images = _original
    except Exception:
        pass

    for module in _modules_holding(_wrapper):
        module.__dict__["process_images"] = _original

    _installed = False
    _wrapper = None
    _original = None


def _make_wrapper(original):
    from . import detect, fanout, settings

    @functools.wraps(original)
    def process_images(p):
        try:
            settings.sync_debug_level()
            decision, reason = detect.should_fan_out(
                p,
                enabled=bool(settings.get(settings.ENABLED)),
                include_inpaint=bool(settings.get(settings.INCLUDE_INPAINT)),
            )
        except Exception:
            logger.warning("activation check failed; running the request unpatched", exc_info=True)
            return original(p)

        if not decision:
            logger.debug("pass-through (%s)", reason)
            return original(p)

        try:
            return fanout.run(
                p,
                original,
                force_recond=bool(settings.get(settings.FORCE_RECOND)),
                add_infotext=bool(settings.get(settings.ADD_INFOTEXT)),
            )
        except Exception:
            logger.error("fan-out failed; no images were produced by the isolated path", exc_info=True)
            raise

    setattr(process_images, WRAPPER_MARK, True)
    return process_images


def _modules_holding(target):
    """Modules whose top-level ``process_images`` is exactly ``target``."""
    found = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        if namespace.get("process_images") is target:
            found.append(module)
    return found


def _rebind_direct_importers() -> int:
    """Point every direct importer of the original function at the wrapper."""
    if _original is None or _wrapper is None:
        return 0

    count = 0
    for module in _modules_holding(_original):
        module.__dict__["process_images"] = _wrapper
        logger.debug("redirected %s.process_images", getattr(module, "__name__", "<module>"))
        count += 1

    return count


def verify_img2img_binding() -> bool:
    """Confirm the real Img2Img caller reaches the wrapper.

    ``modules/img2img.py`` is imported from inside ``create_ui()``. Loaded after
    the patch, its ``from modules.processing import process_images`` binds the
    wrapper on its own; loaded before it, ``_rebind_direct_importers`` catches
    it. This checks that one of the two actually happened.
    """
    if not _installed:
        return False

    module = sys.modules.get("modules.img2img")
    if module is None:
        logger.debug("modules.img2img is not loaded yet")
        return False

    if module.__dict__.get("process_images") is _wrapper:
        logger.debug("modules.img2img reaches the wrapper")
        return True

    banner("WARNING — modules.img2img is not reaching the wrapper; Krea 2 Img2Img batches will run unpatched.")
    return False


def install_late_hooks() -> None:
    """Re-run the rebind once the UI is up, in case a module imported late."""
    global _late_hooks_registered

    if _late_hooks_registered:
        return

    try:
        from modules import script_callbacks
    except Exception:
        return

    def _on_before_ui():
        if _installed:
            _rebind_direct_importers()

    def _on_app_started(*_args):
        if _installed:
            _rebind_direct_importers()
            verify_img2img_binding()

    try:
        script_callbacks.on_before_ui(_on_before_ui)
        script_callbacks.on_app_started(_on_app_started)
        _late_hooks_registered = True
    except Exception:
        logger.debug("could not register late rebind hooks", exc_info=True)
