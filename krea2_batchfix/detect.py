"""Activation conditions for the fan-out.

Detection is deliberately conservative: it looks at the loaded diffusion engine
and at Forge's own Krea flag rather than guessing from checkpoint filenames, and
every unknown returns "not active" so non-Krea Img2Img and txt2img pass straight
through to stock Forge.
"""

from __future__ import annotations

from . import UNIT_SENTINEL
from .logs import logger


def _dynamic_args():
    try:
        from backend.args import dynamic_args

        return dynamic_args
    except Exception:
        return None


def _sd_model():
    try:
        from modules import shared

        return shared.sd_model
    except Exception:
        return None


def is_krea_engine(sd_model=None) -> bool:
    """True when the loaded engine is the native Krea 2 diffusion engine.

    Primary signal is the engine class itself; ``dynamic_args.krea2`` (set by
    ``backend/loader.py`` from the HuggingFace repo name) is accepted as a
    secondary signal so the check still works if the class is ever renamed.
    """
    if sd_model is None:
        sd_model = _sd_model()

    if sd_model is None:
        return False

    try:
        from backend.diffusion_engine.krea import Krea2

        if isinstance(sd_model, Krea2):
            return True
    except Exception:
        pass

    if type(sd_model).__name__ == "Krea2":
        return True

    dynamic_args = _dynamic_args()
    if dynamic_args is not None and bool(getattr(dynamic_args, "krea2", False)):
        # Only trust the flag if the engine also carries the reference state we
        # depend on; a stale flag must never enable the patch on its own.
        return hasattr(sd_model, "ini_latent") and hasattr(sd_model, "ref_latents")

    return False


def reference_mode_enabled() -> bool:
    """True when the Krea 2 reference (Edit) path is switched on in Settings."""
    try:
        from modules.shared import opts

        return bool(getattr(opts, "krea2_do_reference", False))
    except Exception:
        return False


def is_krea_reference_active(sd_model=None) -> bool:
    """Native Krea 2 engine loaded *and* reference-image mode enabled."""
    if sd_model is None:
        sd_model = _sd_model()

    if not is_krea_engine(sd_model):
        return False

    if not hasattr(sd_model, "ini_latent"):
        logger.debug("Krea engine detected but it has no 'ini_latent'; passing through")
        return False

    return reference_mode_enabled()


def primary_reference(p):
    """The single primary Img2Img reference of the request, or ``None``.

    Forge hands Img2Img either ``[image]`` (single upload) or ``[image] * batch_size``
    (outer batch-upload loop), so the primary reference is always element 0. If a
    caller supplied genuinely *different* images in one request — only reachable
    through the API, where Forge treats them as one batch rather than as separate
    reference requests — this returns ``None`` so the request passes through
    unchanged rather than being silently reinterpreted.
    """
    init_images = getattr(p, "init_images", None)
    if not init_images:
        return None

    first = init_images[0]
    if first is None:
        return None

    for other in init_images[1:]:
        if other is first:
            continue
        if not _same_image(first, other):
            logger.warning("Img2Img received %d different init images in one request; passing through to stock Forge (V1 fans out one primary reference per request)", len(init_images))
            return None

    return first


def _same_image(a, b) -> bool:
    try:
        return a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()
    except Exception:
        return False


def is_inpaint_request(p) -> bool:
    """True for inpaint / sketch-with-mask requests, which V1 does not claim."""
    return getattr(p, "image_mask", None) is not None or getattr(p, "latent_mask", None) is not None


def is_unit_job(p) -> bool:
    return bool(getattr(p, UNIT_SENTINEL, False))


def should_fan_out(p, *, enabled: bool, include_inpaint: bool) -> tuple[bool, str]:
    """Evaluate every activation condition.

    Returns ``(decision, reason)``; the reason is only used for debug logging.
    """
    if not enabled:
        return False, "disabled in settings"

    if is_unit_job(p):
        return False, "already an isolated unit job"

    try:
        from modules.processing import StableDiffusionProcessingImg2Img
    except Exception:
        return False, "modules.processing unavailable"

    if not isinstance(p, StableDiffusionProcessingImg2Img):
        return False, "not an Img2Img request"

    n_iter = int(getattr(p, "n_iter", 1) or 1)
    batch_size = int(getattr(p, "batch_size", 1) or 1)
    if n_iter <= 1 and batch_size <= 1:
        return False, "single output requested"

    if not is_krea_reference_active():
        return False, "Krea 2 reference mode not active"

    if is_inpaint_request(p) and not include_inpaint:
        return False, "inpaint/sketch request (not claimed by V1)"

    if primary_reference(p) is None:
        return False, "no single primary Img2Img reference"

    if isinstance(getattr(p, "seed", -1), list) or isinstance(getattr(p, "subseed", -1), list):
        return False, "caller supplied an explicit seed list"

    if getattr(p, "scripts", None) is not None and getattr(p, "script_args", None) is None:
        return False, "script arguments are unavailable"

    return True, "krea2 reference batch"
