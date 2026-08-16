"""Fail-closed compatibility probing against the running Forge Neo.

The extension refuses to install its wrapper unless every Forge API it depends
on is present with the expected shape. A missing API produces a loud, explicit
error instead of a silently-unsafe patch that would emit unreferenced images
(Definition of Done #7).

Tested against Forge Neo "neo" @ 6009ffff99b5d5b4312dc8a8f6476ec0a69b37b1
(``modules_forge.forge_version`` -> neo-2.28).
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass

TESTED_FORGE_VERSION = "neo-2.28"
TESTED_FORGE_COMMIT = "6009ffff99b5d5b4312dc8a8f6476ec0a69b37b1"


@dataclass
class CompatReport:
    ok: bool
    problems: list[str]
    forge_version: str = "unknown"

    @property
    def summary(self) -> str:
        if self.ok:
            return f"compatible (Forge {self.forge_version}, tested against {TESTED_FORGE_VERSION})"
        return "; ".join(self.problems)


def _check_attrs(obj, names, label, problems):
    for name in names:
        if not hasattr(obj, name):
            problems.append(f"{label} is missing '{name}'")


def _forge_version() -> str:
    try:
        from modules_forge.forge_version import release, version

        return f"{version}-{release}"
    except Exception:
        return "unknown"


def check() -> CompatReport:
    """Probe every Forge Neo API this extension relies on."""
    problems: list[str] = []

    try:
        import modules.processing as processing
    except Exception as exc:  # pragma: no cover - only on a broken install
        return CompatReport(False, [f"cannot import modules.processing ({exc!r})"])

    # --- the function we wrap -------------------------------------------------
    process_images = getattr(processing, "process_images", None)
    if not callable(process_images):
        problems.append("modules.processing.process_images is missing")
    else:
        try:
            params = list(inspect.signature(process_images).parameters.values())
            positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(positional) != 1:
                problems.append(f"modules.processing.process_images has an unexpected signature ({len(positional)} positional parameters, expected 1)")
        except (TypeError, ValueError):
            problems.append("modules.processing.process_images signature could not be inspected")

    # --- helpers used while fanning out ---------------------------------------
    for name in ("Processed", "StableDiffusionProcessingImg2Img", "get_fixed_seed", "create_infotext"):
        if not hasattr(processing, name):
            problems.append(f"modules.processing.{name} is missing")

    img2img_cls = getattr(processing, "StableDiffusionProcessingImg2Img", None)
    if img2img_cls is not None:
        _check_attrs(
            img2img_cls,
            ("init_images", "init_latent", "image_conditioning", "denoising_strength", "mask_for_overlay", "init_img_hash"),
            "StableDiffusionProcessingImg2Img",
            problems,
        )
        base = getattr(img2img_cls, "__mro__", (None,))[1] if len(getattr(img2img_cls, "__mro__", ())) > 1 else None
        if base is not None:
            _check_attrs(base, ("batch_size", "n_iter", "overlay_images", "paste_to", "color_corrections"), "StableDiffusionProcessing", problems)

    # --- the Img2Img caller must be redirectable ------------------------------
    # `modules/img2img.py` is imported lazily, from inside `create_ui()`. If it
    # is already loaded its direct import has to be rebound; if it is not, it
    # will bind the wrapper by itself when it does load. Importing it here to
    # find out would pull in `modules.ui` far too early, so only an
    # already-loaded module is inspected.
    img2img_module = sys.modules.get("modules.img2img")
    if img2img_module is not None and not hasattr(img2img_module, "process_images"):
        problems.append("modules.img2img does not expose 'process_images'; the Img2Img caller cannot be redirected to the wrapper")

    # --- Krea 2 engine surface -------------------------------------------------
    try:
        from backend.args import dynamic_args

        if not hasattr(dynamic_args, "krea2"):
            problems.append("backend.args.dynamic_args has no 'krea2' flag")
        if not hasattr(dynamic_args, "ref_latents"):
            problems.append("backend.args.dynamic_args has no 'ref_latents'")
    except Exception as exc:
        problems.append(f"cannot import backend.args.dynamic_args ({exc!r})")

    try:
        from backend.diffusion_engine.base import ForgeDiffusionEngine

        # `ini_latent` / `ref_latents` are assigned in __init__, so they only
        # exist on instances; detect.py re-checks them on the live engine.
        _check_attrs(ForgeDiffusionEngine, ("encode_first_stage", "clear_references", "get_learned_conditioning"), "ForgeDiffusionEngine", problems)
    except Exception as exc:
        problems.append(f"cannot import backend.diffusion_engine.base.ForgeDiffusionEngine ({exc!r})")

    try:
        from backend.diffusion_engine.krea import Krea2

        _check_attrs(Krea2, ("encode_first_stage", "get_learned_conditioning"), "Krea2", problems)
    except Exception as exc:
        problems.append(f"cannot import backend.diffusion_engine.krea.Krea2 ({exc!r}) — this build has no native Krea 2 support")

    return CompatReport(not problems, problems, _forge_version())
