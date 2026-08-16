"""Building sanitized 1x1 unit-processing objects.

A unit job is a shallow copy of the user's processing object with every piece of
*derived runtime state* stripped, so Forge runs the full Img2Img lifecycle from
scratch for it:

    primary image -> VAE/reference encoding -> Krea ini_latent -> conditioning
                  -> exactly one sample

The request configuration (prompts, sampler, scheduler, CFG, distilled CFG,
denoising, dimensions, resize mode, overrides, output paths, scripts and script
arguments) is preserved verbatim.
"""

from __future__ import annotations

import copy

from . import INFOTEXT_FIELD, UNIT_SENTINEL, VERSION
from .seeds import rewrite_filename_pattern

#: Derived state that must not survive into a unit job (spec section 6).
RESET_TO_NONE = (
    "init_latent",
    "image_conditioning",
    "overlay_images",
    "paste_to",
    "mask_for_overlay",
    "color_corrections",
    "c",
    "uc",
    "rng",
    "sampler",
    "mask",
    "nmask",
    "init_img_hash",
    "all_prompts",
    "all_negative_prompts",
    "all_seeds",
    "all_subseeds",
    "prompts",
    "negative_prompts",
    "seeds",
    "subseeds",
    "extra_network_data",
    "main_prompt",
    "main_negative_prompt",
    "modified_noise",
)

#: Accumulator lists that Forge appends to but never clears between runs.
RESET_TO_LIST = (
    "latents_after_sampling",
    "pixels_after_sampling",
    "extra_result_images",
)

#: Scalar runtime state with a well-defined "fresh job" value.
RESET_TO_VALUE = {
    "iteration": 0,
    "batch_index": 0,
    "step_multiplier": 1,
    "is_using_inpainting_conditioning": False,
    "is_hr_pass": False,
}

#: Prompt-comment caches rebuilt per job by whichever script owns them.
DROP_ATTRS = ("_all_prompts_c", "_all_negative_prompts_c")

TOKEN_PATTERN_KEY = "samples_filename_pattern"


def make_unit(
    p,
    *,
    primary_image,
    seed,
    subseed,
    prompt,
    negative_prompt,
    count_index,
    batch_index,
    orig_n_iter,
    orig_batch_size,
    add_infotext_field=True,
):
    """Return a sanitized ``batch_size=1, n_iter=1`` clone of ``p``."""
    unit = copy.copy(p)

    for name in RESET_TO_NONE:
        if hasattr(unit, name):
            setattr(unit, name, None)

    for name in RESET_TO_LIST:
        setattr(unit, name, [])

    for name, value in RESET_TO_VALUE.items():
        if hasattr(unit, name):
            setattr(unit, name, value)

    for name in DROP_ATTRS:
        unit.__dict__.pop(name, None)

    # Give the unit its own mutable containers. `override_settings` in
    # particular is *popped from* by process_images (`sd_vae`, and an
    # unresolvable `sd_model_checkpoint`), so a shared dict would silently drop
    # the user's overrides from the second unit onwards.
    unit.override_settings = dict(getattr(p, "override_settings", None) or {})
    unit.extra_generation_params = dict(getattr(p, "extra_generation_params", None) or {})
    unit.comments = {}

    # The single correct primary Krea reference for this logical output.
    unit.init_images = [primary_image]
    unit.batch_size = 1
    unit.n_iter = 1

    unit.seed = int(seed)
    unit.subseed = int(subseed)
    unit.prompt = prompt
    unit.negative_prompt = negative_prompt

    # Per-unit grids would be one-image grids of the same sample.
    unit.do_not_save_grid = True

    # `p.scripts` / `p.script_args` come across with the shallow copy, and
    # `scripts_setup_complete` comes with them, so assigning them again (which
    # would re-run every script's setup()) is neither needed nor wanted.
    unit.scripts_setup_complete = True

    _freeze_filename_tokens(unit, count_index=count_index, batch_index=batch_index, orig_n_iter=orig_n_iter, orig_batch_size=orig_batch_size)

    if add_infotext_field:
        unit.extra_generation_params[INFOTEXT_FIELD] = VERSION

    setattr(unit, UNIT_SENTINEL, True)

    return unit


def _freeze_filename_tokens(unit, *, count_index, batch_index, orig_n_iter, orig_batch_size):
    """Keep ``[generation_number]`` / ``[batch_number]`` filenames stable.

    Both tokens are resolved from the *unit's* 1x1 counters, which would collapse
    them to nothing and make every output of one source share a filename. The
    logical numbers are substituted instead, reproducing unpatched naming.
    """
    try:
        from modules.shared import opts

        default_pattern = getattr(opts, TOKEN_PATTERN_KEY, "") or ""
    except Exception:
        default_pattern = ""

    pattern = unit.override_settings.get(TOKEN_PATTERN_KEY, default_pattern)
    if not pattern or ("[generation_number]" not in pattern and "[batch_number]" not in pattern):
        return

    rewritten = rewrite_filename_pattern(pattern, orig_n_iter, orig_batch_size, count_index, batch_index)
    if rewritten != pattern:
        unit.override_settings[TOKEN_PATTERN_KEY] = rewritten


def merge_back(p, unit) -> None:
    """Fold a finished unit's collected metadata back onto the source request."""
    try:
        for key, value in getattr(unit, "extra_generation_params", {}).items():
            p.extra_generation_params.setdefault(key, value)
    except Exception:
        pass

    try:
        for comment in getattr(unit, "comments", {}) or {}:
            p.comment(comment)
    except Exception:
        pass

    for name in ("sd_model_name", "sd_model_hash", "sd_vae_name", "sd_vae_hash", "restore_faces", "tiling", "scheduler", "sampler_name", "init_img_hash", "width", "height"):
        value = getattr(unit, name, None)
        if value is not None:
            try:
                setattr(p, name, value)
            except Exception:
                pass
