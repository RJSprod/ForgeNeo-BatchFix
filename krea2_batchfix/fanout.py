"""Logical output isolation — the actual fix.

When native Krea 2 reference mode is active and the user asked for more than one
output, the request is fanned out into one ``batch_size=1, n_iter=1`` unit job
per logical output. Each unit runs the complete Forge/Krea lifecycle:

    primary image -> VAE/reference encoding -> Krea ini_latent created
                  -> conditioning receives the reference -> one sample

so the next output starts a fresh lifecycle and receives the primary reference
again, instead of inheriting the one-shot ``ini_latent`` that Krea consumes and
clears inside ``get_learned_conditioning``.
"""

from __future__ import annotations

from .aggregate import ResultCollector, build_processed
from .clone import make_unit, merge_back
from .detect import primary_reference
from .logs import logger, short_image_hash
from .seeds import PromptShapeError, expand_prompts, iter_logical_positions, seed_sequence, subseed_sequence, total_outputs

SOURCE_INDEX_ATTR = "_krea2_batchfix_source_index"


def run(p, original_process_images, *, force_recond: bool, add_infotext: bool):
    """Fan ``p`` out into isolated unit jobs and return one aggregated result."""
    from modules.processing import get_fixed_seed
    from modules.shared import state

    n_iter = int(getattr(p, "n_iter", 1) or 1)
    batch_size = int(getattr(p, "batch_size", 1) or 1)
    total = total_outputs(n_iter, batch_size)

    reference = primary_reference(p)
    if reference is None:
        return original_process_images(p)

    # Resolve `seed = -1` once for the whole logical request, never per unit,
    # so the isolated outputs follow Forge's normal seed progression instead of
    # all landing on independent random seeds.
    base_seed = get_fixed_seed(getattr(p, "seed", -1))
    base_subseed = get_fixed_seed(getattr(p, "subseed", -1))
    p.seed = base_seed
    p.subseed = base_subseed

    try:
        prompts, negative_prompts = expand_prompts(p.prompt, p.negative_prompt, total)
    except PromptShapeError as exc:
        logger.warning("Passing through to stock Forge: %s", exc)
        return original_process_images(p)

    subseed_strength = float(getattr(p, "subseed_strength", 0) or 0)
    seeds = seed_sequence(base_seed, n_iter, batch_size, subseed_strength)
    subseeds = subseed_sequence(base_subseed, n_iter, batch_size)

    source_index = _next_source_index(p)
    reference_hash = short_image_hash(reference)

    logger.debug(
        "fan-out: source=%d count=%d size=%d outputs=%d base_seed=%d ref=%s",
        source_index,
        n_iter,
        batch_size,
        total,
        base_seed,
        reference_hash,
    )

    _widen_job_count(state, planned=total, budgeted=n_iter)

    collector = ResultCollector()

    try:
        for logical, count_index, batch_index in iter_logical_positions(n_iter, batch_size):
            if state.skipped:
                state.skipped = False

            if state.interrupted or state.stopping_generation:
                logger.info("Interrupted after %d of %d isolated outputs", len(collector), total)
                break

            unit = make_unit(
                p,
                primary_image=reference,
                seed=seeds[logical],
                subseed=subseeds[logical],
                prompt=prompts[logical],
                negative_prompt=negative_prompts[logical],
                count_index=count_index,
                batch_index=batch_index,
                orig_n_iter=n_iter,
                orig_batch_size=batch_size,
                add_infotext_field=add_infotext,
            )

            _prime_krea_reference_state(unit, force_recond=force_recond)

            logger.debug(
                "unit: source=%d output=%d/%d seed=%d subseed=%d ref=%s",
                source_index,
                logical + 1,
                total,
                seeds[logical],
                subseeds[logical],
                reference_hash,
            )

            try:
                processed = original_process_images(unit)
            finally:
                merge_back(p, unit)
                _close(unit)

            taken = collector.add(unit, processed, seed=seeds[logical], subseed=subseeds[logical])
            if taken == 0:
                logger.debug("unit %d of %d produced no image", logical + 1, total)
    finally:
        _release_krea_reference_state()

    logger.debug("fan-out complete: source=%d produced %d of %d requested outputs", source_index, len(collector), total)

    return build_processed(p, collector, base_seed=base_seed, base_subseed=base_subseed)


def _next_source_index(p) -> int:
    """Ordinal of the current source image within an outer batch-upload run.

    ``modules/img2img.py::process_batch`` reuses one processing object across
    every uploaded source, so a counter stored on that object counts sources.
    """
    index = int(getattr(p, SOURCE_INDEX_ATTR, -1)) + 1
    try:
        setattr(p, SOURCE_INDEX_ATTR, index)
    except Exception:
        return 0
    return index


def _widen_job_count(state, *, planned: int, budgeted: int) -> None:
    """Keep the progress bar honest.

    Forge budgeted ``budgeted`` calls to ``state.nextjob()`` for this request
    (one per Batch Count iteration); the fan-out will make ``planned`` of them,
    one per unit job.
    """
    try:
        job_count = getattr(state, "job_count", -1)
        if job_count is None or job_count < 0:
            state.job_count = planned
        else:
            state.job_count = job_count + (planned - budgeted)
    except Exception:
        logger.debug("could not adjust state.job_count", exc_info=True)


def _prime_krea_reference_state(unit, *, force_recond: bool) -> None:
    """Guarantee this unit builds its own Krea primary reference.

    Two things have to hold for the reference to reach the transformer:

    1. ``p.init()`` must run, so ``Krea2.encode_first_stage`` stores a fresh
       ``ini_latent`` — the fan-out gives every unit its own ``init()``.
    2. ``Krea2.get_learned_conditioning`` must actually be called, because that
       is where ``ini_latent`` is turned into ``dynamic_args.ref_latents``. With
       an unchanged prompt and an unchanged init latent the cond cache would hit
       and skip it, leaving the previous output's reference in place. Giving the
       unit its own empty cache forces the miss.

    Image Stitch's persistent ``sd_model.ref_latents`` is deliberately left
    alone: it is a separate path, it survives conditioning re-encoding, and its
    own parameter cache stops it from re-appending references per unit.
    """
    if force_recond:
        try:
            unit.clear_prompt_cache()
        except Exception:
            logger.debug("could not clear the conditioning cache for this unit", exc_info=True)

    _set_ini_latent(None)


def _release_krea_reference_state() -> None:
    """Drop any unconsumed primary reference so it cannot leak into a later request."""
    _set_ini_latent(None)


def _set_ini_latent(value) -> None:
    try:
        from modules import shared

        sd_model = shared.sd_model
        if sd_model is not None and hasattr(sd_model, "ini_latent"):
            sd_model.ini_latent = value
    except Exception:
        logger.debug("could not reset the Krea primary reference state", exc_info=True)


def _close(unit) -> None:
    """Release the unit's sampler and conditioning; Forge only closes the outer `p`."""
    try:
        unit.close()
    except Exception:
        logger.debug("unit close() failed", exc_info=True)
