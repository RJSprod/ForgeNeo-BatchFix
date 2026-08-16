"""Seed and prompt expansion, mirroring Forge Neo semantics.

Pure functions only — no Forge imports — so the acceptance matrix around seeds
and ordering can be exercised by the unit tests without a WebUI install.

Reference (modules/processing.py @ Forge Neo "neo"):

    p.all_seeds    = [seed    + (x if p.subseed_strength == 0 else 0)
                      for x in range(len(p.all_prompts))]
    p.all_subseeds = [subseed + x for x in range(len(p.all_prompts))]

    p.seeds = p.all_seeds[n * batch_size : (n + 1) * batch_size]

so the logical index of an output is ``count_index * batch_size + batch_index``
and that index is exactly the offset applied to the resolved base seed.
"""

from __future__ import annotations

from typing import Any


class PromptShapeError(ValueError):
    """Raised when prompt lists do not line up with the requested output count."""


def total_outputs(n_iter: int, batch_size: int) -> int:
    return int(n_iter) * int(batch_size)


def logical_index(count_index: int, batch_index: int, batch_size: int) -> int:
    """Position of an output within the flattened request."""
    return count_index * int(batch_size) + batch_index


def iter_logical_positions(n_iter: int, batch_size: int):
    """Yield ``(logical_index, count_index, batch_index)`` in Forge's own order."""
    for count_index in range(int(n_iter)):
        for batch_index in range(int(batch_size)):
            yield (
                logical_index(count_index, batch_index, batch_size),
                count_index,
                batch_index,
            )


def seed_sequence(base_seed: int, n_iter: int, batch_size: int, subseed_strength: float) -> list[int]:
    """Seeds Forge would have used for the whole request, in logical order."""
    count = total_outputs(n_iter, batch_size)
    if subseed_strength == 0:
        return [int(base_seed) + i for i in range(count)]
    return [int(base_seed)] * count


def subseed_sequence(base_subseed: int, n_iter: int, batch_size: int) -> list[int]:
    """Subseeds Forge would have used for the whole request, in logical order."""
    count = total_outputs(n_iter, batch_size)
    return [int(base_subseed) + i for i in range(count)]


def expand_prompts(prompt: Any, negative_prompt: Any, count: int) -> tuple[list, list]:
    """Replicate ``StableDiffusionProcessing.setup_prompts`` list handling.

    Styles are intentionally *not* applied here: each unit job keeps the raw
    prompt plus the original ``styles`` list, so Forge applies styles itself and
    the resulting infotext is identical to an unpatched run.
    """
    if isinstance(prompt, list):
        prompts = list(prompt)
    elif isinstance(negative_prompt, list):
        prompts = [prompt] * len(negative_prompt)
    else:
        prompts = [prompt] * count

    if isinstance(negative_prompt, list):
        negatives = list(negative_prompt)
    else:
        negatives = [negative_prompt] * len(prompts)

    if len(prompts) != len(negatives):
        raise PromptShapeError(f"Received a different number of prompts ({len(prompts)}) and negative prompts ({len(negatives)})")

    if len(prompts) != count:
        raise PromptShapeError(f"Prompt list length ({len(prompts)}) does not match Batch Count * Batch Size ({count})")

    return prompts, negatives


def rewrite_filename_pattern(pattern: str, orig_n_iter: int, orig_batch_size: int, count_index: int, batch_index: int) -> str:
    """Freeze ``[generation_number]`` / ``[batch_number]`` for an isolated unit.

    Forge resolves both tokens from ``p.n_iter`` / ``p.batch_size`` /
    ``p.iteration`` / ``p.batch_index``. A unit job always runs 1x1, so both
    tokens would collapse to "skip" and every output of a source would land on
    the same filename. Substituting the logical numbers keeps the filenames that
    an unpatched run would have produced.

    Tokens that Forge would legitimately skip (because the *original* request
    was 1x1, or its Batch Size was 1) are left untouched so Forge's own
    skip-previous-text behaviour still applies.
    """
    if not pattern:
        return pattern

    orig_n_iter = int(orig_n_iter)
    orig_batch_size = int(orig_batch_size)

    if not (orig_n_iter == 1 and orig_batch_size == 1):
        generation_number = count_index * orig_batch_size + batch_index + 1
        pattern = pattern.replace("[generation_number]", str(generation_number))

    if orig_batch_size != 1:
        pattern = pattern.replace("[batch_number]", str(batch_index + 1))

    return pattern
