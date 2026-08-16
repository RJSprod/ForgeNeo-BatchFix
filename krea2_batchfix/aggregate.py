"""Collecting isolated unit results into a single ``Processed``.

Outputs stay in logical order (Batch Count outer, Batch Size inner), which is
also the order Forge itself would have produced them in, so the outer
batch-upload loop in ``modules/img2img.py`` can keep appending source after
source to the same gallery.
"""

from __future__ import annotations

from .logs import logger


class ResultCollector:
    """Accumulates images, infotexts and seeds across unit jobs."""

    def __init__(self):
        self.images: list = []
        self.extra_images: list = []
        self.infotexts: list[str] = []
        self.seeds: list[int] = []
        self.subseeds: list[int] = []
        self.prompts: list = []
        self.negative_prompts: list = []

    def __len__(self) -> int:
        return len(self.images)

    def add(self, unit, processed, *, seed: int, subseed: int) -> int:
        """Append one unit's samples. Returns how many images were taken."""
        if processed is None:
            return 0

        images = list(getattr(processed, "images", None) or [])
        # A unit runs with do_not_save_grid, so index_of_first_image is 0; honour
        # it anyway in case a script inserted something ahead of the samples.
        first = int(getattr(processed, "index_of_first_image", 0) or 0)
        if first:
            images = images[first:]

        infotexts = list(getattr(processed, "infotexts", None) or [])
        if first:
            infotexts = infotexts[first:]

        if len(infotexts) < len(images):
            filler = infotexts[-1] if infotexts else getattr(processed, "info", "") or ""
            infotexts = infotexts + [filler] * (len(images) - len(infotexts))
        elif len(infotexts) > len(images):
            infotexts = infotexts[: len(images)]

        self.images.extend(images)
        self.infotexts.extend(infotexts)
        self.extra_images.extend(list(getattr(processed, "extra_images", None) or []))

        unit_prompts = list(getattr(processed, "all_prompts", None) or [getattr(unit, "prompt", "")])
        unit_negatives = list(getattr(processed, "all_negative_prompts", None) or [getattr(unit, "negative_prompt", "")])

        for i in range(len(images)):
            self.seeds.append(seed)
            self.subseeds.append(subseed)
            self.prompts.append(unit_prompts[min(i, len(unit_prompts) - 1)] if unit_prompts else "")
            self.negative_prompts.append(unit_negatives[min(i, len(unit_negatives) - 1)] if unit_negatives else "")

        return len(images)


def _as_list(prompt):
    return list(prompt) if isinstance(prompt, list) else [prompt]


def build_processed(p, collector: ResultCollector, *, base_seed: int, base_subseed: int):
    """Assemble the aggregated ``Processed`` that the caller receives."""
    from modules.processing import Processed

    all_prompts = collector.prompts or p.all_prompts or _as_list(p.prompt)
    all_negative_prompts = collector.negative_prompts or p.all_negative_prompts or _as_list(p.negative_prompt)
    all_seeds = collector.seeds or [base_seed]
    all_subseeds = collector.subseeds or [base_subseed]

    # create_infotext() and the grid filename read these off `p`.
    p.all_prompts = all_prompts
    p.all_negative_prompts = all_negative_prompts
    p.all_seeds = all_seeds
    p.all_subseeds = all_subseeds
    p.main_prompt = all_prompts[0]
    p.main_negative_prompt = all_negative_prompts[0]

    images = list(collector.images)
    infotexts = list(collector.infotexts)
    index_of_first_image = 0

    grid_infotext = _maybe_add_grid(p, images, infotexts)
    if grid_infotext is not None:
        index_of_first_image = 1

    if not infotexts:
        infotexts = [""]

    processed = Processed(
        p,
        images_list=images,
        seed=base_seed,
        info=infotexts[0],
        subseed=base_subseed,
        all_prompts=all_prompts,
        all_negative_prompts=all_negative_prompts,
        all_seeds=all_seeds,
        all_subseeds=all_subseeds,
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
        extra_images_list=list(collector.extra_images),
    )

    return processed


def _maybe_add_grid(p, images: list, infotexts: list[str]):
    """Reproduce Forge's end-of-job grid for the aggregated samples.

    Each unit runs with ``do_not_save_grid`` so no per-unit grids exist; this
    rebuilds the one grid the user would have got from an unpatched run.
    Returns the grid infotext when a grid was inserted, otherwise ``None``.
    """
    try:
        from modules import images as images_module
        from modules.shared import opts
    except Exception:
        return None

    if getattr(p, "do_not_save_grid", False):
        return None

    if not (getattr(opts, "return_grid", False) or getattr(opts, "grid_save", False)):
        return None

    if len(images) < 2 and getattr(opts, "grid_only_if_multiple", True):
        return None

    if not images:
        return None

    try:
        grid = images_module.image_grid(images, p.batch_size)
    except Exception:
        logger.warning("Could not build the aggregated grid", exc_info=True)
        return None

    text = infotexts[0] if infotexts else ""
    inserted = False

    if getattr(opts, "return_grid", False):
        infotexts.insert(0, text)
        if getattr(opts, "enable_pnginfo", False):
            try:
                grid.info["parameters"] = text
            except Exception:
                pass
        images.insert(0, grid)
        inserted = True

    if getattr(opts, "grid_save", False):
        try:
            images_module.save_image(
                grid,
                p.outpath_grids,
                "grid",
                p.all_seeds[0],
                p.all_prompts[0],
                opts.grid_format,
                info=text,
                short_filename=not getattr(opts, "grid_extended_filename", False),
                p=p,
                grid=True,
            )
        except Exception:
            logger.warning("Could not save the aggregated grid", exc_info=True)

    return text if inserted else None
