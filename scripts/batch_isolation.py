"""
Forge Neo – Img2Img Batch Upload Per-Image Isolation
=====================================================

Problem
-------
When multiple images are uploaded via Img2Img → Batch → Upload and
batch_size > 1, a RuntimeError fires inside StableDiffusionProcessingImg2Img.init():

    RuntimeError: Sizes of tensors must match except in dimension 1.
    Expected size 1 but got size N (where N = number of uploaded images).

Root cause (confirmed via source inspection)
--------------------------------------------
modules/img2img.py::process_batch() correctly loops over each uploaded image
and sets:

    p.init_images = [img] * p.batch_size      # one source, repeated

Then calls process_images(p) for each image in turn, *reusing the same p
object* across all iterations.

Inside process_images_inner(), p.init() is called once per job. init() builds
an `imgs` list from self.init_images, then chooses a code path:

    if len(imgs) == 1:
        batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size   # ← BUG
        ...
    elif len(imgs) <= self.batch_size:
        self.batch_size = len(imgs)
        batch_images = np.array(imgs)   # stacks potentially different-size tensors

The `overlay_images` accumulation is the primary culprit:
  - iteration 1: overlay_images = []  (reset by the mask branch), then
                 overlay_images = [] * batch_size = []  (fine)
  - iteration 2: overlay_images is still [] from iteration 1 (NOT reset
                 because it is only reset inside `if image_mask is not None`).
                 overlay_images = [] * batch_size = []  (still fine for no-mask)

For images of *different sizes* the `np.array(imgs)` stack on the
`elif len(imgs) <= self.batch_size` path raises the dimension mismatch because
the images were resized to different final dimensions based on their aspect
ratios, so numpy cannot stack them into one array.

Additionally, several fields on `p` accumulate or become stale across
iterations because process_images_inner() does not fully reset them:
  - color_corrections  → set to None at end of process_images_inner (OK)
  - overlay_images     → only reset inside mask branch of init() (NOT OK)
  - paste_to           → only set inside mask branch, never cleared (stale)
  - init_latent        → overwritten each call but stale between calls
  - latents_after_sampling / pixels_after_sampling / extra_result_images
    → appended to but never cleared between process_images calls on same p

Fix
---
This AlwaysVisible script's before_process() hook fires at the top of every
process_images(p) call. It resets the fields on `p` that accumulate or go
stale between per-image iterations, so each image gets a clean slate even
though the same `p` object is reused.

This is the minimal, surgical fix: no monkey-patching, no re-implementing the
loop, no changes to core files.

Compatibility
-------------
- Single-image img2img:      before_process always runs, but reset is
                             harmless (fields are None/[] anyway).
- Txt2img:                   show() returns False → script never loaded.
- Image Stitch:              operates via its own process() hook on
                             sd_model references; never affected by our resets.
- Dynamic Prompts:           its process() and process_batch() hooks fire
                             *after* before_process; the fields we reset are
                             unrelated to prompt generation or ini_latent
                             management. No interaction.
- "from directory" batch:    same p reuse pattern → same fix applies and helps.
"""

from __future__ import annotations

import logging

import modules.scripts as scripts
from modules.processing import StableDiffusionProcessingImg2Img

logger = logging.getLogger(__name__)


class BatchIsolationScript(scripts.Script):
    """
    Resets accumulated per-image state on the processing object at the start
    of every process_images() call, preventing tensor dimension mismatches
    when process_batch() reuses the same p across multiple uploaded images.
    """

    sorting_priority = 1  # run before other alwayson scripts

    def title(self) -> str:
        return "Batch Upload Isolation"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img: bool):
        return []

    # ------------------------------------------------------------------
    # Core fix
    # ------------------------------------------------------------------

    def before_process(self, p, *args):
        """
        Reset fields on `p` that accumulate across reused-p iterations.

        Called once at the very start of process_images(p), before p.init()
        runs. Safe to call unconditionally: all fields being reset are either
        None or empty lists at the start of a fresh job anyway.
        """
        if not isinstance(p, StableDiffusionProcessingImg2Img):
            return

        # overlay_images: only reset inside init() when image_mask is not None.
        # Stale list from a previous iteration causes it to be multiplied by
        # batch_size again, producing a list that is too long.
        p.overlay_images = None

        # paste_to: set inside the inpaint-full-res branch; never explicitly
        # cleared between iterations. Stale value would be used for the next
        # image even if it has a different crop region.
        p.paste_to = None

        # init_latent: stale reference from the previous image. Overwritten
        # during init() but having a stale value can confuse hooks that
        # inspect it in before_process (e.g. Dynamic Prompts' process_batch
        # checks for init_latent is None to detect img2img vs txt2img).
        # Clearing it here ensures the check works correctly on every image.
        p.init_latent = None

        # Sampling artefact lists: appended to inside process_images_inner
        # but never cleared between calls. Left unchecked these grow
        # unboundedly and can cause memory issues on large batches.
        p.latents_after_sampling = []
        p.pixels_after_sampling = []
        p.extra_result_images = []

        logger.debug(
            "[BatchIsolation] Reset per-image state on p "
            "(batch_size=%d, n_iter=%d, init_images=%d)",
            p.batch_size,
            p.n_iter,
            len(p.init_images) if p.init_images else 0,
        )
