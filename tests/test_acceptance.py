"""The acceptance matrix from section 10 of the specification.

Every test runs against ``tests/forge_stub.py``, a transcription of the Forge
Neo / Krea 2 reference lifecycle. The baseline tests show the failure without
the patch; the rest show the behaviour the specification requires with it.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forge_stub  # noqa: E402

from krea2_batchfix import INFOTEXT_FIELD, VERSION, clone, detect, patch, settings  # noqa: E402


def samples(processed):
    """Sample images only, with any leading grid removed."""
    return processed.images[processed.index_of_first_image :]


class ForgeTestCase(unittest.TestCase):
    """Installs the stub Forge and, by default, the extension's wrapper."""

    install_patch = True
    engine = None

    def setUp(self):
        forge_stub.install()
        self.sd_model = forge_stub.reset(self.engine() if self.engine else None)
        patch.uninstall()
        settings.DEFAULTS[settings.DEBUG] = False
        if self.install_patch:
            self.assertTrue(patch.install(), "the compatibility gate rejected the stub Forge")

    def tearDown(self):
        patch.uninstall()
        forge_stub.uninstall()

    # -- helpers -----------------------------------------------------------
    def make_request(self, image, **kwargs):
        import modules.processing as processing

        params = dict(
            prompt="a photo",
            negative_prompt="",
            seed=1000,
            subseed=500,
            steps=20,
            width=512,
            height=512,
            init_images=[image],
            denoising_strength=1.0,
        )
        params.update(kwargs)
        p = processing.StableDiffusionProcessingImg2Img(**params)
        return p

    def run_request(self, p):
        import modules.img2img as img2img

        # modules/img2img.py holds a direct import of process_images; going
        # through it proves the wrapper is actually reached by the real caller.
        return img2img.process_images(p)

    def run_batch_upload(self, p, sources):
        """Mimic modules/img2img.py::process_batch: one shared `p`, one call per source."""
        import modules.img2img as img2img
        import modules.processing as processing
        from modules.shared import state

        p.seed = processing.get_fixed_seed(p.seed)
        p.subseed = processing.get_fixed_seed(p.subseed)
        state.job_count = len(sources) * p.n_iter

        batch_results = None
        for source in sources:
            p.init_images = [source] * p.batch_size
            proc = img2img.process_images(p)
            if batch_results is None:
                batch_results = proc
            else:
                batch_results.images.extend(proc.images)
                batch_results.infotexts.extend(proc.infotexts)
        return batch_results


# ==========================================================================
# Baseline: the bug, without the extension
# ==========================================================================
class BaselineFailureTests(ForgeTestCase):
    install_patch = False

    def test_batch_count_loses_the_primary_reference(self):
        """Later Batch Count iterations fall back to an unreferenced generation.

        `ini_latent` is consumed by the first conditioning pass and `p.init()`
        never runs again, so as soon as anything invalidates the cond cache
        (here: a per-output prompt) the reference is simply gone.
        """
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, prompt=["one", "two", "three"], n_iter=3, batch_size=1)

        processed = self.run_request(p)
        used = [s.primary for s in samples(processed)]

        self.assertEqual(len(used), 3)
        self.assertEqual(used[0], a, "the first output does get the reference")
        self.assertEqual(used[1:], [None, None], "later outputs lose it")

    def test_batch_size_cannot_reference_every_member(self):
        """One stored reference cannot be concatenated with a multi-member batch."""
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=1, batch_size=4)

        with self.assertRaises(RuntimeError):
            self.run_request(p)


# ==========================================================================
# Single primary reference (spec 10, "Single primary A")
# ==========================================================================
class SinglePrimaryReferenceTests(ForgeTestCase):
    def assert_all_use(self, processed, image, expected_count):
        used = [s.primary for s in samples(processed)]
        self.assertEqual(len(used), expected_count)
        self.assertEqual(used, [image] * expected_count)

    def test_count_1_size_1_is_untouched(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=1, batch_size=1)

        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "single output requested")

        self.assert_all_use(self.run_request(p), a, 1)

    def test_count_4_size_1(self):
        a = forge_stub.FakeImage("A")
        self.assert_all_use(self.run_request(self.make_request(a, n_iter=4, batch_size=1)), a, 4)

    def test_count_1_size_4(self):
        a = forge_stub.FakeImage("A")
        self.assert_all_use(self.run_request(self.make_request(a, n_iter=1, batch_size=4)), a, 4)

    def test_count_3_size_2(self):
        a = forge_stub.FakeImage("A")
        self.assert_all_use(self.run_request(self.make_request(a, n_iter=3, batch_size=2)), a, 6)

    def test_per_output_prompts_still_keep_the_reference(self):
        """The exact case the unpatched build fails."""
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, prompt=["one", "two", "three"], n_iter=3, batch_size=1)

        processed = self.run_request(p)
        self.assert_all_use(processed, a, 3)
        self.assertEqual([s.prompt for s in samples(processed)], ["one", "two", "three"])

    def test_no_duplicate_or_dropped_samples(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, n_iter=3, batch_size=2))
        self.assertEqual(len(samples(processed)), 6)
        self.assertEqual(len(processed.infotexts), 6)
        self.assertEqual(len({id(s) for s in samples(processed)}), 6)


# ==========================================================================
# Batch upload (spec 2B / 8)
# ==========================================================================
class BatchUploadTests(ForgeTestCase):
    def test_three_sources_count_1_size_1(self):
        sources = [forge_stub.FakeImage(name) for name in "ABC"]
        p = self.make_request(sources[0], n_iter=1, batch_size=1)

        processed = self.run_batch_upload(p, sources)
        self.assertEqual([s.primary for s in processed.images], sources)

    def test_three_sources_count_2_size_3_gives_18_in_source_order(self):
        sources = [forge_stub.FakeImage(name) for name in "ABC"]
        p = self.make_request(sources[0], n_iter=2, batch_size=3)

        processed = self.run_batch_upload(p, sources)

        used = [s.primary for s in processed.images]
        self.assertEqual(len(used), 18)
        self.assertEqual(used[0:6], [sources[0]] * 6)
        self.assertEqual(used[6:12], [sources[1]] * 6)
        self.assertEqual(used[12:18], [sources[2]] * 6)

    def test_mixed_resolution_sources(self):
        sources = [
            forge_stub.FakeImage("A", size=(512, 512)),
            forge_stub.FakeImage("B", size=(768, 512)),
            forge_stub.FakeImage("C", size=(640, 960)),
        ]
        p = self.make_request(sources[0], n_iter=2, batch_size=2)

        processed = self.run_batch_upload(p, sources)
        used = [s.primary for s in processed.images]
        self.assertEqual(len(used), 12)
        for i, source in enumerate(sources):
            self.assertEqual(used[i * 4 : (i + 1) * 4], [source] * 4)

    def test_no_state_leaks_between_sources(self):
        sources = [forge_stub.FakeImage(name) for name in "AB"]
        p = self.make_request(sources[0], n_iter=2, batch_size=2)

        self.run_batch_upload(p, sources)
        self.assertIsNone(self.sd_model.ini_latent, "an unconsumed primary reference survived the request")


# ==========================================================================
# Image Stitch (spec 9)
# ==========================================================================
class ImageStitchTests(ForgeTestCase):
    def make_scripts(self, *stitch_images):
        stitch = forge_stub.ImageStitch([forge_stub.FakeImage(name) for name in stitch_images])
        runner = forge_stub.ScriptRunner(stitch)
        return runner, stitch

    def attach(self, p, runner):
        p.script_args = ()
        p.scripts = runner

    def test_stitch_applies_to_every_isolated_output(self):
        a = forge_stub.FakeImage("A")
        runner, stitch = self.make_scripts("S")

        p = self.make_request(a, n_iter=3, batch_size=2)
        self.attach(p, runner)

        processed = self.run_request(p)
        result = samples(processed)

        self.assertEqual(len(result), 6)
        for sample in result:
            self.assertEqual(sample.primary, a)
            self.assertEqual([ref.name for ref in sample.stitch], ["S"])

    def test_stitch_references_are_not_appended_twice(self):
        a = forge_stub.FakeImage("A")
        runner, _ = self.make_scripts("S", "T")

        p = self.make_request(a, n_iter=2, batch_size=2)
        self.attach(p, runner)

        self.run_request(p)

        self.assertEqual([ref.name for ref in self.sd_model.ref_latents], ["S", "T"])

    def test_stitch_with_batch_upload(self):
        sources = [forge_stub.FakeImage(name) for name in "AB"]
        runner, _ = self.make_scripts("S")

        p = self.make_request(sources[0], n_iter=2, batch_size=2)
        self.attach(p, runner)

        processed = self.run_batch_upload(p, sources)
        result = processed.images

        self.assertEqual(len(result), 8)
        for sample in result[:4]:
            self.assertEqual(sample.primary, sources[0])
            self.assertEqual([r.name for r in sample.stitch], ["S"])
        for sample in result[4:]:
            self.assertEqual(sample.primary, sources[1])
            self.assertEqual([r.name for r in sample.stitch], ["S"])

    def test_stitch_alone_without_batching_is_unchanged(self):
        a = forge_stub.FakeImage("A")
        runner, _ = self.make_scripts("S")

        p = self.make_request(a, n_iter=1, batch_size=1)
        self.attach(p, runner)

        processed = self.run_request(p)
        result = samples(processed)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].primary, a)
        self.assertEqual([r.name for r in result[0].stitch], ["S"])
        self.assertEqual(runner.process_calls, 1, "the request was fanned out when it should not have been")

    def test_script_setup_is_not_re_run_per_unit(self):
        a = forge_stub.FakeImage("A")
        runner, _ = self.make_scripts("S")

        p = self.make_request(a, n_iter=2, batch_size=2)
        self.attach(p, runner)
        setup_calls_before = runner.setup_calls

        self.run_request(p)
        self.assertEqual(runner.setup_calls, setup_calls_before)

    def test_every_unit_runs_the_alwayson_hooks(self):
        a = forge_stub.FakeImage("A")
        runner, _ = self.make_scripts("S")

        p = self.make_request(a, n_iter=2, batch_size=2)
        self.attach(p, runner)

        self.run_request(p)
        self.assertEqual(runner.before_process_calls, 4)
        self.assertEqual(runner.process_calls, 4)


# ==========================================================================
# Seeds, metadata and ordering (spec 7)
# ==========================================================================
class SeedAndMetadataTests(ForgeTestCase):
    def test_fixed_seed_follows_forge_progression(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, seed=1000, n_iter=3, batch_size=2))

        self.assertEqual([s.seed for s in samples(processed)], [1000, 1001, 1002, 1003, 1004, 1005])
        self.assertEqual(processed.all_seeds, [1000, 1001, 1002, 1003, 1004, 1005])
        self.assertEqual(processed.seed, 1000)

    def test_subseed_progression(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, seed=1000, subseed=77, n_iter=2, batch_size=2))
        self.assertEqual([s.subseed for s in samples(processed)], [77, 78, 79, 80])

    def test_variation_seed_holds_the_base_seed(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, seed=1000, subseed=77, subseed_strength=0.4, n_iter=2, batch_size=2))
        self.assertEqual([s.seed for s in samples(processed)], [1000] * 4)
        self.assertEqual([s.subseed for s in samples(processed)], [77, 78, 79, 80])

    def test_random_seed_is_resolved_once_for_the_request(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, seed=-1, subseed=-1, n_iter=2, batch_size=3))

        seeds = [s.seed for s in samples(processed)]
        self.assertEqual(len(seeds), 6)
        self.assertNotEqual(seeds[0], -1)
        self.assertEqual(seeds, [seeds[0] + i for i in range(6)], "outputs must not each draw their own random seed")
        self.assertEqual(processed.seed, seeds[0])

    def test_outputs_are_not_identical(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, n_iter=2, batch_size=2))
        self.assertEqual(len({s.seed for s in samples(processed)}), 4)

    def test_diagnostic_infotext_field(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, n_iter=2, batch_size=1))

        for text in processed.infotexts:
            self.assertIn(f"{INFOTEXT_FIELD}: {VERSION}", text)

    def test_normal_metadata_is_preserved(self):
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, n_iter=2, batch_size=1))

        self.assertIn("Seed: 1000", processed.infotexts[0])
        self.assertIn("Seed: 1001", processed.infotexts[1])
        self.assertIn("Steps: 20", processed.infotexts[0])
        self.assertIn("Denoising strength", processed.infotexts[0])

    def test_diagnostic_field_can_be_switched_off(self):
        from modules.shared import opts

        opts.krea2_refbatchfix_infotext = False
        try:
            a = forge_stub.FakeImage("A")
            processed = self.run_request(self.make_request(a, n_iter=2, batch_size=1))
            for text in processed.infotexts:
                self.assertNotIn(INFOTEXT_FIELD, text)
        finally:
            opts.krea2_refbatchfix_infotext = True


# ==========================================================================
# Grids and progress
# ==========================================================================
class GridAndProgressTests(ForgeTestCase):
    def test_units_do_not_produce_their_own_grids(self):
        from modules.shared import opts

        opts.return_grid = True
        a = forge_stub.FakeImage("A")
        processed = self.run_request(self.make_request(a, n_iter=3, batch_size=2))

        self.assertEqual(processed.index_of_first_image, 1)
        self.assertEqual(len(processed.images), 7, "expected exactly one aggregated grid plus six samples")
        self.assertEqual(len(samples(processed)), 6)
        self.assertEqual(len(processed.infotexts), 7)

    def test_grid_is_skipped_for_a_single_output(self):
        from modules.shared import opts

        opts.return_grid = True
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=1, batch_size=1)
        processed = self.run_request(p)
        self.assertEqual(processed.index_of_first_image, 0)
        self.assertEqual(len(processed.images), 1)

    def test_job_count_covers_every_unit(self):
        from modules.shared import state

        a = forge_stub.FakeImage("A")
        state.begin("test")
        self.run_request(self.make_request(a, n_iter=3, batch_size=2))

        self.assertEqual(state.job_count, 6)
        self.assertEqual(state.job_no, 6)

    def test_job_count_covers_every_unit_of_every_source(self):
        from modules.shared import state

        sources = [forge_stub.FakeImage(name) for name in "ABC"]
        p = self.make_request(sources[0], n_iter=2, batch_size=2)
        state.begin("test")
        self.run_batch_upload(p, sources)

        self.assertEqual(state.job_count, 12)
        self.assertEqual(state.job_no, 12)


# ==========================================================================
# Interruption (spec 7 / 10)
# ==========================================================================
class InterruptionTests(ForgeTestCase):
    def test_interrupt_stops_the_fan_out_and_returns_partial_results(self):
        from modules.shared import state

        class InterruptAfterTwoUnits(forge_stub.ScriptRunner):
            def postprocess(self, p, res):
                super().postprocess(p, res)
                if self.postprocess_calls == 2:
                    state.interrupted = True

        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=4, batch_size=1)
        p.script_args = ()
        p.scripts = InterruptAfterTwoUnits()

        processed = self.run_request(p)

        result = samples(processed)
        self.assertEqual(len(result), 2, "units finished before the interrupt are kept, the rest are abandoned")
        self.assertEqual([s.primary for s in result], [a, a])
        self.assertEqual([s.seed for s in result], [1000, 1001])

    def test_stopping_generation_is_respected(self):
        from modules.shared import state

        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=3, batch_size=1)
        state.stopping_generation = True

        processed = self.run_request(p)
        self.assertEqual(len(samples(processed)), 0)
        self.assertEqual(processed.infotexts, [""])

    def test_skip_does_not_abandon_the_remaining_units(self):
        from modules.shared import state

        a = forge_stub.FakeImage("A")
        state.skipped = True
        processed = self.run_request(self.make_request(a, n_iter=3, batch_size=1))
        self.assertEqual(len(samples(processed)), 3)


# ==========================================================================
# Pass-through (spec 5 / 10)
# ==========================================================================
class PassThroughTests(ForgeTestCase):
    def test_txt2img_is_untouched(self):
        import modules.processing as processing

        p = processing.StableDiffusionProcessingTxt2Img(prompt="a photo", seed=1000, n_iter=2, batch_size=2)
        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "not an Img2Img request")

    def test_disabled_setting_passes_through(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=2, batch_size=1)
        decision, reason = detect.should_fan_out(p, enabled=False, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "disabled in settings")

    def test_reference_mode_off_passes_through(self):
        from modules.shared import opts

        opts.krea2_do_reference = False
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=2, batch_size=1)
        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "Krea 2 reference mode not active")

    def test_inpaint_is_not_claimed_by_v1(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=2, batch_size=1)
        p.image_mask = forge_stub.FakeImage("mask")

        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "inpaint/sketch request (not claimed by V1)")

        decision, _ = detect.should_fan_out(p, enabled=True, include_inpaint=True)
        self.assertTrue(decision)

    def test_unit_jobs_never_fan_out_again(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=2, batch_size=2)
        unit = clone.make_unit(
            p,
            primary_image=a,
            seed=1,
            subseed=1,
            prompt="a photo",
            negative_prompt="",
            count_index=0,
            batch_index=0,
            orig_n_iter=2,
            orig_batch_size=2,
        )
        unit.n_iter = 4  # even if something inflates it again

        decision, reason = detect.should_fan_out(unit, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "already an isolated unit job")

    def test_multiple_distinct_init_images_pass_through(self):
        a = forge_stub.FakeImage("A")
        b = forge_stub.FakeImage("B")
        p = self.make_request(a, n_iter=2, batch_size=2)
        p.init_images = [a, b]

        self.assertIsNone(detect.primary_reference(p))
        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "no single primary Img2Img reference")

    def test_repeated_identical_init_images_are_one_reference(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=2, batch_size=2)
        p.init_images = [a, a]
        self.assertIs(detect.primary_reference(p), a)


class NonKreaTests(ForgeTestCase):
    engine = forge_stub.PlainEngine

    def test_non_krea_img2img_is_untouched(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=1, batch_size=4)

        decision, reason = detect.should_fan_out(p, enabled=True, include_inpaint=False)
        self.assertFalse(decision)
        self.assertEqual(reason, "Krea 2 reference mode not active")

        processed = self.run_request(p)
        self.assertEqual(len(samples(processed)), 4)


if __name__ == "__main__":
    unittest.main()
