"""Interception mechanics, unit-job sanitizing and the fail-closed gate."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forge_stub  # noqa: E402

from krea2_batchfix import UNIT_SENTINEL, clone, compat, patch  # noqa: E402


class StubbedForgeTestCase(unittest.TestCase):
    def setUp(self):
        forge_stub.install()
        self.sd_model = forge_stub.reset()
        patch.uninstall()

    def tearDown(self):
        patch.uninstall()
        forge_stub.uninstall()

    def make_request(self, image, **kwargs):
        import modules.processing as processing

        params = dict(
            prompt="a photo",
            negative_prompt="ugly",
            seed=1000,
            subseed=500,
            init_images=[image],
            n_iter=3,
            batch_size=2,
        )
        params.update(kwargs)
        return processing.StableDiffusionProcessingImg2Img(**params)


# ==========================================================================
# Interception (spec 5)
# ==========================================================================
class PatchInstallTests(StubbedForgeTestCase):
    def test_install_replaces_processing_and_the_direct_importer(self):
        import modules.img2img as img2img
        import modules.processing as processing

        original = processing.process_images
        self.assertIs(img2img.process_images, original)

        self.assertTrue(patch.install())

        self.assertIsNot(processing.process_images, original)
        self.assertIs(img2img.process_images, processing.process_images, "modules/img2img.py imports process_images directly and must reach the wrapper")
        self.assertIs(patch.original_process_images(), original)

    def test_install_is_idempotent(self):
        import modules.processing as processing

        self.assertTrue(patch.install())
        wrapper = processing.process_images

        self.assertTrue(patch.install())
        self.assertIs(processing.process_images, wrapper, "installing twice must not double-wrap")

    def test_uninstall_restores_forge(self):
        import modules.img2img as img2img
        import modules.processing as processing

        original = processing.process_images
        patch.install()
        patch.uninstall()

        self.assertIs(processing.process_images, original)
        self.assertIs(img2img.process_images, original)
        self.assertFalse(patch.is_installed())

    def test_module_imported_after_the_patch_binds_the_wrapper(self):
        """modules/img2img.py is imported from inside create_ui(), i.e. after us."""
        import types

        import modules.processing as processing

        del sys.modules["modules.img2img"]
        self.assertTrue(patch.install())

        # `from modules.processing import process_images`, executed late
        late = types.ModuleType("modules.img2img")
        late.process_images = processing.process_images
        sys.modules["modules.img2img"] = late

        self.assertIs(late.process_images, processing.process_images)
        self.assertTrue(patch.verify_img2img_binding())

    def test_verify_reports_a_stale_binding(self):
        import types

        original = sys.modules["modules.img2img"].process_images
        self.assertTrue(patch.install())

        stale = types.ModuleType("modules.img2img")
        stale.process_images = original
        sys.modules["modules.img2img"] = stale

        self.assertFalse(patch.verify_img2img_binding())

    def test_late_hooks_are_only_registered_once(self):
        import modules.script_callbacks as callbacks

        registered = []
        callbacks.on_app_started = lambda cb, **kw: registered.append(cb)
        callbacks.on_before_ui = lambda cb, **kw: registered.append(cb)

        patch.install()
        patch._late_hooks_registered = False
        patch.install_late_hooks()
        patch.install_late_hooks()

        self.assertEqual(len(registered), 2)

    def test_wrapper_is_marked_for_reentry_detection(self):
        import modules.processing as processing

        patch.install()
        self.assertTrue(getattr(processing.process_images, patch.WRAPPER_MARK, False))

    def test_no_forge_source_file_is_written(self):
        """The patch is in-memory only; nothing is imported that writes to disk."""
        patch.install()
        forge_root = os.path.dirname(os.path.abspath(forge_stub.__file__))
        before = {name: os.path.getmtime(os.path.join(forge_root, name)) for name in os.listdir(forge_root) if name.endswith(".py")}

        a = forge_stub.FakeImage("A")
        import modules.img2img as img2img

        img2img.process_images(self.make_request(a))

        after = {name: os.path.getmtime(os.path.join(forge_root, name)) for name in os.listdir(forge_root) if name.endswith(".py")}
        self.assertEqual(before, after)


# ==========================================================================
# Fail-closed compatibility (spec 5, DoD 7)
# ==========================================================================
class CompatibilityGateTests(StubbedForgeTestCase):
    def test_stub_forge_is_accepted(self):
        report = compat.check()
        self.assertTrue(report.ok, report.problems)

    def test_missing_api_disables_the_patch(self):
        import modules.processing as processing

        original = processing.process_images
        removed = processing.create_infotext
        del processing.create_infotext
        try:
            report = compat.check()
            self.assertFalse(report.ok)
            self.assertIn("modules.processing.create_infotext is missing", report.problems)

            self.assertFalse(patch.install(), "an incompatible build must not be patched")
            self.assertIs(processing.process_images, original)
            self.assertFalse(patch.is_installed())
            self.assertIsNotNone(patch.disabled_reason())
        finally:
            processing.create_infotext = removed

    def test_changed_signature_disables_the_patch(self):
        import modules.processing as processing

        original = processing.process_images
        processing.process_images = lambda p, extra_argument: original(p)
        try:
            report = compat.check()
            self.assertFalse(report.ok)
            self.assertFalse(patch.install())
        finally:
            processing.process_images = original

    def test_missing_krea_engine_disables_the_patch(self):
        krea_module = sys.modules.pop("backend.diffusion_engine.krea")
        try:
            report = compat.check()
            self.assertFalse(report.ok)
            self.assertTrue(any("Krea2" in problem for problem in report.problems))
        finally:
            sys.modules["backend.diffusion_engine.krea"] = krea_module


# ==========================================================================
# Unit-job cloning (spec 6)
# ==========================================================================
class CloneTests(StubbedForgeTestCase):
    def build_unit(self, p, image, **kwargs):
        params = dict(
            primary_image=image,
            seed=1234,
            subseed=99,
            prompt="a photo",
            negative_prompt="ugly",
            count_index=1,
            batch_index=1,
            orig_n_iter=3,
            orig_batch_size=2,
        )
        params.update(kwargs)
        return clone.make_unit(p, **params)

    def test_unit_is_a_single_output_job_with_one_reference(self):
        a = forge_stub.FakeImage("A")
        unit = self.build_unit(self.make_request(a), a)

        self.assertEqual(unit.batch_size, 1)
        self.assertEqual(unit.n_iter, 1)
        self.assertEqual(unit.init_images, [a])
        self.assertEqual(unit.seed, 1234)
        self.assertEqual(unit.subseed, 99)
        self.assertTrue(getattr(unit, UNIT_SENTINEL))
        self.assertTrue(unit.do_not_save_grid)

    def test_derived_runtime_state_is_reset(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a)

        p.init_latent = ("stale",)
        p.image_conditioning = ("stale",)
        p.overlay_images = ["stale"]
        p.paste_to = (1, 2, 3, 4)
        p.mask_for_overlay = "stale"
        p.color_corrections = ["stale"]
        p.c = "stale"
        p.uc = "stale"
        p.rng = "stale"
        p.sampler = "stale"
        p.latents_after_sampling = ["stale"]
        p.pixels_after_sampling = ["stale"]
        p.extra_result_images = ["stale"]
        p.modified_noise = "stale"
        p.all_seeds = [1, 2, 3]
        p.seeds = [1]
        p.extra_network_data = {"lora": ["stale"]}
        p.iteration = 2
        p.batch_index = 1
        p._all_prompts_c = ["stale"]

        unit = self.build_unit(p, a)

        for name in clone.RESET_TO_NONE:
            self.assertIsNone(getattr(unit, name), f"{name} was not reset")
        for name in clone.RESET_TO_LIST:
            self.assertEqual(getattr(unit, name), [], f"{name} was not reset")
        self.assertEqual(unit.iteration, 0)
        self.assertEqual(unit.batch_index, 0)
        self.assertFalse(hasattr(unit, "_all_prompts_c"))

        # the source object is untouched
        self.assertEqual(p.init_latent, ("stale",))

    def test_request_configuration_is_preserved(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, steps=33, width=768, height=1024, cfg_scale=4.5, distilled_cfg_scale=2.5, denoising_strength=0.62, sampler_name="DPM++ 2M", scheduler="Karras", resize_mode=2, styles=["cinematic"])
        unit = self.build_unit(p, a)

        for name in ("steps", "width", "height", "cfg_scale", "distilled_cfg_scale", "denoising_strength", "sampler_name", "scheduler", "resize_mode", "styles", "outpath_samples", "outpath_grids", "subseed_strength"):
            self.assertEqual(getattr(unit, name), getattr(p, name), name)

    def test_scripts_and_arguments_survive_without_re_running_setup(self):
        a = forge_stub.FakeImage("A")
        runner = forge_stub.ScriptRunner()
        p = self.make_request(a)
        p.script_args = ("arg-1", "arg-2")
        p.scripts = runner
        setup_calls = runner.setup_calls

        unit = self.build_unit(p, a)

        self.assertIs(unit.scripts, runner)
        self.assertEqual(unit.script_args, ("arg-1", "arg-2"))
        self.assertTrue(unit.scripts_setup_complete)
        self.assertEqual(runner.setup_calls, setup_calls)

    def test_override_settings_are_copied_not_shared(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, override_settings={"sd_vae": "vae.safetensors"})
        unit = self.build_unit(p, a)

        self.assertIsNot(unit.override_settings, p.override_settings)
        unit.override_settings.pop("sd_vae")
        self.assertEqual(p.override_settings, {"sd_vae": "vae.safetensors"})

    def test_extra_generation_params_are_copied_not_shared(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a)
        p.extra_generation_params["Existing"] = "kept"
        unit = self.build_unit(p, a)

        self.assertEqual(unit.extra_generation_params["Existing"], "kept")
        unit.extra_generation_params["Unit only"] = "x"
        self.assertNotIn("Unit only", p.extra_generation_params)

    def test_filename_tokens_are_frozen_per_output(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, override_settings={"samples_filename_pattern": "source-[generation_number]"})

        unit = self.build_unit(p, a, count_index=2, batch_index=1)
        self.assertEqual(unit.override_settings["samples_filename_pattern"], "source-6")

    def test_unit_gets_a_private_conditioning_cache_when_cleared(self):
        a = forge_stub.FakeImage("A")
        p = self.make_request(a)
        unit_a = self.build_unit(p, a)
        unit_b = self.build_unit(p, a)

        unit_a.clear_prompt_cache()
        unit_b.clear_prompt_cache()
        self.assertIsNot(unit_a.cached_c, unit_b.cached_c, "each unit must miss the cond cache so Krea re-injects its reference")


# ==========================================================================
# Overrides survive the fan-out end to end
# ==========================================================================
class OverrideSettingsIntegrationTests(StubbedForgeTestCase):
    def test_every_unit_still_applies_the_vae_override(self):
        import modules.img2img as img2img

        patch.install()
        a = forge_stub.FakeImage("A")
        p = self.make_request(a, n_iter=3, batch_size=1, override_settings={"sd_vae": "vae.safetensors"})

        img2img.process_images(p)

        self.assertEqual(forge_stub.applied_vae_overrides, ["vae.safetensors"] * 3)


if __name__ == "__main__":
    unittest.main()
