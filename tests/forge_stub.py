"""A miniature Forge Neo + Krea 2 that reproduces the real reference lifecycle.

This is not a mock of the extension's own behaviour — it is a transcription of
the parts of Forge Neo "neo" @ 6009ffff that the bug lives in, so the tests can
show the failure without the patch and the fix with it:

  modules/processing.py
    StableDiffusionProcessing.setup_prompts / cached_params /
    get_conds_with_caching / clear_prompt_cache
    StableDiffusionProcessingImg2Img.init  (single-image repeat path)
    process_images / process_images_inner  (n_iter loop, seed progression)

  backend/diffusion_engine/krea.py
    Krea2.encode_first_stage      -> stores x[0] as the one-shot `ini_latent`
    Krea2.get_learned_conditioning-> consumes `ini_latent`, clears it, and
                                     publishes dynamic_args.ref_latents

  backend/nn/krea.py
    the reference tensors are concatenated with the image tokens, which needs
    the reference batch to match the sample batch — modelled as the same
    RuntimeError torch.cat raises.

  extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py
    references live on sd_model.ref_latents, are re-read (not consumed) on every
    conditioning pass, and are only re-encoded when its parameter cache changes.
"""

from __future__ import annotations

import random
import sys
import types


# --------------------------------------------------------------------------
# stand-ins for PIL images and latent tensors
# --------------------------------------------------------------------------
class FakeImage:
    """Just enough of a PIL image for hashing and identity checks."""

    def __init__(self, name: str, size=(512, 512), mode="RGB"):
        self.name = name
        self.size = size
        self.mode = mode
        self.info = {}

    def tobytes(self) -> bytes:
        return f"{self.name}:{self.size}:{self.mode}".encode()

    def __repr__(self) -> str:
        return f"<FakeImage {self.name}>"


class Sample:
    """One generated output, recording exactly which references produced it."""

    def __init__(self, primary, stitch, seed, subseed, prompt):
        self.primary = primary
        self.stitch = tuple(stitch)
        self.seed = seed
        self.subseed = subseed
        self.prompt = prompt
        self.info = {}

    def __repr__(self) -> str:
        return f"<Sample primary={self.primary} stitch={self.stitch} seed={self.seed}>"


# --------------------------------------------------------------------------
# backend.args.dynamic_args
# --------------------------------------------------------------------------
class dynamic_args:
    krea2 = True
    ref_latents: list = []
    is_referencing = False

    @classmethod
    def reset(cls):
        cls.ref_latents.clear()


# --------------------------------------------------------------------------
# backend.diffusion_engine
# --------------------------------------------------------------------------
class ForgeDiffusionEngine:
    def __init__(self):
        self.ini_latent = None
        self.ref_latents: list = []
        self.extra_generation_params = {}
        self.comments = []
        self.is_sd1 = False
        self.use_distilled_cfg_scale = False
        self.use_shift = False
        self.tiling_enabled = False

    def encode_first_stage(self, batch):
        raise NotImplementedError

    def get_learned_conditioning(self, prompts, is_negative=False):
        raise NotImplementedError

    def clear_references(self):
        self.ref_latents.clear()


class Krea2(ForgeDiffusionEngine):
    """backend/diffusion_engine/krea.py"""

    def encode_first_stage(self, batch):
        if opts.krea2_do_reference:
            start_image = batch[0]  # x[0] — only ever the first batch member
            if dynamic_args.is_referencing:
                self.ref_latents.append(start_image)
            else:
                self.ini_latent = start_image
        return [("latent", item) for item in batch]

    def get_learned_conditioning(self, prompts, is_negative=False):
        if not is_negative:
            references = [*self.ref_latents]
            if self.ini_latent is not None:
                references.insert(0, self.ini_latent)
                self.ini_latent = None  # one-shot

            if opts.krea2_do_reference and references:
                dynamic_args.ref_latents = list(references)
                return ("cond", tuple(prompts), tuple(references))

            dynamic_args.ref_latents.clear()

        return ("cond", tuple(prompts), ())


class PlainEngine(ForgeDiffusionEngine):
    """A non-Krea engine, used to prove non-Krea workflows pass through."""

    def encode_first_stage(self, batch):
        return [("latent", item) for item in batch]

    def get_learned_conditioning(self, prompts, is_negative=False):
        return ("cond", tuple(prompts), ())


# --------------------------------------------------------------------------
# modules.shared
# --------------------------------------------------------------------------
class Options:
    def __init__(self, **values):
        self.__dict__["data"] = {}
        self.__dict__["data_labels"] = {}
        self.__dict__["data"].update(values)

    def __getattr__(self, item):
        data = self.__dict__["data"]
        if item in data:
            return data[item]
        raise AttributeError(item)

    def __setattr__(self, key, value):
        self.__dict__["data"][key] = value

    def add_option(self, key, info):
        self.__dict__["data_labels"][key] = info
        if key not in self.__dict__["data"]:
            self.__dict__["data"][key] = info.default


class OptionInfo:
    def __init__(self, default=None, label="", section=None, **kwargs):
        self.default = default
        self.label = label
        self.section = section

    def info(self, _text):
        return self

    def link(self, *_args):
        return self


class State:
    def __init__(self):
        self.job = ""
        self.job_count = -1
        self.job_no = 0
        self.skipped = False
        self.interrupted = False
        self.stopping_generation = False

    def begin(self, job="(unknown)"):
        self.job = job
        self.job_count = -1
        self.job_no = 0
        self.skipped = False
        self.interrupted = False
        self.stopping_generation = False

    def nextjob(self):
        self.job_no += 1


opts = Options(
    krea2_do_reference=True,
    persistent_cond_cache=True,
    samples_save=False,
    samples_filename_pattern="",
    return_grid=False,
    grid_save=False,
    grid_only_if_multiple=True,
    grid_format="png",
    grid_extended_filename=False,
    enable_pnginfo=False,
    img2img_batch_use_original_name=False,
    add_model_name_to_info=False,
    add_model_hash_to_info=False,
    add_version_to_infotext=False,
    add_user_name_to_info=False,
    save_init_img=False,
    CLIP_stop_at_last_layers=1,
    face_restoration_model=None,
    face_restoration=False,
    tiling=False,
)

state = State()
sd_model = Krea2()


# --------------------------------------------------------------------------
# modules.processing
# --------------------------------------------------------------------------
def get_fixed_seed(seed):
    if seed == "" or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1
    if seed == -1:
        return int(random.randrange(4294967294))
    return seed


def _hash(value):
    try:
        return hash(value)
    except TypeError:
        return hash(repr(value))


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None):
    if use_main_prompt:
        index = 0
    elif index is None:
        index = position_in_batch + iteration * p.batch_size

    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative = p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]

    params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Seed": p.all_seeds[0] if use_main_prompt else all_seeds[index],
        "Size": f"{p.width}x{p.height}",
    }
    params.update(p.extra_generation_params)

    body = ", ".join(f"{k}: {v}" for k, v in params.items() if v is not None)
    negative_text = f"\nNegative prompt: {negative}" if negative else ""
    return f"{prompt_text}{negative_text}\n{body}".strip()


class Processed:
    def __init__(self, p, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments="", extra_images_list=()):
        self.images = images_list
        self.extra_images = list(extra_images_list)
        self.prompt = p.prompt if not isinstance(p.prompt, list) else p.prompt[0]
        self.negative_prompt = p.negative_prompt if not isinstance(p.negative_prompt, list) else p.negative_prompt[0]
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = "".join(f"{c}\n" for c in p.comments)
        self.width = p.width
        self.height = p.height
        self.batch_size = p.batch_size
        self.index_of_first_image = index_of_first_image
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info] * len(images_list)
        self.video_path = None


class StableDiffusionProcessing:
    cached_c = [None, None, None]
    cached_uc = [None, None, None]

    # Forge declares these as dataclass fields, so they exist on the class too;
    # krea2_batchfix.compat probes for them.
    batch_size = 1
    n_iter = 1
    overlay_images = None
    paste_to = None
    color_corrections = None

    def __init__(self, **kwargs):
        self.prompt = ""
        self.negative_prompt = ""
        self.styles = []
        self.seed = -1
        self.subseed = -1
        self.subseed_strength = 0
        self.seed_resize_from_h = -1
        self.seed_resize_from_w = -1
        self.batch_size = 1
        self.n_iter = 1
        self.steps = 20
        self.cfg_scale = 7.0
        self.distilled_cfg_scale = 3.5
        self.width = 512
        self.height = 512
        self.sampler_name = "Euler"
        self.scheduler = "Simple"
        self.restore_faces = False
        self.tiling = False
        self.do_not_save_samples = True
        self.do_not_save_grid = False
        self.outpath_samples = ""
        self.outpath_grids = ""
        self.override_settings = {}
        self.override_settings_restore_afterwards = True
        self.extra_generation_params = {}
        self.comments = {}
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0
        self.user = None
        self.sd_model_name = None
        self.sd_model_hash = None
        self.sd_vae_name = None
        self.sd_vae_hash = None
        self.is_using_inpainting_conditioning = False
        self.is_hr_pass = False
        self.paste_to = None
        self.overlay_images = None
        self.color_corrections = None
        self.c = None
        self.uc = None
        self.rng = None
        self.sampler = None
        self.step_multiplier = 1
        self.iteration = 0
        self.batch_index = 0
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.main_prompt = None
        self.main_negative_prompt = None
        self.prompts = None
        self.negative_prompts = None
        self.seeds = None
        self.subseeds = None
        self.extra_network_data = None
        self.latents_after_sampling = []
        self.pixels_after_sampling = []
        self.extra_result_images = []
        self.modified_noise = None
        self.scripts_value = None
        self.script_args_value = None
        self.scripts_setup_complete = False

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.cached_c = StableDiffusionProcessing.cached_c
        self.cached_uc = StableDiffusionProcessing.cached_uc

    # -- Forge's script plumbing -------------------------------------------
    @property
    def scripts(self):
        return self.scripts_value

    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    def setup_scripts(self):
        self.scripts_setup_complete = True
        self.scripts.setup_scripts(self)

    # -- prompt / conditioning ---------------------------------------------
    def comment(self, text):
        self.comments[text] = 1

    def clear_prompt_cache(self):
        self.cached_c = [None, None, None]
        self.cached_uc = [None, None, None]
        StableDiffusionProcessing.cached_c = [None, None, None]
        StableDiffusionProcessing.cached_uc = [None, None, None]

    def setup_prompts(self):
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(self, required_prompts):
        return (tuple(required_prompts), self.steps, self.width, self.height, _hash(getattr(self, "init_latent", None)))

    def get_conds_with_caching(self, required_prompts, caches, is_negative):
        key = self.cached_params(required_prompts)
        for cache in caches:
            if cache[0] is not None and key == cache[0]:
                return cache[1]

        cache = caches[0]
        cache[1] = sd_model.get_learned_conditioning(required_prompts, is_negative=is_negative)
        cache[0] = key
        return cache[1]

    def setup_conds(self):
        self.uc = self.get_conds_with_caching(self.negative_prompts, [self.cached_uc], True)
        self.c = self.get_conds_with_caching(self.prompts, [self.cached_c], False)

    def get_token_merging_ratio(self, for_hr=False):
        return 0

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        if not opts.persistent_cond_cache:
            self.clear_prompt_cache()


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    pass


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images = None
    init_latent = None
    image_conditioning = None
    denoising_strength = 0.75
    mask_for_overlay = None
    init_img_hash = None

    def __init__(self, **kwargs):
        self.init_images = None
        self.resize_mode = 0
        self.denoising_strength = 0.75
        self.mask = None
        self.image_mask = None
        self.latent_mask = None
        self.nmask = None
        self.mask_blur = 4
        self.inpaint_full_res = True
        self.inpainting_fill = 0
        self.inpainting_mask_invert = 0
        self.initial_noise_multiplier = 1.0
        self.image_cfg_scale = None
        self.init_latent = None
        self.image_conditioning = None
        self.init_img_hash = None
        self.mask_for_overlay = None
        super().__init__(**kwargs)

    def init(self, all_prompts, all_seeds, all_subseeds):
        """modules/processing.py::StableDiffusionProcessingImg2Img.init (trimmed)"""
        self.extra_generation_params["Denoising strength"] = self.denoising_strength

        imgs = list(self.init_images)
        if len(imgs) == 1:
            batch_images = [imgs[0]] * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = list(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        self.init_latent = tuple(img.name for img in batch_images)
        sd_model.encode_first_stage(batch_images)  # sets Krea's one-shot ini_latent
        self.image_conditioning = ("cond", self.init_latent)


#: Every `sd_vae` override actually consumed by a call, in order.
applied_vae_overrides: list = []


def process_images(p):
    """modules/processing.py::process_images"""
    if p.scripts is not None:
        p.scripts.before_process(p)

    # Forge pops these out of the dict, so a processing object whose
    # override_settings is shared loses them from the second call onwards.
    applied_vae_overrides.append(p.override_settings.pop("sd_vae", None))

    return process_images_inner(p)


def process_images_inner(p):
    """modules/processing.py::process_images_inner (trimmed to the batch loop)"""
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    p.sd_model_name = "krea-2-stub"
    p.sd_model_hash = "0000"

    p.setup_prompts()
    p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
    p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    if p.scripts is not None:
        p.scripts.process(p)

    p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

    if state.job_count == -1:
        state.job_count = p.n_iter

    images_out = []
    infotexts = []

    for n in range(p.n_iter):
        p.iteration = n

        if state.skipped:
            state.skipped = False
        if state.interrupted or state.stopping_generation:
            break

        p.prompts = p.all_prompts[n * p.batch_size : (n + 1) * p.batch_size]
        p.negative_prompts = p.all_negative_prompts[n * p.batch_size : (n + 1) * p.batch_size]
        p.seeds = p.all_seeds[n * p.batch_size : (n + 1) * p.batch_size]
        p.subseeds = p.all_subseeds[n * p.batch_size : (n + 1) * p.batch_size]

        p.setup_conds()
        state.nextjob()

        references = list(dynamic_args.ref_latents)

        # backend/nn/krea.py concatenates the reference tokens with the image
        # tokens; a single stored reference cannot be concatenated with a
        # multi-member sample batch.
        if references and p.batch_size > 1:
            raise RuntimeError(f"Sizes of tensors must match except in dimension 1. Expected size 1 but got size {p.batch_size}")

        primary = references[0] if references else None
        stitch = references[1:] if references else []

        for i in range(len(p.prompts)):
            p.batch_index = i
            images_out.append(Sample(primary, stitch, p.seeds[i], p.subseeds[i], p.prompts[i]))
            infotexts.append(create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i, all_negative_prompts=p.negative_prompts))

    index_of_first_image = 0
    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not (len(images_out) < 2 and opts.grid_only_if_multiple) and images_out:
        if opts.return_grid:
            infotexts.insert(0, infotexts[0] if infotexts else "")
            images_out.insert(0, Sample("grid", [], p.all_seeds[0], p.all_subseeds[0], p.all_prompts[0]))
            index_of_first_image = 1

    if not infotexts:
        infotexts.append("")

    res = Processed(
        p,
        images_list=images_out,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
        extra_images_list=p.extra_result_images,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


# --------------------------------------------------------------------------
# extensions-builtin/sd_forge_image_stitch
# --------------------------------------------------------------------------
class ImageStitch:
    """Only the reference lifecycle matters here."""

    cached_parameters = None

    def __init__(self, references=()):
        self.references = list(references)

    def process(self, p):
        if not self.references:
            if ImageStitch.cached_parameters is None:
                return
            ImageStitch.cached_parameters = None
            p.clear_prompt_cache()
            sd_model.clear_references()
            return

        cache = [ref.name for ref in self.references]
        if ImageStitch.cached_parameters == cache:
            return  # already encoded — do not append again

        ImageStitch.cached_parameters = cache
        p.clear_prompt_cache()
        sd_model.clear_references()

        dynamic_args.is_referencing = True
        for reference in self.references:
            sd_model.encode_first_stage([reference])
        dynamic_args.is_referencing = False


class ScriptRunner:
    """Enough of modules.scripts.ScriptRunner for the alwayson hooks used here."""

    def __init__(self, stitch: ImageStitch | None = None):
        self.stitch = stitch
        self.setup_calls = 0
        self.before_process_calls = 0
        self.process_calls = 0
        self.postprocess_calls = 0

    def setup_scripts(self, p, **_kwargs):
        self.setup_calls += 1

    def before_process(self, p):
        self.before_process_calls += 1

    def process(self, p):
        self.process_calls += 1
        if self.stitch is not None:
            self.stitch.process(p)

    def postprocess(self, p, res):
        self.postprocess_calls += 1


# --------------------------------------------------------------------------
# installation into sys.modules
# --------------------------------------------------------------------------
_STUB_MODULES = (
    "modules",
    "modules.processing",
    "modules.shared",
    "modules.img2img",
    "modules.images",
    "modules.script_callbacks",
    "modules.scripts",
    "modules_forge",
    "modules_forge.forge_version",
    "backend",
    "backend.args",
    "backend.diffusion_engine",
    "backend.diffusion_engine.base",
    "backend.diffusion_engine.krea",
)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install():
    """Install the stub Forge into ``sys.modules``. Returns the stub namespace."""
    this = sys.modules[__name__]

    modules_pkg = _module("modules")
    modules_pkg.__path__ = []

    processing = _module(
        "modules.processing",
        Processed=Processed,
        StableDiffusionProcessing=StableDiffusionProcessing,
        StableDiffusionProcessingImg2Img=StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img=StableDiffusionProcessingTxt2Img,
        process_images=process_images,
        process_images_inner=process_images_inner,
        get_fixed_seed=get_fixed_seed,
        create_infotext=create_infotext,
    )

    shared = _module(
        "modules.shared",
        opts=opts,
        state=state,
        sd_model=sd_model,
        OptionInfo=OptionInfo,
    )

    # modules/img2img.py does `from modules.processing import process_images`
    _module("modules.img2img", process_images=process_images)

    _module(
        "modules.images",
        image_grid=lambda images, batch_size=1: Sample("grid", [], 0, 0, ""),
        save_image=lambda *a, **k: None,
    )

    _module(
        "modules.script_callbacks",
        on_ui_settings=lambda callback, **kw: None,
        on_before_ui=lambda callback, **kw: None,
        on_app_started=lambda callback, **kw: None,
    )

    _module("modules.scripts", ScriptRunner=ScriptRunner)

    modules_forge = _module("modules_forge")
    modules_forge.__path__ = []
    _module("modules_forge.forge_version", version="neo", release="2.28")

    backend = _module("backend")
    backend.__path__ = []
    _module("backend.args", dynamic_args=dynamic_args)

    engines = _module("backend.diffusion_engine")
    engines.__path__ = []
    _module("backend.diffusion_engine.base", ForgeDiffusionEngine=ForgeDiffusionEngine)
    _module("backend.diffusion_engine.krea", Krea2=Krea2)

    modules_pkg.processing = processing
    modules_pkg.shared = shared

    return this


def uninstall():
    for name in _STUB_MODULES:
        sys.modules.pop(name, None)


def reset(engine=None):
    """Return the stub to a clean, single-request state."""
    global sd_model

    sd_model = engine if engine is not None else Krea2()
    sys.modules["modules.shared"].sd_model = sd_model

    dynamic_args.ref_latents = []
    dynamic_args.is_referencing = False
    dynamic_args.krea2 = isinstance(sd_model, Krea2)

    StableDiffusionProcessing.cached_c = [None, None, None]
    StableDiffusionProcessing.cached_uc = [None, None, None]

    ImageStitch.cached_parameters = None
    applied_vae_overrides.clear()

    state.begin("test")
    opts.krea2_do_reference = True
    opts.persistent_cond_cache = True
    opts.return_grid = False
    opts.grid_save = False
    opts.samples_filename_pattern = ""

    return sd_model


def make_img2img(**kwargs):
    p = StableDiffusionProcessingImg2Img(**kwargs)
    return p
