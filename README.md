# Krea 2 Reference Batch Fix — for Stable Diffusion WebUI Forge Neo

Makes **every** output of a native Krea 2 Img2Img / reference-image request use its
primary reference image, for any combination of **Batch Count** and **Batch Size**.

Extension only — no Forge Neo core file is modified, patched on disk, or required to change.

---

## Tested against

| | |
|---|---|
| Repository | [`Haoming02/sd-webui-forge-classic`](https://github.com/Haoming02/sd-webui-forge-classic) (mirrored as `opparco/sd-webui-forge-neo`) |
| Branch | `neo` |
| Commit | `6009ffff99b5d5b4312dc8a8f6476ec0a69b37b1` |
| `modules_forge.forge_version` | `neo-2.28` |

The extension re-verifies this API surface at startup and **refuses to patch anything**
if the shape of Forge has changed (see [Compatibility gate](#compatibility-gate)).

### Exact supported workflow

* **Img2Img** tab, **img2img** mode (`mode == 0`), single upload *or* **Batch → Upload / from dir**
* A native **Krea 2** checkpoint loaded (Forge's own `Krea2` diffusion engine)
* Settings → **`[Krea2] Enable Reference`** switched **on**
* Any **Batch Count** and/or **Batch Size**
* With or without **ImageStitch Integrated** references

Not claimed by v1: Inpaint, Inpaint sketch, Img2Img sketch, and Wan/PiD video paths.
Those pass straight through to stock Forge unless you opt in
(see [Settings](#settings)). txt2img and every non-Krea checkpoint are untouched.

---

## The bug

With one image in Img2Img, Krea 2 uses it as the reference for the **first** generated
image, and later outputs of the same request silently come back as if no reference had
been supplied. ImageStitch references keep working the whole time.

### Why

Three facts from the Forge Neo source line up badly:

1. **`p.init()` runs once, before the Batch Count loop.**
   `modules/processing.py::process_images_inner` calls `p.init(...)` once, then enters
   `for n in range(p.n_iter)`.

2. **The Krea primary reference is one-shot state.**
   `backend/diffusion_engine/krea.py`:

   ```python
   def encode_first_stage(self, x):
       if opts.krea2_do_reference:
           start_image = x[0]...            # only the first batch member
           if dynamic_args.is_referencing:
               self.ref_latents.append(...) # ImageStitch — persistent
           else:
               self.ini_latent = ...        # Img2Img primary — one-shot
   ```

   ```python
   def get_learned_conditioning(self, prompt):
       if not prompt.is_negative_prompt:
           _references = [*self.ref_latents]
           if self.ini_latent is not None:
               _references.insert(0, self.ini_latent)
               self.ini_latent = None       # consumed, never rebuilt
           if opts.krea2_do_reference and bool(_references):
               return self.get_learned_conditioning_with_image(prompt, _references)
           else:
               dynamic_args.ref_latents.clear()   # <-- the reference disappears here
   ```

3. **The reference only reaches the transformer through the conditioning pass.**
   `get_learned_conditioning_with_image` publishes `dynamic_args.ref_latents`, which
   `backend/nn/krea.py` concatenates with the image tokens. So a *second* output only
   keeps its reference if `get_learned_conditioning` runs again **and** `ini_latent`
   has been rebuilt. On iteration 2 neither is guaranteed: `init()` does not re-run, and
   the moment anything invalidates the cond cache (a per-output prompt, a wildcard, an
   extra-network change, another extension calling `clear_prompt_cache`), Krea takes the
   `else` branch and clears the reference outright.

`Batch Size > 1` is broken for a second reason: only `x[0]` is ever stored, so one
reference tensor has to be concatenated with a multi-member sample batch —
`torch.cat` raises `Sizes of tensors must match except in dimension 1`.

ImageStitch survives all of this because `self.ref_latents` is *read*, never consumed.

### Why the previous `before_process()` approach could not work

`Script.before_process()` fires once, around `process_images()`. The Batch Count loop
runs later, inside `process_images_inner()`. Clearing fields on `p` at that point cannot
recreate Krea reference conditioning for outputs 2..N — it only clears state that is
about to be rebuilt anyway.

---

## The fix — logical output isolation

When native Krea 2 reference mode is active and the user asked for more than one output,
the request is fanned out into one **`batch_size = 1, n_iter = 1`** unit job per logical
output. Each unit runs the complete lifecycle:

```
primary image -> VAE/reference encoding -> Krea ini_latent created
              -> conditioning receives the reference -> one sample
```

so the next output starts a fresh lifecycle and receives the primary reference again.
Correctness over batched-inference efficiency — v1 trades throughput for a reference on
every image.

Two details make it deterministic rather than lucky:

* **Every unit gets its own conditioning cache.** Krea injects the reference *while the
  prompt is encoded*, so a cond-cache hit would skip `get_learned_conditioning` entirely
  and leave the previous output's `dynamic_args.ref_latents` in place. Each unit calls
  `clear_prompt_cache()` first, guaranteeing the miss and therefore the re-injection.
  (Switchable — see [Settings](#settings).)
* **`sd_model.ini_latent` is cleared before each unit and after the whole request**, so
  an unconsumed primary reference can never leak into an unrelated later request.

`sd_model.ref_latents` — ImageStitch's persistent path — is never touched. Its own
parameter cache means it re-encodes once per user request and then short-circuits, so
repeated unit jobs cannot append duplicate references.

### Interception

```
modules.processing.process_images   ← preserved, called by every unit job
                ↑
        idempotent wrapper          ← installed at extension startup
                ↑
modules.img2img.process_images      ← rebound (it imports the function directly)
modules.api.api.process_images      ← rebound (same reason)
```

`modules/img2img.py` does `from modules.processing import process_images`, so replacing
only the `modules.processing` attribute would leave the real Img2Img caller pointing at
the unpatched function. Both import orders are covered:

* it is imported **after** the patch (Forge imports it lazily, from inside
  `create_ui()`) → its own `from … import` binds the wrapper;
* it is imported **before** the patch → every module already holding a reference to the
  original is rebound **by identity**, so another extension's wrapper is never clobbered.

An `on_app_started` hook re-runs the rebind and then verifies that
`modules.img2img.process_images` really is the wrapper, printing a warning if it somehow
is not. The extension never imports `modules.img2img` itself — doing so at script-load
time would pull `modules.ui` in far too early.

The wrapper passes straight through unless *all* of these hold:

* `p` is a `StableDiffusionProcessingImg2Img`
* the loaded engine is Forge's native `Krea2` **and** `[Krea2] Enable Reference` is on
* a single primary Img2Img reference exists
* original Batch Count > 1 **or** Batch Size > 1
* `p` is not already an isolated unit job (`p._krea2_reference_batchfix_unit`)

Detection reads the engine class and Forge's own `dynamic_args.krea2` flag — never
checkpoint filenames.

### Batch upload

Forge's outer batch-upload loop in `modules/img2img.py::process_batch` already walks each
uploaded source and calls `process_images()` per source; the extension rides on that
rather than parsing uploads itself. Each source fans out into `Batch Count × Batch Size`
units and returns one aggregated `Processed`, which Forge appends to the gallery:

```
N uploaded references × Batch Count × Batch Size outputs, in source order
```

State from source A cannot reach source B — each unit is built from a sanitized clone.

### Seeds, order and metadata

`seed = -1` is resolved **once** for the logical request, then Forge's own progression is
reproduced exactly:

```
logical_index = count_index * batch_size + batch_index
seed[i]       = base_seed + i          (base_seed when subseed_strength != 0)
subseed[i]    = base_subseed + i
```

Infotext is otherwise unchanged; the extension adds one diagnostic field,
`Krea Ref Batch Fix: 1.0` (switchable). Per-unit grids are disabled and a single grid is
rebuilt over the aggregated samples, matching an unpatched run. Interrupt / Stop is
honoured between units and returns the outputs produced so far.

Two fidelity details worth knowing about:

* `override_settings` is **copied per unit**, because `process_images` *pops* `sd_vae`
  (and an unresolvable `sd_model_checkpoint`) out of the dict — a shared dict would drop
  your VAE override from the second output onwards.
* `[generation_number]` / `[batch_number]` in a filename pattern are resolved from
  `p.n_iter` / `p.batch_size`, which are 1 for a unit job. The logical numbers are
  substituted so filenames match an unpatched run instead of every output colliding on
  one name.

---

## Install

**Extensions → Install from URL**, or clone into the WebUI's `extensions` folder:

```bash
git clone https://github.com/RJSprod/ForgeNeo-BatchFix extensions/krea2-reference-batch-fix
```

Restart the WebUI (a UI reload is not enough — the wrapper is installed at script load).
No third-party dependencies.

On startup you should see:

```
[Krea2 RefBatchFix] active (v1.0) — compatible (Forge neo-2.28, tested against neo-2.28); redirected 2 direct import(s)
```

## Settings

**Settings → Krea 2 Reference Batch Fix**

| Setting | Default | What it does |
|---|---|---|
| Enable the Krea 2 reference batch fix | on | Master switch. Off = stock Forge behaviour. |
| Re-encode conditioning for every isolated output | on | Guarantees Krea re-injects the reference per output. Turning it off is faster but lets the cond cache decide, which is the situation the bug comes from. |
| Also isolate Inpaint / Sketch requests | off | Untested in v1; off means those requests pass through. |
| Add the `Krea Ref Batch Fix` field to infotext | on | One diagnostic field; all other metadata is unchanged. |
| Log one line per isolated output | off | Debug builds: source index, logical output index, resolved seed, and a short hash of the primary input image. Image contents are never logged. |

Debug output looks like:

```
[Krea2 RefBatchFix] fan-out: source=0 count=3 size=2 outputs=6 base_seed=1000 ref=3f2a91c0de
[Krea2 RefBatchFix] unit: source=0 output=1/6 seed=1000 subseed=500 ref=3f2a91c0de
[Krea2 RefBatchFix] unit: source=0 output=2/6 seed=1001 subseed=501 ref=3f2a91c0de
```

## Compatibility gate

At startup `krea2_batchfix/compat.py` probes every Forge API the extension depends on —
`process_images` and its signature, `Processed`, `StableDiffusionProcessingImg2Img` and
its fields, `get_fixed_seed`, `create_infotext`, `modules.img2img`'s direct import,
`dynamic_args.krea2` / `.ref_latents`, `ForgeDiffusionEngine` and `Krea2`.

If anything is missing or has changed shape, **no patch is applied** and the reason is
printed:

```
[Krea2 RefBatchFix] DISABLED — this Forge build does not match the tested API surface:
[Krea2 RefBatchFix]   - modules.processing.process_images has an unexpected signature (2 positional parameters, expected 1)
[Krea2 RefBatchFix]   tested against Forge neo-2.28 @ 6009ffff99b5d5b4312dc8a8f6476ec0a69b37b1
[Krea2 RefBatchFix]   Krea 2 Img2Img batches will run unpatched; no unsafe patch was applied.
```

Krea 2 batches then behave as they do without the extension — never as silently
unreferenced images.

---

## Layout

```
scripts/krea2_reference_batch_fix.py   entry point: sys.path, settings, install
krea2_batchfix/
  compat.py      Forge API probing, fail-closed  ─┐ Forge-version-specific
  detect.py      activation conditions           ─┤
  patch.py       wrapper install / rebinding     ─┘
  fanout.py      the isolation loop              ─┐ generation logic,
  clone.py       sanitized unit-job construction  ┤ independent of Forge internals
  aggregate.py   result collection + grid         ┤
  seeds.py       seed / prompt / filename math   ─┘
  settings.py    Settings section
  logs.py        logging + image digests
tests/
  forge_stub.py            a miniature Forge Neo + Krea 2
  test_seeds.py            pure seed / prompt / filename logic
  test_acceptance.py       the specification's acceptance matrix
  test_patch_and_clone.py  interception, cloning, compatibility gate
```

Forge-version-specific knowledge is confined to `compat.py`, `detect.py` and `patch.py`,
so a future Forge Neo change can be adapted without touching the generation logic.

## Tests

```bash
python3 -m unittest discover -s tests
```

No WebUI, no torch, no GPU: `tests/forge_stub.py` is a transcription of the parts of
Forge Neo the bug lives in — the `n_iter` loop and seed progression from
`modules/processing.py`, the single-image repeat path in
`StableDiffusionProcessingImg2Img.init`, Krea's one-shot `ini_latent` and its consuming
`get_learned_conditioning`, the reference/sample batch concatenation from
`backend/nn/krea.py`, and ImageStitch's cached persistent references.

That makes the failure reproducible without the patch:

```python
def test_batch_count_loses_the_primary_reference(self):
    p = self.make_request(a, prompt=["one", "two", "three"], n_iter=3, batch_size=1)
    used = [s.primary for s in samples(self.run_request(p))]
    self.assertEqual(used[0], a)              # the first output does get the reference
    self.assertEqual(used[1:], [None, None])  # later outputs lose it

def test_batch_size_cannot_reference_every_member(self):
    with self.assertRaises(RuntimeError):     # tensor size mismatch
        self.run_request(self.make_request(a, n_iter=1, batch_size=4))
```

and the acceptance matrix verifiable with it.

### Acceptance matrix

| Case | Expected | Covered by |
|---|---|---|
| A, Count 1 × Size 1 | 1 output, A, not fanned out | `test_count_1_size_1_is_untouched` |
| A, Count 4 × Size 1 | 4 outputs, all A | `test_count_4_size_1` |
| A, Count 1 × Size 4 | 4 outputs, all A | `test_count_1_size_4` |
| A, Count 3 × Size 2 | 6 outputs, all A | `test_count_3_size_2` |
| A/B/C, Count 1 × Size 1 | 3 outputs, each matched to its source | `test_three_sources_count_1_size_1` |
| A/B/C, Count 2 × Size 3 | 18 outputs: A×6, then B×6, then C×6 | `test_three_sources_count_2_size_3_gives_18_in_source_order` |
| A + S, Count 3 × Size 2 | 6 outputs, every one uses A **and** S | `test_stitch_applies_to_every_isolated_output` |
| A/B + S, Count 2 × Size 2 | 8 outputs: A+S ×4, B+S ×4 | `test_stitch_with_batch_upload` |
| Mixed-resolution sources | each source matched to its own outputs | `test_mixed_resolution_sources` |
| `seed = -1` | resolved once, contiguous progression | `test_random_seed_is_resolved_once_for_the_request` |
| Fixed seed | Forge's `base + logical_index` progression | `test_fixed_seed_follows_forge_progression` |
| Interruption during fan-out | partial results, no exception | `test_interrupt_stops_the_fan_out_and_returns_partial_results` |
| Krea Count 1 / Size 1 | unchanged | `test_count_1_size_1_is_untouched` |
| Non-Krea Img2Img | unchanged | `test_non_krea_img2img_is_untouched` |
| txt2img | unchanged | `test_txt2img_is_untouched` |
| ImageStitch without batching | unchanged | `test_stitch_alone_without_batching_is_unchanged` |
| No duplicate Stitch references | `ref_latents` stays `[S, T]` | `test_stitch_references_are_not_appended_twice` |
| No reference leaks after a request | `sd_model.ini_latent is None` | `test_no_state_leaks_between_sources` |
| Overrides survive the fan-out | VAE override applied to every unit | `test_every_unit_still_applies_the_vae_override` |
| Incompatible Forge | patch refused, reason printed | `test_missing_api_disables_the_patch` |

The stub is a model, not the real engine — re-run the matrix in a live Forge Neo whenever
Forge updates its Krea or Img2Img processing code, and re-pin the commit above.

## Limitations

* v1 fans out instead of batching, so a Count 3 × Size 2 request runs six single-image
  jobs. Correctness first; true batched inference is a v2 question and needs Krea's
  reference-conditioning tensor shapes verified for multiple batch members.
* Inpaint / sketch modes are not claimed (opt-in switch provided, untested).
* Script `postprocess` hooks run per unit job rather than once over the whole gallery —
  an inherent consequence of every output being its own Forge job.
* An API request supplying several genuinely different `init_images` in one call passes
  through unchanged; Forge treats those as one batch rather than as separate reference
  requests, and reinterpreting them would change the caller's output count.
* Activation is decided from the **currently loaded** engine. A request that uses an
  `sd_model_checkpoint` override to switch *into* a Krea 2 model from a non-Krea one is
  not detected and passes through; the reverse case is detected and simply runs the
  isolated path unnecessarily, which is slower but produces the same images.
