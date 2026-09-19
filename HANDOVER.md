# HANDOVER — state of play

**Updated:** 2026-09-18. Read `AGENTS.md` first for the ground rules.

## One-paragraph summary

ATLAS is a relightable Gaussian representation whose renderer is an exact linear
operator in the illumination, so relighting costs the same whether the
illumination is one point light or a full HDR environment. The mathematics is
implemented and tested; **nothing has yet rendered a pixel**, because the model,
the loader and the trainer are not written. The next task is to run the capture
inspector on real data, because the loader is written against what it finds.

## Executed vs reviewed

| Component | State |
| --- | --- |
| `atlas/functional/` — atoms, transport, near-field, prefilter, compress, inverse, calibration, splits | **Executed.** 144 tests, CPU |
| `atlas/tools/inspect_capture.py` | **Executed.** 19 tests, CPU. Never run on real data |
| `atlas/ply.py`, `atlas/model.py` | **Executed apart from the rasteriser call**, which is CUDA-only. 21 tests |
| `atlas/config.py`, `atlas/run.py` | **Executed.** 69 tests. Config hashing, run directories, provenance, ledger |
| `atlas/eval.py`, `atlas/imageio.py` | **Executed.** 66 tests. Tonemapped metrics, the two held-out splits, the gate, the comparison sheet |
| `atlas/bench.py`, chunked contraction | **Executed on CPU.** 56 tests. The sizes that decide `B` need the runner |
| `atlas/reference.py` | **Executed.** 31 tests. CPU rasteriser: oracle, generator, smoke test |
| `atlas/data/synthetic.py` | **Executed.** 22 tests. A capture with known ground truth |
| `atlas/data/loader.py` | **Executed.** 22 tests. Streaming reads, the sealed four-way split |
| `atlas/device.py` | **Executed on CPU, CUDA branch faked.** 39 tests. Detection, preflight, precision, seeding |
| Learned atoms, `ParameterDict` densification view | **Executed on CPU.** 12 tests |
| `atlas/train.py`, `atlas/render.py` | **Not written** |
| CI: `cpu.yml`, `gpu.yml`, `tests/gpu/` | **Written, never executed** — needs the repo and the runner |
| Anything on a GPU | **Never run** |

`make check` is the whole of what has been verified: 524 tests, about 50
seconds, no GPU and no `gsplat` required. It also happens to pass with numpy
absent, which is how this container came back after a restart -- nothing under
`atlas/` imports it.

The one thing CPU tests cannot reach is `gsplat.rasterization` itself. Every
input to it is covered -- the transport contraction, the activations, the PLY
geometry, the shape contracts -- but the call is closed only by the `--smoke`
run on the workstation.

## Measured numbers

From `tests/`, on CPU:

| Gate | Target | Measured |
| --- | --- | --- |
| `max\|Path A − Path B\|`, float64 | — | 1.3e-15 |
| `max\|Path A − Path B\|`, float32 | < 1e-5 | 1.2e-06 |
| Superposition residue | structural | 8.9e-16 |
| Prefilter commutation | < 1e-6 | 4.0e-15 |
| Inverse-lighting iterations to 1e-10 KKT | interactive | 149–159 |
| Flash-bracket offset, 100 mm sphere, ½-px ray error | < 25 mm | 18 mm |
| SSIM against a literal transcription of Wang et al., float64 | exact | < 1e-10 |
| PNG: adaptive filtering vs unfiltered scanlines, on a gradient | pays for itself | 2.23x smaller |
| Chunked contraction, peak RSS, per-primitive light, N=200k B=64 | lower | 150 MB → 11 MB |
| Chunked contraction, peak RSS, screen space, 540x960 B=32 | lower | 198 MB → 21 MB |
| Chunked vs unchunked disagreement, float32, worst of 20 seeds | < 1e-6 | 2.2e-7 of output scale |

## The finding that changes the capture protocol

**A bracket-mounted flash cannot support the held-out-light gate.**

`docs/relighting-atlas.md` specifies an **off-camera** flash whose position is
recovered per shot from chrome spheres. When the chrome-sphere *position* gate
proved geometrically unreachable it was replaced by a bracket-mounted flash at
one fitted camera-frame offset. Those two are not interchangeable:

> If the light is a fixed function of the camera, holding out a light holds out
> its view as well. `split_lights` and `split_views` select the same shots, the
> two reported numbers are the same number, and the gate passes regardless of
> what the model learned.

`tests/test_synthetic.py::test_a_bracket_flash_collapses_the_two_splits_onto_each_other`
runs both splits on a bracket capture and shows they select identical shots.

**What this means for the shoot.** The flash has to move independently of the
camera for the project's central gate to mean anything — a second operator, or
a flash on a stand repositioned between passes. A bracket capture is still
worth having: it gives geometry, it exercises the loader and trainer, and it is
what the offset calibration was built for. It just cannot answer the question
the project exists to answer. `SyntheticConfig.flash_mode` is `"free"` or
`"bracket"`, and the manifest records `splits_are_independent` so nothing
downstream has to infer it.

## What the inspector said about a capture at last

`atlas/tools/inspect_capture.py` was written before any data existed and has
only ever been run against fixtures it built itself. Run against the generated
capture it returns **usable**, finds the `nerf-transforms` solve, all 36 masks,
a highlight-centroid RMS of 0.352 of the frame and a relative luminance spread
of 1.09.

One false positive, recorded rather than patched: it reports "about 16 flash /
no-flash pairs" on a capture that contains none. With no EXIF timestamps the
detector falls back to brightness, and brightness here varies because the light
moves — the same signal. It feeds no verdict, so it stays, but the count must
not be read as a total on the real capture.

## A fourth finding: a memory claim that measurement refuted

`atlas/functional/transport.py` asserted in a comment that the shared far-field
contraction, `einsum("ncb,cb->nc", ...)`, builds a full `[N, 3, B]` temporary
before reducing it, and that chunking would therefore halve its footprint.
`ru_maxrss` says it allocates **nothing** beyond its output: it reduces through
a strided matmul without ever forming the product.

The two paths that *do* allocate a full temporary are the per-primitive light
(near-field training) and the screen-space buffer, measured at 150 MB → 11 MB
and 198 MB → 21 MB respectively. Those are the ones chunking is for. The
comment now says all of this, `tests/test_chunking.py` measures it in a
subprocess rather than arguing it from the shapes, and one test exists purely
to record that the shared path has no saving to make.

The rule this produced is in `AGENTS.md`: measure a memory claim, do not derive
it from the shapes.

## A third finding, from the evaluation harness

**SSIM cannot see the failure the project is gated on.** On a synthetic capture
where the prediction gets the diffuse response right and the specular highlight
35% wrong -- which is precisely what a model that memorised its training
illuminations does on an unseen light -- the two held-out splits separate by
**14.6 dB** of `psnr/mu` and by **0.001** of `ssim/mu`. A run reported on SSIM
alone would look fine.

That is why `RelightingReport` reports both splits in every domain and gates on
PSNR in the mu-law domain, and why there is no unnamed PSNR anywhere in
`atlas/eval.py`: on the same pair, linear-radiance PSNR ranks a model that has
lost its entire diffuse response *above* one that is 10% off on a single
highlight pixel. Both of those are pinned as tests.

## Two findings that changed the plan

Both were found by running code, not reading it, and both are pinned as tests.

**Chrome-sphere light *position* is not recoverable from one small ball.** The
triangulation baseline is the ball's own diameter while its curvature amplifies
ray noise by `standoff / radius`. Measured worst case at a half-pixel ray error:
784 mm with a 40 mm ball. The 2 mm gate was **withdrawn**, not loosened. What
survives is *direction*, at about 0.9°. Position now comes from a bracket-mounted
flash with one camera-frame offset fitted across the whole capture.

**A perfectly regular orbit cannot determine that offset at all.** Constant
standoff, elevation and roll leave the ball stationary in camera coordinates, so
every shot gives the same rank-2 constraint; the fit returns an answer over a
metre wrong with a residual of exactly zero. `solve_flash_offset` returns a
condition number for this reason. A capture must vary all three.

Practical consequence: error falls as `1/radius` — 101 mm at 40 mm diameter,
18 mm at 100 mm, 7.5 mm at 200 mm — while tripling a 60-shot capture barely
moves the 40 mm number. **Use a 100 mm chrome sphere or larger.**

## The next task, and its gate

### 1. Register the GPU runner — the highest-leverage step

`docs/runner-setup.md` has the exact commands. Twenty minutes, no inbound
network needed.

This is not housekeeping. Nobody working on this project from a session can
reach the workstation — there is no `ssh` client and outbound port 22 is
blocked. Without a runner, every GPU question costs a round trip: write code
blind, someone runs it, someone pastes the output. With one, a push runs on the
GPU and the result comes back through the GitHub API, readable by anyone,
including an agent with no shell on the machine.

The first run's **Report the hardware** step printing `nvidia-smi` into a GitHub
log *is* the proof that the loop is closed.

### 2. Run the inspector on the real capture

```bash
pip install -e ".[dev,capture]"
python -m atlas.tools.inspect_capture /path/to/capture -o capture_report.json
```

Needs no GPU and no `gsplat`. It reads; it never writes inside the capture. It
ends in one of three verdicts:

- `usable` — lighting varies and a camera solve exists. Proceed.
- `usable_with_work` — the report names what is missing.
- `not_a_relighting_capture` — the illumination does not change between frames,
  so no relighting model can be trained from it. Worth knowing in a minute.

**Send `capture_report.json` back.** The loader is written against it, not
against a guess. This is the one thing that cannot be obtained without the data.

Note what the tool deliberately does not claim: from pixels alone it cannot
separate a moving light from a moving camera. The EXIF flash tag settles it when
present; otherwise the definitive test needs the camera solve, and the report
says so.

### 3. Then, in order

- ~~`atlas/model.py`~~ — **done.** `RelightSplats` plus Path A rendering.
  `from_ply` initialises geometry from an existing fixed-light reconstruction
  and sets the transport so that **under a uniform white environment the model
  reproduces the colour that reconstruction had** — so iteration zero looks like
  the object, and the smoke test tells you something. Default **B = 32 atoms**,
  not 128: near-field training needs a per-primitive `[N, B]` intermediate,
  77 MB at 600k primitives and B=32 but 307 MB at B=128, before autograd.
- `atlas/data/` — the loader, against the report.
- `atlas/train.py` — trainer. Carry over what the fixed-light work in
  `Cyruskraad/gsplat@codex/gsplat-hq-fixed-light` proved: mask-aware loss on a
  padded foreground crop, alpha BCE, deterministic sealed splits, resume,
  best-checkpoint selection, JSON + CSV stats. New: `log1p` losses on linear
  radiance, HDR metrics, and held-out-**light** evaluation reported next to
  held-out-view.
- First real run. **The gate that decides everything:**

  > Held-out-light PSNR within 1 dB of held-out-view PSNR.

  If it fails, the model is memorising illuminations and none of the speed work
  is worth doing. Diagnose in order: light calibration against the inspector's
  report, then `B`, then the near-field model (it has an ablation switch), then
  geometry.

- `atlas/render.py` — relight under an HDR environment, offline, plus the
  cost-versus-number-of-lights curve. This is the first output that shows the
  idea working rather than merely running.

Everything after that — screen-space Path B, the viewer, the transport rank
study, batched multi-view, inverse lighting as a control — is in
`docs/relighting-atlas.md` and none of it is on the critical path.

## Provenance

The functional layer and its tests were developed in
`Cyruskraad/gsplat`, branch `claude/realtime-multiview-relighting-i9jiiq`, draft
PR #7, and moved here unchanged apart from the package path. That PR carries the
full commit history and the reasoning behind each decision.
