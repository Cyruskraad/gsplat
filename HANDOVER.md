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
| `atlas/data/`, `atlas/train.py`, `atlas/render.py` | **Not written** |
| CI: `cpu.yml`, `gpu.yml`, `tests/gpu/` | **Written, never executed** — needs the repo and the runner |
| Anything on a GPU | **Never run** |

`make check` is the whole of what has been verified: 188 tests, about 20
seconds, no GPU and no `gsplat` required.

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
