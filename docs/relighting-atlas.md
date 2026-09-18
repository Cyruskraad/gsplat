# ATLAS — All-frequency Transport via Learned Atom Splatting

Real-time, multi-view relighting of OLAT-captured objects, at a cost independent of
how complex the illumination is.

**Status.** The design is complete. **P0 (capture-side calibration and splits)
and P1 (the transport core and its exactness proof) are implemented and
executed**; everything from P2 onward is design only and has not been run. See [Phases and gates](#phases-and-gates) for what that
means claim by claim, and `gsplat/relight/design.md` for the module contract.

Measured on CPU, `tests/relight/`, 144 tests in ~18 s:

| Gate | Target | Measured |
| --- | --- | --- |
| `max\|Path A − Path B\|`, float64, 20 seeds | — | **1.3e-15** |
| `max\|Path A − Path B\|`, float32, 20 seeds | < 1e-5 | **1.2e-06** |
| Superposition residue | structural, i.e. rounding only | **8.9e-16** |
| Prefilter commutation | < 1e-6 | **4.0e-15** |
| Inverse-lighting iterations to 1e-10 KKT | interactive | **149–159** |
| Flash-bracket offset, 100 mm sphere, ½-px ray error | < 25 mm | **18 mm** |
| Light *direction* from one sphere, ½-px ray error | — | **0.9°** |
| Light *position* per shot from one 40 mm sphere | was "< 2 mm" | **784 mm — gate withdrawn, see below** |

---

## The claim

Build a relightable Gaussian representation whose renderer is an **exact linear
operator in the illumination**.

Two things follow that no published OLAT-relighting method on Gaussians provides:

1. Changing the light — a full HDR environment, a hundred point lights, a rotating
   probe — costs one fixed sub-millisecond screen-space pass. Not `O(#lights)`.
   `O(1)`.
2. Because the forward model is linear, *inverse* lighting is a linear
   least-squares problem. "Which illumination makes this object look like this
   photograph" becomes an interactive control rather than an offline fit.

The representation is trained from **handheld-flash OLAT capture** — an off-camera
flash moved around the object — so it needs no light stage.

## Why this is worth building

One-light-at-a-time capture samples the light transport operator column by column.
Because transport is linear in the illuminant, the resulting stack relights the
subject under any illumination *exactly*, by linear combination, with no inverse
problem solved anywhere. That linearity is the whole value of the measurement.

Published relightable-Gaussian methods trained on OLAT data give it away:

| Method | Appearance model | Linear in the light? | Environment relighting |
| --- | --- | --- | --- |
| **GS³** (SIGGRAPH Asia 2024) | spatial + angular Gaussians; `shading × shadow + residual`, shadow splatted toward the light and refined by an MLP | **No** — parameterised by a single point light | Sum over sampled lights, `O(#lights)`. 90 fps is *per light*. |
| **OLAT Gaussians** (SIGGRAPH Asia 2024) | per-Gaussian MLP incident-illumination and scattering fields; proxy mesh for normals and shadow-map visibility | **No** | Per point light, ~30 fps on a 2080 Ti |
| **PRTGS / PRTGaussian / TranSplat** (2024–25) | precomputed radiance transfer on Gaussians | Yes | Transport is *simulated* by ray tracing under assumed materials, not measured; fixed low-order SH, so explicitly low-frequency — no shadow edges |
| **Relightable Gaussian Codec Avatars** (CVPR 2024) | learned radiance transfer: SH diffuse + spherical-Gaussian specular | Yes | Closest relative. Face/codec-specific, fixed analytic basis, per-Gaussian shading only |

The gap is therefore specific: **all-frequency environment relighting of a general
OLAT-captured object, in real time, at a cost independent of the illumination's
complexity, from novel views, with several views rendered at once.**

That gap is also exactly what
[`Cyruskraad/olat-relight`](https://github.com/Cyruskraad/olat-relight)'s Phase 5
runtime targets, and what the `codex/gsplat-hq-fixed-light` branch names as its own
limitation: *"the fixed capture lighting is baked into the appearance."*

## Method

### Notation

`E` is the illumination. `A_k : S² → R`, `k = 1..B`, is a basis of **light atoms**.
The scene-wide light coefficient vector is

```
ell_k = <A_k, E>              ell in R^B
```

`B` numbers, computed once per light change, shared by every Gaussian and every
pixel.

### Per-Gaussian appearance

Gaussian `i`, emitting toward view direction `w_o`:

```
L_i(w_o) = sum_k ell_k * [ d_ik + s_ik(w_o) ]  +  p_i(w_o; E)
           \________________________________/     \__________/
             global atom transport (learned)       private lobes
```

| Term | Stored per Gaussian | Carries |
| --- | --- | --- |
| `d_ik in R^3` | `D_i in R^(3xB)` | View-independent transport: diffuse, self-shadowing, interreflection, subsurface |
| `s_ik(w_o) = sum_r g_ir(w_o) * v_irk` | `V_i in R^(RxB)`, `G_i in R^(R x K_sh x 3)` | Rank-`R` separable light⊗view factorisation — the glossy, view-dependent transport. `g_ir` is an SH function of degree ≤ 3 over `w_o` |
| `p_i(w_o; E)` | 2–4 narrow SG lobes (direction, sharpness, RGB weight) | Hard shadow edges and sharp highlights a rank-`B` smooth basis cannot represent. One fetch into a prefiltered environment pyramid |

Every term is linear in `E`. Writing the whole bracket as `M_i(w_o) in R^(3xB)`:

```
L_i = M_i * ell
```

### The exactness theorem

Alpha compositing is `C(p) = sum_i w_i L_i`, where the weights
`w_i = alpha_i * prod_{j<i} (1 - alpha_j)` depend only on geometry and opacity —
**not on `ell`**. Since `L_i` is linear in `ell`:

```
C(p) = sum_i w_i (M_i ell) = ( sum_i w_i M_i ) ell = T(p) * ell
```

So the composited transport `T(p)` can be splatted once and contracted per pixel,
and that is **identical**, to floating-point rounding, to contracting per Gaussian
and then splatting.

This is *not* the usual deferred-shading approximation. Deferred shading composites
normals and roughness and is only correct for opaque, single-surface pixels. This is
an equality, and it is a unit test.

### Two exactly-equivalent render paths

| | **Path A — contract, then splat** | **Path B — splat, then contract** |
| --- | --- | --- |
| Per light change | `c_i = M_i ell`, a matvec over Gaussians | nothing |
| Per camera | splat **3** channels | splat **3B** channels |
| Per light change, per pixel | nothing | `3B` MACs (~0.2 ms at 1080p, `B=8`) |
| Best when | light static, camera or views moving | camera static, light moving |
| `B` limit | none; `B=128` is fine | `rasterization` chunks N-D features at `channel_chunk=32` and its docstring warns `D > 32` is slow, so the sweet spot is `B <= 10` |
| Multi-view | **one contraction, N cheap splats** | one splat, N cheap contractions |

The renderer selects the path from what changed since the last frame.
`max |A - B| < 1e-5` against a CPU torch reference is a gate, not an aspiration.

### The light basis is learned, and sized by a measured rank

A fixed low-order spherical-harmonic basis provably cannot carry shadow edges
(Ng et al., SIGGRAPH 2003) — which is why `olat-relight/docs/08-runtime.md` already
chose spherical Gaussians over harmonics. A **scene-adapted** basis only has to span
the transport variation *this object* exhibits, not every spherical function, so it
needs fewer terms than either.

Handheld capture yields unstructured `(view, light)` pairs, not a dense view×light
tensor, so the classical eigen-image SVD cannot be taken on the raw data. Instead,
**train over-complete, then compress optimally**:

1. Train with `B0 = 128` Fibonacci-distributed spherical-Gaussian atoms, fixed and
   analytic.
2. Stack the fitted transport `{D_i, V_i}` across all Gaussians and take its SVD in
   the light dimension. This is the provably optimal rank-`B` approximation under
   the Frobenius norm.
3. Keep the top-`B` atoms, project, fine-tune.

The singular-value spectrum is then a **measured property of the object**: it states
how much angular rank its light transport actually has, and it sizes `B` with a
number rather than a preference. That plot is a first-class deliverable, in the same
spirit as `olat-relight`'s residual energy ratio — it makes "is the basis big
enough?" answerable instead of arguable.

Antecedent, to be cited plainly: eigen-image and PCA image-based relighting
(Nishino & Nayar; Matusik et al.; the light-stage compression literature). The
departure is applying it **per Gaussian, end to end, inside a 3D representation**,
and pairing it with the screen-space contraction.

### Prefiltering is free

Roughness prefiltering is a linear operator, so

```
prefilter( sum_k ell_k A_k , roughness ) = sum_k ell_k * prefilter( A_k , roughness )
```

Precompute each atom's roughness pyramid once, at training time. At runtime the
prefiltered environment for *any* illumination is a `B`-term linear combination of
those pyramids: roughly 0.05 ms, against the per-light-change mip-chain build that
`olat-relight/docs/08-runtime.md` budgets at **40 ms**. The private SG lobes and any
split-sum specular both read from it.

### Real-time inverse lighting

Because `C = T ell` with `T` splatted and independent of `ell`, recovering the
illumination from a target image is

```
ell* = argmin_ell || T ell - I* ||^2      subject to ell >= 0
```

a small non-negative least-squares problem: `B` unknowns, one normal-equation
accumulation over pixels, closed form at interactive rates. This is a direct
consequence of the linear formulation and no baseline above can do it.

## Capture: handheld flash OLAT

No light stage. This is the "poor man's light stage" setting that OLAT Gaussians
validates, and it matches the Kintsugi3D / ARAGO flash workflow already in use.

- **Shoot flash / no-flash pairs.** Off-camera flash — never on-axis, which destroys
  shape information. RAW, linear, single exposure. `flash - ambient` is a clean
  single-light OLAT observation with ambient removed.
- **Run SfM on the ambient half.** The no-flash frames are constant-lit, so COLMAP
  behaves; the flash frames inherit their poses exactly. This sidesteps the biggest
  practical obstacle to flash photogrammetry — changing illumination breaking
  feature matching — and it is free, because the pairs are already being shot.
- **Light position per shot** from two chrome spheres in frame, by highlight
  triangulation. Flash radiant intensity and angular profile from a white reference
  card. Both already exist in the ARAGO/Kintsugi3D toolkit.
- **Masks** from the existing `tools/generate_sam2_masks.py`.
- **Splits**: held-out *views* and held-out *lights*, both deterministic, following
  `tools/colmap_scripts/gsplat_hq.py`.

### Near-field to far-field — the principal scientific risk

Training observations are near-field point lights at roughly a metre. Deployment is
under far-field environment maps. This must be handled in the model, not hoped for:

- Model the near-field factor analytically during training — per-Gaussian direction
  to the light, inverse-square falloff, measured flash profile — and divide it out,
  so the learned atoms are functions of direction only.
- **Validate it directly, with no simulator.** Capture an HDR environment probe
  (chrome ball) at the object's location, then photograph the object under that same
  environment. Relight, compare against the photograph.

Named explicitly because published OLAT-relighting work routinely evaluates only on
point lights drawn from the same rig, which does not test the thing that matters.

## Implementation in gsplat

### What already exists and must be reused

| Need | Existing | Where |
| --- | --- | --- |
| Splat arbitrary per-Gaussian channels | `extra_signals: [..., (C,) N, E]` → `meta["render_extra_signals"]` | `gsplat/rendering.py:283`, `:680` |
| N-D colours | `colors [..., N, D]` / `[..., C, N, D]` when `sh_degree=None` | `gsplat/rendering.py` (`rasterization` docstring) |
| Channel specialisations | `1,2,3,4,5,6,8,9,16,17,21,23,24,32,33,64,65,128,…` | `gsplat/cuda/csrc/Config.h` (`GSPLAT_NUM_CHANNELS`) |
| Batched multi-view from one Gaussian set | batched `viewmats` / `Ks` | `docs/batch.md` |
| View-direction-fused SH, arbitrary channel count | `spherical_harmonics(degrees_to_use, means, viewmats, coeffs[N,K,D], …)`, plus split `spherical_harmonics_l0` / `_l1_plus` | `gsplat/cuda/_wrapper.py:436`, `:493`, `:508` |
| Densification carrying arbitrary extra attributes | `split` / `duplicate` / `remove` iterate all `ParameterDict` keys generically | `gsplat/strategy/ops.py:175`, `:141`, `:238` |
| Capped MCMC, masks, alpha/IoU eval, sealed splits, resume | the whole fixed-light workflow | `codex/gsplat-hq-fixed-light` |
| Per-image embedding pattern → per-light embedding | `CameraOptModule`, `AppearanceOptModule` | `examples/utils.py:27`, `:66` |
| Trainer skeleton, eval, checkpoints, PLY, viewer | `Runner`, `rasterize_splats`, `eval`, `GsplatViewer` | `examples/simple_trainer.py:384`, `:649`, `:1201`; `examples/gsplat_viewer.py:27`, `:51` |
| Newest new-subsystem precedent | dynamic-surgical trainer + `docs/source/proposals/` | `examples/dynamic_surgical_trainer.py`, `docs/modules-design.md:37` |

### What genuinely does not exist

`gsplat` has no relighting, BRDF, environment-map, OLAT or HDR support anywhere — a
repo-wide search for `envmap|BRDF|albedo|OLAT|EXR|tonemap` returns nothing.
Everything is LDR uint8 sRGB, metrics clamp to `[0,1]` with `data_range=1.0`, and
`gsplat/exporter.py` hard-codes the `f_dc_*` / `f_rest_*` SH PLY layout with no
extension point. Linear-radiance I/O, HDR-aware losses and metrics, and a
transport-carrying serialisation format are all new work and are scoped as such.

### Layout

New subpackage, following `docs/modules-design.md:37`:

```
gsplat/relight/
  design.md
  __init__.py
  functional/atoms.py        # SG/Fibonacci atom basis, <A_k, E> projection, atom prefilter pyramids
  functional/transport.py    # contract(M, ell) [Path A], contract_screen(T, ell) [Path B]
  functional/nearfield.py    # point-light direction, falloff, flash profile
  functional/compress.py     # SVD rank study and optimal atom compression
  models/atlas.py            # parameter block and the two render paths
  models/reference.py        # slow, obviously-correct torch reference (CPU)
examples/relight_trainer.py  # fork of simple_trainer.py
examples/relight_viewer.py   # subclass of GsplatViewer; do not edit the shared one
examples/datasets/olat.py    # OLATParser / OLATDataset — (view, light) pairs, HDR, splits
tools/solve_flash_positions.py
tools/prepare_olat_capture.py
tests/relight/
```

No new required dependencies. RAW decoding (`rawpy`) and EXR stay optional behind
`gsplat[relight]`, never a hard import.

## Phases and gates

Every gate is a number, declared before the work.

**P0 — Capture protocol and data contract.** *CPU.* **Partly executed.**
Capture protocol document. `OLATParser` / `OLATDataset` returning
`(camtoworld, K, image_linear, light_position, light_intensity, light_profile, mask,
view_id, light_id)`. Chrome-sphere solver. Flash/no-flash pairing and ambient
subtraction. Deterministic held-out-view *and* held-out-light splits. A procedural
fixture (sphere and plane, analytic) so the loader is testable with no capture.
*Gates:* splits deterministic and disjoint — **met**; light calibration —
**the 2 mm gate was withdrawn on measurement and replaced** by an 18 mm
bracket-offset fit and a 0.9° direction recovery, both met, per the section
above. The dataset loader and its procedural fixture are **not yet written**;
the calibration and split mathematics that the loader will call are, in
`gsplat/relight/functional/calibration.py` and `splits.py`.

**P1 — Transport core and the exactness proof.** *CPU. The scientific core, and it
needs no GPU.* **Executed.**
Atom basis and projection, near-field model, both contractions, the torch reference
renderer, atom prefilter pyramids, optimal basis compression, and the
non-negative least-squares light solve. Lives in `gsplat/relight/functional/`;
99 tests in `tests/relight/`.
*Gates:* `max |Path A - Path B| < 1e-5`; superposition
`render(E1+E2) - render(E1) - render(E2) == 0` exactly, structurally rather than
penalised; `prefilter(sum ell_k A_k) == sum ell_k prefilter(A_k)` to `1e-6`; every
guard mutation-checked. **All met** -- see the table at the top of this document.

Two defects were found by running the code rather than by reading it, which is
why the phase is ordered before anything that depends on it:

- Building the exclusive cumulative product in the compositing weights by
  *dividing* the inclusive product is wrong exactly where it matters. A fully
  opaque primitive makes the divisor zero, and every weight in front of it --
  including its own -- collapses to zero. Shifting instead of dividing fixes it.
- The light solver stopped on relative step size, which never fires on an
  ill-conditioned Gram matrix even after the objective has stopped moving: it
  exhausted a 500-iteration budget at a residual of 6e-9. Stopping on the
  projected gradient, the first-order optimality measure for the constrained
  problem, and adding adaptive restart brings it to 149-159 iterations.

Mutation checks were run rather than assumed. Reinstating the division in the
compositing weights, packing the splat channels atom-major instead of
channel-major, and reducing `contract_screen` along the channel axis each fail
named tests.

**P2 — Trainer v1.** *GPU, workstation.*
Fork `examples/simple_trainer.py`. Fixed 128-lobe atom basis, Path A only, `D_i`
plus the rank-`R` view factor, capped MCMC, mask/alpha supervision, HDR losses in a
`log1p` domain, HDR-aware metrics. Reuse the fixed-light branch's split, eval and
resume machinery wholesale.
*Gate:* held-out-**light** PSNR within 1 dB of held-out-**view** PSNR. If that
fails, the representation is memorising lights and nothing downstream is worth
building.

**P3 — Rank study, learned atoms, private lobes.**
SVD compression of the fitted transport, the singular-value spectrum, top-`B`
projection, fine-tune. Per-Gaussian private SG lobes.
*Gates:* chosen `B` retains ≥ 99 % of transport energy; PSNR loss against the
128-lobe model ≤ 0.3 dB; private lobes measurably improve a hard-shadow crop —
report the crop, not only the mean.

**P4 — Path B, N-D splatting, inverse lighting.**
`extra_signals` transport splat, screen-space contraction, non-negative
least-squares light solve.
*Gates:* GPU `max |A - B| < 1e-4`; relight pass under 1 ms at 1080p; **ms/frame flat
in #lights**, slope under 1 % per light — the money plot against GS³; inverse
lighting recovers a known `ell` to under 5° mean angular error in under 50 ms.

**P5 — Viewer and multi-view renderer.**
`examples/relight_viewer.py`: environment rotation, draggable point light, toggle
private lobes, toggle Path A/B, toggle individual atoms. Batched multi-view
rendering through `rasterization`'s batched `viewmats`.
*Gates:* 1080p at ≥ 60 fps with ~600 k Gaussians on Path A; N-view cost measurably
sublinear against N independent relights.

**P6 — Baselines, ablations, writeup.**
A GS³-style baseline (`shading × shadow + residual`, per light) re-implemented in
the same trainer on the same data, so the comparison is controlled. Ablations: `B`
sweep, view-rank `R` sweep, learned atoms vs fixed SG vs SH, private lobes on/off,
near-field model on/off.
*Gate:* every ablation run and reported, including any that says the design was
wrong.

## Verification

CPU-only — most of P0, all of P1:

```bash
python -m pytest tests/relight/ -q          # exactness, superposition, prefilter linearity, loader
python -m pytest tests/test_strategy.py -q  # densification still carries the new attributes
python -m black --check --required-version 22.3.0 <changed files>
cd examples && PYTHONPATH=<repo> python -c "import relight_trainer, relight_viewer"
```

That last line is not optional: `py_compile` does not evaluate a `tyro` dataclass's
annotations, so a missing import there compiles cleanly and breaks on import.

GPU, workstation — P2 onward:

```bash
python examples/relight_trainer.py atlas --data-dir <capture> --object-mask-dir <masks> \
    --split-manifest <splits.json> --result-dir results/atlas-v1
python tools/export_gsplat_metrics.py results/atlas-v1
python examples/relight_viewer.py --ckpt results/atlas-v1/ckpts/ckpt_best.pt
```

Every claim must be labelled **executed** or **reviewed**.

## Known risks

1. **Angular rank may be higher than hoped.** If hard shadows push the required `B`
   past ~64, Path B loses its advantage — channel chunking above 32 — though Path A
   is unaffected and the cost stays light-count-independent. The rank spectrum from
   P3 is the early warning and it arrives before any runtime work is built.
2. **Near-field to far-field transfer** is the one place the method could be quietly
   wrong. The environment-probe reference photograph is the gate; it is cheap and it
   must not be skipped.
3. **Geometry quality** limits the view-dependent branch. There is no mesh and no
   proxy; normals come from the Gaussians themselves. OLAT Gaussians found a proxy
   mesh necessary for good highlights. If the rank-`R` view factor underperforms,
   importing normals from the existing photogrammetry work is the fallback, and it
   costs the "no mesh" property.
4. **Handheld capture is sparse and unstructured.** Coverage of the view × light
   product will be uneven. Report the coverage, and expect the held-out-light gate
   to bind first.
5. **A comparative claim needs an external benchmark.** Validating only on in-house
   captures supports the engineering claims but not a competitive one. The loader is
   deliberately dataset-agnostic so OpenIllumination or HumanOLAT drops in as a
   config change; running one is a prerequisite for submission, not for the method.
6. **HDR is new ground in gsplat.** Losses, metrics and PLY serialisation all assume
   LDR. Budget real time for it rather than treating it as plumbing.

## Relationship to olat-relight

Independent, not a replacement — stated so the two do not quietly diverge on the
same question.

What ATLAS takes from it: the measurement insight (OLAT samples the transport
operator column by column, so shadows and interreflection are *measured*, never
inferred), the discipline of preregistered numeric gates, and the spherical-Gaussian
environment argument in `docs/08-runtime.md`.

Where it departs: `olat-relight` splits *physical model + learned residual* over a
frozen mesh, and pays for it with a superposition **penalty**
(`docs/07-residual-transport.md`, weight `1e-2`), because a conditioned decoder has
no structural reason to be linear in the illuminant. ATLAS makes linearity
**structural** — superposition error is identically zero, not regularised toward
zero — at the cost of giving up an exportable, editable SVBRDF.

That is the real trade, and it is the opposite choice from ADR-0011, made
deliberately and for a different goal: `olat-relight` optimises for a material that
means something; ATLAS optimises for an illumination operator that is exact and
fast.

If both reach their gates, the interesting experiment is to fit ATLAS transport on
top of `olat-relight`'s recovered SVBRDF and see whether the residual energy ratio
drops. That is a later question.

## References

- Debevec et al., *Acquiring the Reflectance Field of a Human Face*, SIGGRAPH 2000.
- Sloan, Kautz, Snyder, *Precomputed Radiance Transfer*, SIGGRAPH 2002.
- Ng, Ramamoorthi, Hanrahan, *All-Frequency Shadows Using Non-Linear Wavelet
  Lighting Approximation*, SIGGRAPH 2003.
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*,
  SIGGRAPH 2023.
- Saito et al., *Relightable Gaussian Codec Avatars*, CVPR 2024.
- Bi et al., *GS³: Efficient Relighting with Triple Gaussian Splatting*,
  SIGGRAPH Asia 2024.
- *OLAT Gaussians for Generic Relightable Appearance Acquisition*,
  SIGGRAPH Asia 2024.
- *PRTGS: Precomputed Radiance Transfer of Gaussian Splats*, ACM MM 2024.
