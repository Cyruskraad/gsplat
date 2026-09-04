# Plan: sharp texturing via per-face view selection + seam levelling

**Status:** in progress — steps 0, 1 and 2 are landed, **start at step 3**.
**Branch:** `claude/photogrammetry-techniques-plan-jb0pod` (PR #3).
**Read first:** [`photogrammetry_status.md`](photogrammetry_status.md) for the
pipeline's overall state, ground rules, and blockers.

| Step | What | State |
|---|---|---|
| 0 | Module split: `texturing.py` / `mesh_extraction.py` / `_open3d.py` | **done** (`f2c1011`) |
| 1 | `face_view_quality`, `_face_adjacency`, `select_views_mrf` | **done** (`385099c`) |
| 2 | View-selected atlas bake + `atlas_sharpness` metric | **done** |
| 3 | Global seam levelling + `seam_discontinuity` metric | **next** |

---

## Context

Every texturing path in the pipeline — per-vertex colours and the UV atlas
alike — builds each sample by **blending every view that sees it**, weighted by
view/normal alignment and inverse distance.

Blending is the wrong operator for sharpness. Two views of the same surface
point are never registered to sub-pixel accuracy after real SfM, so averaging
them is a low-pass filter: the texture comes out systematically blurrier than
any single source photograph. The iterative sigma-clipping in `693d07e` fixes
*outliers* (a pedestrian, a specular highlight) but cannot fix this — the
surviving inliers still get averaged, and averaging slightly-misaligned inliers
is exactly what destroys high-frequency detail.

Production photogrammetry texturers solve this the same way (Waechter et al.,
*Let There Be Color!*, ECCV 2014): choose **one** view per face, so each face is
textured from a single un-averaged photograph, then remove the colour
discontinuities where neighbouring faces chose different views.

**Outcome:** `--texture_mode atlas --texture_view_selection` produces a
markedly sharper atlas than blending, with seams levelled away, measurable via
two new metrics in `pipeline_report.json`.

## Premise, measured — this dictates the success metric

Probed before the plan was written, because the obvious framing is wrong.
Synthetic sphere, 24 ray-traced views, cameras rotated by a small random angle
to simulate residual SfM pose error, textured with a known analytic pattern:

| pose error | blended: contrast retained | best-view: contrast retained |
|---|---|---|
| 0′  | 96.5% | 100.3% |
| 20′ | 78.6% | 102.4% |
| 45′ | **52.9%** | **104.9%** |

Blending destroys **half the high-frequency detail** at 45′; single-view
sampling retains all of it. That is the justification for this work.

> **Re-measured in step 2, and the caveat is load-bearing:** these numbers
> only appear when the pattern's detail is near the misregistration scale.
> With `tests/test_mesh_extraction.py`'s `_surface_pattern` (wavelength ~half
> the sphere) blending retains 98–100% of the gradient out to 90′ and the
> effect is invisible. `tests/test_texturing.py::_high_frequency_pattern`
> exists for this reason. Re-measured at that frequency: blended **59%**,
> view-selected **106%**, L1 0.171 vs 0.199.

**But per-pixel L1 error goes the other way** — at 45′, blended 0.152 vs
best-view 0.185 — and on a *smooth* pattern blending wins at every error level
tested. Blending *attenuates* detail while single-view *displaces* it, and a
displaced-but-sharp texture scores worse pointwise than a blurred one even
though it looks far better.

Two consequences, both binding:

1. **The success metric is contrast/gradient retention, not error against
   ground truth.** A test asserting "view selection has lower L1 error" would
   fail and make the feature look worthless. This is the third instance on this
   branch of a plausible-looking test measuring the wrong thing (cf. the sphere
   normal-map test and the whole-frame contamination test).
2. **View selection stays opt-in, not the default.** It is a genuine tradeoff —
   sharper, but pointwise less accurate — and which one a user wants depends on
   whether the asset is for viewing or for measurement. Say so in the docs
   rather than selling it as strictly better.

## Constraints

- No GPU, no CUDA extension, no capture data; **CI cannot run**. Everything
  must be CPU-verifiable by hand.
- **No new required dependencies.** `gsplat[mesh]` is deliberately just
  `open3d` + `imageio`, and an earlier change removed a `scipy` dependency on
  purpose. The MRF optimiser and the linear solve are pure NumPy. (scipy is
  installed and has `maximum_flow`/`cg`, and is in the `lidar` extra —
  acceptable only as an optional accelerator, never a hard import.)
- `black==22.3.0`, `py_compile`, **plus `import` of each changed example
  script** — `py_compile` does not evaluate `tyro` dataclass annotations, and
  that gap already let a `NameError` through once.
- Commits and the PR stay explicit about what was executed vs. reviewed.

## Verified facts this design rests on

Established empirically — not assumptions:

- **Per-texel face id is recoverable.** Cast from `position + normal*eps` along
  `-normal` onto the mesh; `cast_rays()["primitive_ids"]` gives the face.
  Measured 100% recovery on a sphere, all 360 faces hit, mean texel-to-centroid
  distance 0.089 against a face size of 0.33.
- open3d's `cast_rays` barycentric weights are `(1-u-v, u, v)`.
- `compute_uvatlas` is **non-deterministic** and **segfaults** on non-manifold
  input. Both are guarded in `_unwrap_and_rasterize`, which reuses a mesh's
  existing `triangle_uvs` when present — any new map must ride that same layout.
- No graph-cut/MRF/seam/sparse-solver code exists elsewhere in the repo.

---

## Step 0 — module split — **DONE** (`f2c1011`)

`gsplat/photogrammetry/texturing.py` holds everything that samples views or
writes an atlas; `mesh_extraction.py` keeps surface reconstruction and
decimation; `_open3d.py` holds the shared `_require_open3d` guard so neither
depends on the other. `mesh_extraction.py` re-exports the moved names, because
the example CLIs and the test suite import bakers from that path.

Verified as a pure move by AST comparison of every top-level definition: 19
before, 19 after, none missing, added or changed; suite 85 → 85 with no test
edited.

## Step 1 — quality + MRF labelling — **DONE** (`385099c`)

In `texturing.py`:

- `face_view_quality(mesh, dataset, max_views=None) -> (F, V)` — gradient
  energy over each face's projection (Waechter's data term: folds "close and
  fronto-parallel" and "in focus" into one number). Visibility reuses
  `_view_samples`, so occlusion is the same ray cast the blended bakes use. A
  summed-area table of each view's gradient magnitude makes the footprint
  statistic O(1) per pair.
- `_face_adjacency(triangles) -> (E, 2)` — canonical edge keys + `np.unique`.
- `select_views_mrf(quality, adjacency, smoothness, max_iterations, max_seeds)`
  minimises `Σ_f -log(quality[f, l_f]) + smoothness · (seam count)`, returning
  `(labels, stats)`. `NO_VIEW = -1` marks faces no view can texture.
- `_view_samples` now also yields the view index.

**Two findings worth keeping in mind for the remaining steps:**

- Single-seed ICM is badly seed-dependent: with a strong smoothness term the
  first face to move can cascade every other face onto its neighbour's label
  (measured energy +4.14 where −4.39 was available). It now runs from several
  seeds — greedy, plus "every face takes view α" for the strongest few views —
  and keeps the lowest energy. Do not remove the extra seeds; a test pins the
  energy gap.
- Unusable views get `+inf`, not a large finite cost. A finite penalty can be
  outweighed by enough smoothness pressure and would texture a face from a
  camera that never saw it.

---

## Step 2 — view-selected bake + `atlas_sharpness` — **DONE**

`bake_texture_atlas_view_selected` in `texturing.py` returns
`(mesh, texture, stats)`; `bake_mesh_texture` reaches it via
`view_selection=True` plus a `stats_out` dict (an out-parameter, because
several callers unpack its `(mesh, texture)` pair). `atlas_sharpness` is in
`metrics.py`; `--texture_view_selection` / `--texture_mrf_lambda` are on
`examples/extract_mesh.py`, and the stats land in `mesh_metrics.json` under
`"view_selection"`.

**The premise measurement had to be redone, and the redo matters.** The
numbers in the table above do *not* reproduce with
`tests/test_mesh_extraction.py`'s `_surface_pattern`: its wavelength is about
half the sphere, far coarser than the few pixels a pose error displaces a
projection by, so blending it loses nothing (blended gradient retention stayed
at 98-100% out to 90 arcminutes). The effect only appears once the pattern's
detail is near the misregistration scale. `tests/test_texturing.py` now
defines `_high_frequency_pattern` (wavelength ~0.14 world units) for exactly
this, and `_SphereDataset` gained `pattern=` and `pose_error_arcmin=`
arguments. At that operating point the original measurement reproduces
closely: blended retains **59%** of the ground truth's gradient, view-selected
**106%**, while pointwise L1 goes the other way (0.171 blended vs 0.199
selected).

**Two bugs found and fixed while building it** (both mutation-checked):

- `_gradient_summed_area` accumulated the summed-area table in **float32**,
  because that is the dtype training images arrive in. Over a 512x512 image
  that leaves box readouts wrong by 0.087 against a table maximum of ~1e5 --
  four orders of magnitude worse than the 1.1e-5 the input rounding alone
  costs, and the error lands directly in the (face, view) quality scores.
- `_box_means` could return a **negative** mean of gradient magnitudes, from
  four-corner cancellation on a near-empty box. That reaches `-log()` in
  `select_views_mrf` as `NaN`, and `np.argmin` returns a `NaN`'s index in
  preference to every real cost -- so the face would be textured from the one
  view that cannot see it. It also breaks the seed search, since every
  comparison against `NaN` is False. Now clamped at zero.

### What was executed

Full suite **110 passed** (98 before). Delivery path run end to end on CPU:
3480 -> 400 triangles, view-selected albedo + normal + AO on one shared UV
layout, OBJ/MTL/PNGs written and read back, `mesh_metrics.json` round-tripped.
Mutations checked: dropping the `_box_means` clamp, dropping the float64
promotion, blacking out the fallback, ignoring the MRF labels, ignoring the
per-texel face id, and disabling view selection entirely -- each fails a test.
The `--texture_view_selection` CLI guard is placed beside the existing
`--normal_map` guard and was **verified by code review only**: reaching it
needs a real checkpoint.

<details>
<summary>Original step 2 design (for reference)</summary>

### Bake from labels

Per atlas texel: recover its face id (the validated ray cast above) → that
face's label → project into that one view → sample. No averaging anywhere.

- Texels whose face is `NO_VIEW`, and texels outside every chart, fall back to
  the existing **blended** result from `bake_texture_atlas`. The blend is
  computed anyway and is the correct answer where no single view is usable.
- Then `_fill_texture_holes` as today, so seam dilation behaviour is unchanged.
- Must reuse `_unwrap_and_rasterize` so the albedo shares its UV layout with
  the normal and AO maps.

### Metric (`gsplat/photogrammetry/metrics.py`)

`atlas_sharpness(texture, covered_mask)` — mean gradient magnitude over covered
texels, plus texel-colour standard deviation (contrast). These are the numbers
that should *rise* when view selection replaces blending, and per the measured
premise they are the **only** ones that will; pointwise error will not improve,
by design.

### CLI (`examples/extract_mesh.py`)

`--texture_view_selection` (requires `--texture_mode atlas`; error clearly
otherwise, matching the existing `--normal_map` guard) and
`--texture_mrf_lambda`. View selection and `--texture_outlier_sigma` are
**alternatives, not exclusive**: robust blending still governs fallback regions.

### Tests (`tests/test_texturing.py`)

1. **Detail retention beats blending.** Reuse `_SphereDataset` but render a
   *high-frequency* pattern (a smooth pattern cannot show this effect at all),
   and perturb each view's `camtoworld` by a small rotation.
   *Premise to assert first:* blending must measurably lose contrast — assert
   retained contrast falls below ~70% of ground truth at the test's
   perturbation. Without that the test proves nothing.
   *Assert:* the view-selected atlas retains ≳95% of ground-truth contrast
   where the blended one retains ≲70%, and its `atlas_sharpness` is higher.
   *Explicitly assert the tradeoff too* — that view selection's pointwise L1
   error is **allowed to be worse** — so a later reader cannot "fix" this test
   by flipping it to an error comparison and conclude the feature is broken.
2. **No-view fallback.** A region no camera sees comes back with the blended
   colour, never a hole and never black.
3. **Shared UV layout.** View-selected albedo, normal and AO maps must report
   identical `triangle_uvs`.

</details>

## Step 3 — seam levelling + `seam_discontinuity` — **START HERE**

After labelling, a vertex on a seam has two colours depending on which side you
sample. Solve for an additive correction `g` per **(vertex, label)** pair — not
per vertex; a single per-vertex offset provably cannot fix a discontinuity *at*
that vertex, since both sides would receive it.

Minimise

```
E(g) = Σ_(v, l1, l2) ‖ (c[v,l1] + g[v,l1]) − (c[v,l2] + g[v,l2]) ‖²   (seam mismatch)
     + λ_s Σ_(v,w) edge of a face labelled l  ‖ g[v,l] − g[w,l] ‖²     (smoothness)
```

first term over seam vertices whose incident faces carry both `l1` and `l2`;
`c[v,l]` is the colour view `l` sees at vertex `v`. Solve the normal equations
`(AᵀA + λ_s L) g = Aᵀb` with a **hand-rolled conjugate gradient** (~30 lines)
applying `AᵀA` as a matvec, never materialising the matrix. It is symmetric
positive semi-definite; **anchor the gauge** by fixing the mean of `g` to zero
— the energy is invariant to a global shift, so without an anchor CG drifts.
Apply `g` to the atlas by barycentric interpolation from each face's three
`(vertex, label)` corrections; solve the three channels independently.

`seam_discontinuity(mesh, texture, labels, triangle_uvs)` — mean colour
difference sampled either side of label boundaries; should *fall* after
levelling. CLI: `--texture_seam_smoothness` (λ_s), reporting `num_seams` and
`seam_discontinuity` before/after, warning if levelling failed to reduce it.

### Tests

4. **Levelling removes seams.** Give each view a distinct constant exposure
   offset, so the true colour is unambiguous but each view reports it shifted.
   *Premise:* `seam_discontinuity` before levelling must be clearly non-zero.
   *Assert:* it drops by a large factor after levelling, **and** the levelled
   atlas is no further from the mean-exposure ground truth than before (i.e.
   levelling removed the steps without introducing a global colour cast).
5. **CG solver** tested directly against a small dense system solved with
   `np.linalg.solve`.

---

## Verification standard

Follow the pattern that has found 11 bugs on this branch: synthetic scenes
whose correct answer is known **independently**, each test asserting its own
premise before measuring, and every guard mutation-checked (revert the fix,
confirm the test genuinely fails).

Plus, end to end by hand: the full delivery path (decimate → view-selected
albedo → normal → AO → OBJ/MTL/PNGs) written and read back, as already done for
the blended three-map path.

## Risks

- **ICM local minima** on scenes with many near-equal-quality views: mitigated
  by the multi-seed search and by seam levelling; the interface is kept
  optimiser-agnostic so a graph cut can be dropped in behind it.
- **Cost.** Quality evaluation is `F × V`; the extra atlas passes and the CG
  solve add up. Reuse the existing chunking and keep `max_views`.
- **CG conditioning** if λ_s is too small — anchor the gauge, cap iterations,
  and report non-convergence rather than returning a silently bad correction.
- **Sub-texel face-id error** at chart borders, where the margin extrapolates
  positions off-surface. Those texels already take the dilation path; make sure
  they take the fallback rather than a wrong face's label.
- **Verification ceiling:** none of this can be judged on a real capture here.
  Both metrics are designed so a later GPU run produces the numbers that would
  settle it.
- **The feature may not be worth defaulting on.** The measured premise says it
  trades pointwise accuracy for detail. If a real capture's poses are
  well-registered enough that blending retains most contrast, the honest
  outcome is that this stays a niche option — and these metrics are what would
  show that. State it in the PR rather than discovering it later.
