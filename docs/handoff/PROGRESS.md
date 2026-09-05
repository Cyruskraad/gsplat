# Progress

## Executed vs reviewed — read this before trusting anything below

There is **no CI on this fork** and **no GPU, CUDA extension, `colmap` binary or
capture data** in the environment this was built in. So the split below is not
pedantry; it is the difference between what is known to work and what merely
looks right.

### Executed, on CPU, with real `pycolmap`/`open3d`/`scikit-learn`/`opencv`

- **168 tests pass** (153 before this session). Every guard
  mutation-checked — the fix reverted, a test
  confirmed to genuinely fail.
- **Bundle adjustment**, non-dry-run, against a synthetic COLMAP model with
  perturbed points: removed **96.5%** of the reprojection error.
- **The `priors` gate** in all three modes; the summariser against real
  `stats/*.json`; `run_pipeline.py --dry_run` command construction.
- **The full delivery path, end to end, four times**, each writing OBJ + MTL +
  PNGs and reading them back:
  - blended, 9024 → 400 triangles;
  - view-selected + seam-levelled, 3480 → 400, seam discontinuity 0.262 → 0.090;
  - fit-target decimated 6240 → 142 with a **16-bit** normal map read back
    byte-identical;
  - every sizing decision measured: 6960 faces → cull 3480 → fit-decimate 592 →
    atlas 512 chosen from the evidence.
- **`examples/extract_mesh.py`'s `main()` itself**, against a synthetic capture
  written to disk by the new `examples/make_synthetic_capture.py` — with the
  full delivery flag set (`--cull_unobserved --texture_mode atlas
  --texture_view_selection --texture_seam_smoothness --target_fit_ratio
  --normal_map --normal_map_bits 16 --ao_map`), reading the OBJ/MTL/PNGs back.
  This is what converts the "reviewed only" block below from ~ten guards to
  two, and it is the pin that reproduces bug 14: reverted, it fails with the
  original `TypeError`.
- **`run_pipeline.py` non-dry-run**, delivering a textured mesh through the
  `extract_mesh` stage with no checkpoint and no GPU.
- **Photometric camera refinement** (Zhou & Koltun 2014), on views ray-cast
  from the same mesh it refines against, at 45' of injected pose error:
  photometric residual **0.145 → 0.041** (3.5x), pairwise pose disagreement
  **55.9' → 24.8'** (2.3x). Its effect on texturing, on the larger fixture:
  blending's contrast retention **59.1% → 92.5%** against a
  perfectly-registered ceiling of 92.9%, atlas L1 **0.180 → 0.057**, and
  view selection's L1 **0.229 → 0.053** — which *flips the L1 ranking* to the
  one exact poses produce. The headline tradeoff is a symptom of
  misregistration, and this is it going away, measured.
- **Three claims this session set out to confirm and instead falsified**, each
  recorded in `ISSUES.md` § 4 rather than quietly dropped: open3d's shipped
  implementation of the same algorithm (worse in every configuration, including
  on exact poses); "single-scale will not recover 45'" (false at equal work);
  and "blending retains >= 95% after refinement" (above the ceiling — exact
  poses reach ~93%).
- `black --check --required-version 22.3.0`, `py_compile`, and an `import` of
  every changed example script.

### Reviewed by code inspection only

- The GPU-only half of `extract_mesh.py`: `--method tsdf` (which renders depth
  maps from a checkpoint) and the `--texture_texels_per_pixel` clamp warning,
  which needs an atlas larger than the cap. Everything else in that script now
  runs in `tests/test_extract_mesh_cli.py` — see the executed list above; the
  `assert cfg.ckpt` that used to gate all of it is gone.
- The GPU stages themselves: `train`, `extract_mesh`, `dense_mvs` end to end.
- Both AI-prior recipes against a *real* model. The Mask R-CNN recipe's full API
  path was executed with a randomly-initialised model and its output loaded
  through the real `Dataset`; pretrained weights cannot be fetched here.
- **The `core_tests.yml` change has never run.** Simulated locally: without it,
  four of the six photogrammetry files skip wholesale and only 26 tests run.

### Deliberately *not* mutation-checked, and why

Recorded so nobody assumes coverage that does not exist. Each says so in its own
docstring too.

| Guard | Why not |
|---|---|
| `_conjugate_gradient(project_mean=...)` gauge anchor | Removing it changes the seam solve by <1e-9: every row is a difference of two unknowns, so the RHS is already orthogonal to the constants and CG from zero never leaves the range space. Kept against rounding on a larger system. |
| Returning the *smallest* feasible decimation probe rather than the last | An optimality refinement, not correctness — every candidate is feasible *by measurement*. On well-behaved input the two coincide and no synthetic scene here separates them. |
| The `minimum` clamping the bilinear sampler's `x1`/`y1` | Unreachable: the coordinate clip already forces the fractional weight to zero at the border, so wrapping instead gives identical results. Kept so the intent survives a later change to the clip. |

---

## The 16 bugs found and fixed

Each was confirmed by reverting the fix and showing a test genuinely fails.

1. **`refine_reconstruction` pose-writing broken on modern pycolmap.**
   `Image.cam_from_world` is read-only under the newer rig/frame model; the
   direct-assignment path raised `AttributeError`. Version-tolerant write.
2. **`--mono_depth_loss` alone rendered the wrong depth flavour.** `render_mode`
   only checked `cfg.depth_loss`, so raw accumulated depth ("D") was used
   instead of alpha-normalised expected depth ("ED") — silently scaling every
   pixel by its own local alpha before a correlation loss.
3. **`mono_depth` misaligned under lens distortion.** Resized to the
   post-undistortion shape but never put through the same remap/ROI-crop; 97%
   mismatch without the fix.
4. **`merge_point_maps_to_tracks` chaining.** Single-linkage clustering could
   transitively chain far-apart points, contradicting its own documented
   merge-radius guarantee. Switched to complete-linkage.
5. **Fisheye ROI mask never patch-cropped** (pre-existing), silently mismatching
   the image shape under `--patch_size`.
6. **`run_pipeline.py` threw away its report on failure.** A failing stage
   re-raised out of `main()` before the report was written — and a stale
   `pipeline_report.json` from an earlier run stayed on disk still claiming
   success. Found by actually running the new gate under `--strict`.
7. **`(1, H, W)` monocular depth maps silently corrupted.** That is the shape a
   transformers depth pipeline's `predicted_depth` commonly has, and what this
   project's own documented recipe produced. `cv2.resize` read it as a one-row
   image with W channels and returned `(H, W, W)` **without raising** — a whole
   training run supervised against reshaped noise.
8. **`point_to_mesh_distance` on an empty mesh** died with open3d's
   `IndexError: _Map_base::at`, at the very end of a long GPU run.
9. **`point_cloud_stats` on an empty cloud** raised numpy's bare "zero-size
   array to reduction operation minimum". Relatedly `point_to_mesh_distance`
   with no points now returns `None`, not `0.0` — "nothing measured" and
   "perfect fit" must not look alike to a metric that divides by it.
10. **`compute_uvatlas` is not deterministic** — one mesh, four unwraps, four
    layouts. Albedo and normal map would have had *different* UV layouts, so
    the normal map would be sampled through the albedo's coordinates.
11. **AO against a separate occluder mesh was uniformly wrong.** 80% of a
    decimated mesh's texels lie *inside* the dense mesh it came from, so rays
    started under the occluder and hit instantly: mean AO 0.204 instead of ~0.98.
12. **The view-selection summed-area table was accumulated in float32** — the
    dtype training images arrive in. Over a 512×512 image that leaves box
    readouts wrong by **0.087** against a table maximum of ~1e5.
13. **`_box_means` could return a negative mean of gradient magnitudes** from
    four-corner cancellation. That reaches `-log()` as `NaN`, and `np.argmin`
    returns a `NaN`'s index in preference to every real cost — so the face
    would be textured from the one view that *cannot see it*.
14. **The delivery CLI could not run at all.** `examples/extract_mesh.py`
    passed `seam_smoothness=` to `bake_mesh_texture()`, which had no such
    parameter and no `**kwargs`. `bake_texture_` defaults to `True` and the
    kwarg went unconditionally, so **every texture-baking run of
    `extract_mesh.py` — and therefore `run_pipeline.py`'s whole delivery stage —
    died with `TypeError`.** Seam levelling was reachable from the library but
    not from the CLI. It shipped in `fa70683` and nothing caught it because **no
    test had ever called `extract_mesh.main()`**: the script opened with an
    `assert cfg.ckpt` that ran *before* the method dispatch, so no test could
    reach `main()` without a trained GPU checkpoint. Fifth instance of the
    failure mode in `ISSUES.md` § 5, and the first that was a hard crash.
15. **Geometry read off disk was never mapped into the dataset's frame.**
    `extract_mesh.py` builds its `Parser` with `normalize=True`, which rescales
    and reorients the world and hands out every pose in *that* frame; the
    `--method poisson` path read `--dense_points` with `o3d.io.read_point_cloud`
    and used it raw. `Parser` applies exactly this transform to its own
    `dense_points_path`, so the two disagreed. Measured on the synthetic
    capture the frames differ by ~3.4x in scale plus a rotation, which puts
    every camera *inside* the reconstructed mesh. Pre-existing, and found only
    by running the path.

16. **A rotation delta that does not carry its translation moves the camera
    centre.** `bundle_adjustment._optimize`'s `R = exp(δ)·R₀`, `t = t₀ + Δt`
    is fine there — reprojection BA's translation is always free to compensate
    — but reused as-is for a photometric solve it swings the optical centre by
    |t₀|·δ, which at |t₀| ≈ 3.5 and 45' is 0.046 world units: a third of the
    texture's wavelength. The solve then made registration **worse**
    (62' → 155'). Fixed by rotating `t₀` with the delta. Found by measuring the
    outcome rather than the objective — the residual fell throughout.

Two sharp edges are **guarded rather than fixed**, because they are upstream:
`compute_uvatlas` segfaults (exit 139) on non-manifold input, so manifoldness is
checked up front; and 8-bit normal maps have a hard `2/255` quantization floor,
now answerable with `--normal_map_bits 16`.

---

## Commit history (33 commits, oldest first)

| # | Commit | What |
|---:|---|---|
| 1 | `f000692` | The initial subpackage: bundle adjustment, dense MVS, mesh extraction |
| 2 | `2c134b8` | Satisfy the repo's pinned `black==22.3.0` gate |
| 3 | `01a86d5` | `--extract_mesh` on `simple_trainer_2dgs.py` |
| 4 | `d615367` | AI-assisted: monocular depth priors, neural-SfM import |
| 5 | `77714af` | Three bugs found by self code review |
| 6 | `56134c2` | Automatic quality metrics |
| 7 | `0bb996e` | Transient/dynamic-object masking |
| 8 | `1c1a63d` | End-to-end integration with per-stage metrics |
| 9 | `c822085` | The first handoff status document |
| 10 | `c22e031` | UV-atlas texture baking |
| 11 | `5f0ef00` | The `priors` stage becomes a real quality gate |
| 12 | `a181d65` | Make CI actually exercise the suite (still unrun) |
| 13–14 | `f6a3a48`, `70239bc` | Diff size correction; cross-stage derived metrics |
| 15 | `5b4e9d9` | Fix silent `(1, H, W)` depth-prior corruption |
| 16 | `b46149a` | One-command pipeline reaches UV-atlas texturing |
| 17 | `221a474` | Report degenerate mesh/cloud input clearly |
| 18 | `87c870c` | Quadric decimation + normal-map baking (the delivery path) |
| 19–20 | `b5ece59`, `b8c5569` | Diff size; ambient occlusion completes the map set |
| 21 | `693d07e` | Robust multi-view fusion (iterative sigma clipping) |
| 22 | `f2c1011` | Split texturing out of mesh_extraction (verified a pure move by AST) |
| 23 | `385099c` | Per-face view selection: quality term + MRF labelling |
| 24 | `7f7c7e6` | The texturing plan committed as a repo handoff doc |
| 25 | `6735775` | Bake the atlas from one chosen view per face |
| 26 | `fa70683` | Seam levelling |
| 27 | `abe94b3` | Decimate to a fit target instead of a triangle count |
| 28 | `5733a2a` | 16-bit normal maps |
| 29 | `92554d9` | Bilinear source sampling |
| 30 | `82e770a` | Cull geometry no camera ever saw |
| 31 | `4c46599` | Size the atlas from the evidence |
| 32 | `a3108a9` | Multi-page atlases |
| 33 | `b342d5d` | One-command path reaches the whole delivery path |

## Headline measurements

Kept together because they are the evidence behind the design choices, and
because several of them are counter-intuitive.

| Feature | Measurement |
|---|---|
| Robust fusion | Mean colour error vs ground truth **3x** better (0.045 → 0.015); up to 9x with more views. A single clipping pass improved it by only 4% — the bad samples inflate the spread they are measured against. |
| View selection | Detail retained **59% → 106%** at 45′ of simulated pose error. But pointwise L1 goes the **other way** (0.171 → 0.199). |
| Seam levelling | Seam discontinuity **2.1x** lower with ±0.15 exposure, 1.5x without; and L1 vs mean-exposure truth 0.078 → 0.052. |
| Bilinear sampling | Per-vertex bake **1.9x** more accurate (0.0052 → 0.0027); blended atlas 18% better; seam levelling unchanged. |
| 16-bit normals | Mean normal error 0.0033 → 0.0013 on a light decimation. The 8-bit error is *entirely* quantization (predicted 0.0034 from the step size). |
| Fit-target decimation | ratio 0.25 → 1184 triangles (81% fewer), 1.0 → 142 (98%), 4.0 → 24 (99.6%) on an analytic sphere. |
| Culling | 360 of 720 faces on nested spheres — exactly the sealed inner shell. |
| Atlas sizing | Projected areas match the analytic silhouette disc to **0.1%**. |
| Multi-page | 4 pages of 256² ≈ one 512² page (0.0318 vs 0.0310); a single 256² manages only 0.0569. |
