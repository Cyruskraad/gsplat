# gsplat Photogrammetry Pipeline — Project Status

**Repo:** `Cyruskraad/gsplat` (fork of `nerfstudio-project/gsplat`)
**Branch:** `claude/photogrammetry-techniques-plan-jb0pod`
**PR:** [#3 — Add state-of-the-art photogrammetry pipeline](https://github.com/Cyruskraad/gsplat/pull/3) (open, draft)
**Diff size:** 29 files changed, +8,396 / −10 lines, 18 commits since branching from `main`

---

## START HERE — picking this up in a new session

1. **Read this file top to bottom.** It is the complete state of the work:
   what exists (§2), what was never actually executed (§3), what's blocked
   (§4), and what to do next (§5).
2. **Check the live state of the PR** — it may have moved since this was
   written: `gh pr view 3 --repo Cyruskraad/gsplat` (or open the URL above)
   for CI status, mergeability, and any review comments.
3. **Confirm the tree is still green** before changing anything:
   ```bash
   python -m pytest tests/test_bundle_adjustment.py tests/test_mesh_extraction.py \
       tests/test_neural_sfm.py tests/test_colmap_dataset.py \
       tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py \
       tests/test_texturing.py tests/test_extract_mesh_io.py -q
   ```
   Expect **150 passed**. Needs `pycolmap`, `open3d`, `scikit-learn`,
   `opencv-python-headless`, `imageio`, `piexif`, `pytest-check` installed.
4. **Before touching the texturing code**, read
   [`photogrammetry_texturing_plan.md`](photogrammetry_texturing_plan.md). All
   four of its steps are landed, so it is no longer a to-do list — it is the
   record of what was measured and why the tests are shaped as they are. Its
   "Premise, measured" section and the step 2/3 findings each document a
   plausible-looking measurement that turned out to measure the wrong thing.
5. **Otherwise start from §5.1.** As of the latest session all three §5.1 items
   are still blocked (re-verified, see §4) — Actions is still off, the PR is
   still an unreviewed draft, and the sandbox still has no GPU/CUDA/`colmap`/
   capture data. Re-check them first anyway, then continue down §5.2.

**Ground rules carried over from the work so far:**

- Develop on branch `claude/photogrammetry-techniques-plan-jb0pod`;
  everything lands in PR #3. Don't start a new branch/PR without being asked.
- **CI cannot run** (§4), so every change must be validated by hand before
  committing: `python -m black --check --required-version 22.3.0 <files>`,
  `python -m py_compile <files>`, and the full test suite above.
  **`py_compile` is not enough for `examples/*.py`:** it compiles without
  executing, so it misses a `NameError` in a `tyro` dataclass's annotations
  (a missing `Literal` import, say) that breaks the script on import. Also run
  `cd examples && PYTHONPATH=<repo> python -c "import <script>"` for each
  example script you touched -- that caught exactly this before a commit.
- **gsplat never bundles model-running code.** AI-assisted features consume
  precomputed model output via files — a documented repo convention (§5.3).
- The sandbox this was built in has **no GPU, no compiled CUDA extension, no
  CUDA `colmap` build, and no real capture data**. Be explicit in commit
  messages and the PR about what was actually executed versus verified by
  code review only (§3) — that honesty has been maintained throughout and
  should continue.

---

## 1. What this project is

`gsplat` already implements most state-of-the-art Gaussian-splatting
*training* techniques (3DGS adaptive density control, MCMC densification,
2DGS surfels, 3DGUT, Mip-splatting anti-aliasing, pose/appearance
optimization, bilateral-grid post-processing, depth losses). What it lacked
was the classic **photogrammetry loop** around that training step: refining
SfM camera poses, densifying the sparse point cloud, and turning a trained
scene into an actual textured surface mesh — plus, on top of that, AI-assisted
priors, automatic quality metrics, and a way to run the whole thing as one
measurable pipeline instead of a pile of disconnected scripts.

This work adds a new `gsplat.photogrammetry` subpackage that closes that
loop:

```
SfM (COLMAP or neural-SfM)
   → bundle adjustment
      → dense MVS
         → Gaussian-splat training (+ monocular depth priors, transient-object masking)
            → mesh extraction / texturing
               → automatic quality metrics at every step
```

Every stage is independently useful and independently runnable, but can
also be run end-to-end with one command via a new pipeline orchestrator.

---

## 2. What has been done

### 2.1 Core photogrammetry stages (`gsplat/photogrammetry/`)

| Module | What it does |
|---|---|
| `bundle_adjustment.py` | Torch-native, differentiable bundle adjustment: Huber-loss reprojection-error minimization over COLMAP's SfM point tracks (Adam, SO(3) exponential-map rotation updates, gauge-fixed via an anchor image). Reads/writes real `pycolmap` reconstructions — a drop-in refinement of an existing COLMAP model. Returns a stats dict (`mean_reprojection_error_before/after`, counts). |
| `dense_mvs.py` | Densifies the sparse COLMAP point cloud by shelling out to the `colmap` CLI's own patch-match stereo + fusion (undistort → patch-match stereo → stereo fusion). No CUDA-MVS reimplementation — genuinely wraps the real tool. |
| `mesh_extraction.py` | `extract_mesh_tsdf`: TSDF fusion of rendered 2DGS/3DGS depth+color maps into a mesh (Open3D `ScalableTSDFVolume`). `extract_mesh_poisson`: Poisson surface reconstruction from a dense point cloud. `bake_texture`: occlusion-aware per-vertex texture baking via ray casting against the mesh itself. `_clean_mesh`: removes small floating components. |
| `neural_sfm.py` | Tool-agnostic adapter for feed-forward neural-SfM tools (DUSt3R/MASt3R/VGGT-style, run externally, never bundled). `merge_point_maps_to_tracks` merges independent per-image point-map predictions into cross-view COLMAP tracks using **complete-linkage** clustering (bounds every merged track's spatial diameter to `merge_radius` — deliberately not single-linkage, which can transitively chain far-apart points together). `write_colmap_reconstruction` builds a valid COLMAP model from scratch so the rest of the pipeline (bundle adjustment especially) works unchanged on top of it. |
| `metrics.py` | Automatic quality metrics for every stage (see §2.3). |
| `pipeline.py` | Pure-stdlib stage orchestration/reporting layer (see §2.4). |

### 2.2 AI-assisted extras (all follow the repo's existing "consume precomputed output via files, never run the model" convention — documented precedent in `docs/source/examples/dynamic_surgical.rst` and `docs/source/proposals/gsharp_v0_2_port.rst`)

- **Monocular depth-prior supervision.** `examples/datasets/colmap.py`'s
  `Dataset` gained `mono_depth_dir`: loads one precomputed `<image_stem>.npy`
  relative-depth map per training image, correctly undistorted/cropped in
  lock-step with the training image (not just resized — a real geometric
  remap). `simple_trainer_2dgs.py --mono_depth_loss` supervises the *dense*
  rendered depth map against it via the already-implemented (but previously
  unwired) `gsplat.losses.pearson_depth_loss` (scale/shift-invariant, correct
  for relative-depth predictions). Additive to the existing sparse
  `--depth_loss` path. Docs include a Depth Anything V2 recipe.
- **Transient/dynamic-object masking.** `Dataset` also gained `mask_dir`:
  one precomputed `<image_stem>.png` per image (nonzero = keep/static, 0 =
  exclude/transient), aligned the same rigorous way as the depth prior, and
  combined (logical AND) with the pre-existing fisheye-undistortion ROI
  mask. `--mask_dir` on the trainer excludes moving people/vehicles/pets from
  the photometric loss (via `gsplat.losses.masked_l1`/`masked_ssim` — already
  implemented, previously used only by the separate `dynamic_surgical_trainer.py`
  pipeline, now reused here) and from `--mono_depth_loss`'s validity mask, and
  (via `extract_mesh_tsdf`) excluded pixels are zeroed before TSDF fusion so
  transient content isn't baked into the extracted mesh either. Docs include a
  `torchvision` Mask R-CNN recipe (COCO "movable" classes) for producing
  masks.
- **Neural-SfM import** (see `neural_sfm.py` above) is itself the third
  AI-assisted capability — it lets any feed-forward pose/point predictor
  stand in for COLMAP's incremental SfM.

### 2.3 Automatic quality metrics (`gsplat/photogrammetry/metrics.py`)

Every stage now reports quantitative numbers, not just an artifact:

- `point_to_mesh_distance(points, mesh)` — "cloud-to-mesh" fit via Open3D
  ray casting: does an extracted mesh actually pass through the point cloud
  (sparse SfM or dense MVS) it was built from?
- `mesh_quality_stats(mesh)` — watertightness, connected-component count,
  surface area/volume, edge-length stats.
- `point_cloud_stats(points)` — count, bounding-box extent, k-NN spacing
  (density proxy).
- `reconstruction_stats(colmap_dir)` — image/point/observation counts, mean
  track length, mean reprojection error of a COLMAP model — the baseline
  bundle adjustment improves on, and how an imported neural-SfM model is
  judged before it enters the rest of the pipeline.
- `track_stats(tracks)` — track-length distribution, `multi_view_track_fraction`
  (now also returned directly from `merge_point_maps_to_tracks(...)["stats"]`).
- `mask_coverage_stats(mask_dir)` / `depth_prior_stats(mono_depth_dir)` —
  sanity-check the AI-prior directories *before* a long training run (catches
  degenerate/constant/all-NaN depth maps, reports how much of the frame masks
  are excluding).

These are written to `stats/*.json` files by every stage CLI, following the
exact naming convention the trainer's own `eval()` already used for render
quality (`stats/val_step<N>.json`).

### 2.4 Pipeline orchestration (`gsplat/photogrammetry/pipeline.py` + `examples/run_pipeline.py`)

Previously each stage was a separate script with no shared reporting and no
one-command way to run all of them wired together. Now:

- `pipeline.py` (pure stdlib, no torch/CUDA/COLMAP — always importable):
  `PipelineReport`/`StageResult` are a stable JSON-serializable record of a
  run; `run_stage(...)` is a context manager that times a stage and records
  failures as `status="failed"` **without losing the rest of the report to a
  traceback**; `record_skipped(...)` records a stage that didn't run (missing
  dependency, not selected) as first-class report content, not a silent gap;
  `collect_artifact_metrics(result_dir, data_dir)` reads back every
  `stats/*.json` a stage wrote, shared between the new runner and the
  existing summarizer so they agree on one schema.
- `examples/run_pipeline.py`: runs `sfm_input → bundle_adjust → dense_mvs →
  priors → train → extract_mesh` as subprocesses of the *existing* per-stage
  scripts (same pattern `dense_mvs.py` itself already used for the `colmap`
  CLI) — each stage keeps its own CLI as the source of truth for its own
  options, so the runner stays thin. It threads real state between stages
  (the refined COLMAP dir feeds dense MVS and training; the dense point cloud
  feeds training; the latest checkpoint feeds mesh extraction). A stage
  needing something the machine lacks (CUDA `colmap` build, GPU, a prior
  checkpoint) is recorded `skipped` with the reason rather than failing the
  run, unless `--strict`. `--stages` selects a subset; `--dry_run` prints the
  exact commands without running them. Writes one
  `<result_dir>/pipeline_report.json`.
- `examples/summarize_photogrammetry_stats.py` was refactored onto the same
  `collect_artifact_metrics`, and now writes `stats_summary.json` (a
  deliberately different filename from `run_pipeline.py`'s own
  `pipeline_report.json`, so the two writers never clobber each other for the
  same run) — used to aggregate a manually-run (not `run_pipeline.py`-driven)
  sequence of the individual scripts.
- `simple_trainer_2dgs.py` gained `--colmap_dir`/`--dense_points_path`/
  `--dense_mode` CLI flags. The underlying `Parser` already supported these
  (from the very first round of this work), but the trainer's own CLI didn't
  expose them — closing that gap was necessary for `run_pipeline.py`'s
  `train` stage to actually be able to wire the earlier stages' output into
  training from the command line.

### 2.5 New CLI entry points (`examples/`)

`bundle_adjust.py`, `dense_mvs.py`, `extract_mesh.py`,
`summarize_photogrammetry_stats.py`, `run_pipeline.py` — all thin `tyro`
dataclass-config CLIs following the repo's existing example-script
conventions.

### 2.6 Documentation

- `docs/photogrammetry.md` (repo root) — the full guide: step-by-step manual
  usage, the one-command `run_pipeline.py` path, automatic metrics/report,
  monocular depth priors (with a Depth Anything V2 recipe), transient-object
  masking (with a Mask R-CNN recipe), neural-SfM import, and a Python-API
  reference section.
- `docs/source/apis/photogrammetry.rst`, `docs/source/examples/photogrammetry.rst`
  — Sphinx pages mirroring the above for the hosted docs site.
- `README.md` — a "Sep 2026" news bullet summarizing the whole feature.

### 2.7 Bugs found and fixed (self code-review, since CI is unavailable — see §4)

Because GitHub Actions is disabled at the repository level on this fork (and
neither the repo owner nor this session has access to flip that), every
round of work was followed by a **manual self code-review pass** in place of
CI. **Thirteen** real bugs have been found and fixed this way so far, **each
verified by reverting the fix and confirming a test genuinely fails without
it** (not just re-asserting the buggy behavior). The first five were found in
the initial pass and are listed here; the rest are described where the work
that surfaced them is (sixth §2.4, seventh §2.2, eighth/ninth §2.3, tenth
§2.10, eleventh §2.10, twelfth/thirteenth §2.12):

1. `refine_reconstruction`'s pose-writing broke on modern `pycolmap`'s
   rig/frame model (`Image.cam_from_world` became read-only) — fixed with a
   version-tolerant write path.
2. `--mono_depth_loss` alone (without `--depth_loss`) rendered the wrong
   depth flavor ("D" instead of "ED"), silently corrupting the loss — fixed
   by checking both flags in `render_mode`.
3. `mono_depth` maps were misaligned under lens distortion (resized to the
   wrong resolution stage, never put through the same undistortion remap as
   the image) — fixed to undergo the identical geometric transform.
4. `merge_point_maps_to_tracks` could "chain" far-apart points together via
   single-linkage clustering, violating its own documented merge-radius
   guarantee — fixed by switching to complete-linkage clustering.
5. The pre-existing fisheye ROI mask was never patch-cropped under
   `--patch_size`, silently mismatching the image's shape — fixed while
   touching this code for the new mask feature.

### 2.8 Testing

| Test file | Count | Covers |
|---|---|---|
| `tests/test_bundle_adjustment.py` | 3 | Pure-torch optimization core (no pycolmap needed) |
| `tests/test_mesh_extraction.py` | 36 | TSDF fusion + Poisson reconstruction against an analytic sphere; UV-atlas texture baking against an analytically-shaded sphere (see §2.9); decimation to a fit target, with the k-NN spacing checked against a grid of known pitch (see §2.13); culling geometry no camera saw, against a sealed inner shell whose correct answer is known by construction (see §2.16) |
| `tests/test_neural_sfm.py` | 4 | Track merging correctness/non-chaining, COLMAP round-trip, composition with bundle adjustment |
| `tests/test_colmap_dataset.py` | 13 | `Parser`/`Dataset` overrides, `mono_depth_dir` and `mask_dir` alignment (including under real lens distortion and patch cropping), fisheye-ROI combination |
| `tests/test_photogrammetry_metrics.py` | 14 | Geometry metrics against known analytic ground truth, plus `atlas_sharpness` (detail ordering, chart-border exclusion, empty/uint8 handling) |
| `tests/test_photogrammetry_pipeline.py` | 33 | Orchestration (timing/status/failure handling), artifact collection, the four new per-stage metric functions, the `priors` quality gate, report-on-failure, and the cross-stage derived metrics (see §2.9) |
| `tests/test_extract_mesh_io.py` | 5 | Texture-map writing: the 16-bit RGB PNG round trip (and the BGR channel reversal it depends on), and that 16 bits recovers normal detail 8 bits cannot |
| `tests/test_texturing.py` | 42 | Per-face view selection: edge adjacency (vs Euler's identity), the gradient summed-area table, the quality term's geometry and visibility, the MRF's seam/quality tradeoff, unusable-view handling, determinism and multi-seed escape; and the view-selected bake — detail retention vs blending, the blended fallback, the shared UV layout, and the two numerical guards in §2.12; and seam levelling — the conjugate-gradient solver against a dense solve, shared-edge recovery, and that levelling closes the exposure steps without introducing a colour cast |
| **Total** | **150** | **All passing** in an isolated venv with real `pycolmap`/`open3d`/`scikit-learn`/`opencv` installed |

Every new/modified file is also checked against the repo's exact pinned
`black==22.3.0` and `python -m py_compile`.

**Beyond unit tests**, the pipeline orchestration was validated with real
executions, not just `--help`:
- `run_pipeline.py --dry_run` against a synthetic COLMAP dataset, confirming
  the full command-construction/stage-wiring logic.
- A **real** (non-dry-run, CPU) execution of the `sfm_input` + `bundle_adjust`
  stages against that dataset — genuinely invoked `bundle_adjust.py` as a
  subprocess, ran torch-based bundle adjustment on CPU, and read the result
  back correctly.
- `summarize_photogrammetry_stats.py` run end-to-end against that real
  output.

---

### 2.11 Module split: extraction vs. texturing

`mesh_extraction.py` had grown to 1431 lines holding two unrelated jobs.
Texturing moved to `gsplat/photogrammetry/texturing.py` (view sampling, the UV
atlas, normal/AO baking); `mesh_extraction.py` keeps surface reconstruction and
decimation; the shared open3d guard went to a tiny `_open3d.py` so neither
module has to depend on the other. `mesh_extraction.py` re-exports the moved
names, because the example CLIs and the test suite import bakers from that
path.

Verified as a **pure move**, not just by the tests: an AST comparison of every
top-level definition before and after reported 19 definitions before, 19 after,
none missing, none added, **none changed**. The suite went 85 passed -> 85
passed with no test file edited.

### 2.12 Per-face view selection and seam levelling (texturing plan, steps 2–3)

`bake_texture_atlas_view_selected` textures each face from a **single chosen
view** instead of blending every view that sees it — Waechter et al., *Let
There Be Color!* (ECCV 2014). The labelling (`face_view_quality` +
`select_views_mrf`) landed in the previous session; this wires it to the atlas,
adds the `atlas_sharpness` metric, and exposes
`--texture_view_selection` / `--texture_mrf_lambda` on `extract_mesh.py`.

Faces no view can texture, and texels their face's chosen view cannot see, keep
the **blended** colour, so this never punches holes; `--texture_outlier_sigma`
still governs those fallback regions.

**It is opt-in because it is a real tradeoff, not a strict win.** Measured on a
synthetic sphere with cameras rotated 45′ to simulate residual pose error:
blended retains **59%** of the ground truth's gradient detail and view-selected
**106%**, but pointwise L1 goes the *other* way (0.171 blended vs 0.199
selected). Blending attenuates detail; single-view sampling displaces it, and a
displaced-but-sharp texture scores worse pointwise than a blurred one while
looking far better. The docs say so plainly and the CLI warns when view
selection came out *less* sharp than blending — which means that capture's
poses were well enough registered that blending is the better choice.

**The premise had to be re-measured, and the redo is the interesting part.**
The plan's original numbers do not reproduce against
`tests/test_mesh_extraction.py`'s `_surface_pattern`: its wavelength is roughly
half the sphere, far coarser than the few pixels a pose error displaces a
projection by, so blending it loses nothing (98–100% gradient retention out to
90′). The effect only exists when the detail sits near the misregistration
scale, which is why `tests/test_texturing.py` defines its own
`_high_frequency_pattern` and `_SphereDataset` gained `pattern=` and
`pose_error_arcmin=`. **This is the fourth time on this branch that the
obvious measurement measured the wrong thing.**

**Two bugs found while building it** (§2.7's twelfth and thirteenth):

12. **`_gradient_summed_area` accumulated its summed-area table in float32** —
    the dtype training images actually arrive in. Over a 512×512 image that
    leaves box readouts wrong by **0.087** against a table maximum of ~1e5,
    four orders of magnitude worse than the 1.1e-5 the input rounding alone
    costs, and the error lands straight in the (face, view) quality scores the
    table exists to compute. Fixed by promoting to float64 before accumulating.
13. **`_box_means` could return a negative mean of gradient magnitudes**, from
    four-corner cancellation reading a near-empty box out of a large table.
    That reaches `-log()` in `select_views_mrf` as `NaN` — and `np.argmin`
    returns a `NaN`'s index in preference to every real cost, so the face would
    be textured from the one view that *cannot see it*. It also silently breaks
    the multi-seed search, since every comparison against `NaN` is False. Now
    clamped at zero. Observed on a real sphere render (9 negative readouts) and
    reproduced synthetically in the test.

#### Seam levelling (step 3)

One view per face means neighbouring faces meet at a step wherever the two
cameras disagree about exposure and white balance. `level_seams` solves for an
additive correction per **(vertex, label)** pair — not per vertex, since a
single per-vertex offset provably cannot close a discontinuity *at* that
vertex — through the normal equations of a linear least squares, with a
hand-rolled conjugate gradient (`_conjugate_gradient`, ~50 lines applying
`AᵀA` as a matvec, since `gsplat[mesh]` cannot take scipy as a hard
dependency). `seam_discontinuity` measures the result on the atlas as shipped,
and `--texture_seam_smoothness` exposes λ_s.

**The obvious formulation does not work, and the failure is the useful part.**
Comparing what each view reports *at the shared vertex* gives a target
dominated by noise: two views of one vertex disagree by **0.288** (L2 over RGB)
from pixel quantisation and silhouette bleed alone, where the exposure
difference being corrected is 0.26. The solve fitted noise larger than the
signal and made the atlas *worse* — seam discontinuity 0.184 → 0.221 on a scene
with no exposure differences at all. Using each view's **mean colour along the
shared edge**, over the same sample points (which is what Waechter et al.
actually specify), fixed it: **2.1x** reduction with ±0.15 per-view exposure,
1.5x without, and the atlas also moves *closer* to the mean-exposure ground
truth (L1 0.078 → 0.052), so it is removing real error rather than hiding a
boundary. Robust across λ_s from 0.003 to 1; default 0.1.

**A second measurement error, in the metric.** The first `seam_discontinuity`
stepped a *fraction of the face* inward from the shared edge before sampling
either side; on a seam-free ground-truth atlas that reads 0.087, i.e. it was
reporting the texture's own spatial variation as a seam. The inset is now
measured in **texels**. A floor remains — two samples either side of a border
are different surface points — so the metric is a before/after ratio, not an
absolute, and the test measures that floor on the same scene rather than
assuming it. **That is the fifth time on this branch that the obvious
measurement measured the wrong thing.**

**One guard is deliberately not mutation-checked, and is documented as such:**
the solver's `project_mean` gauge anchor. Removing it changes the seam solve by
less than 1e-9, because every row of the system is a difference of two unknowns
so the right-hand side is already orthogonal to the constants and CG from zero
never leaves the range space. It is kept against rounding on a larger system
and against an inconsistent right-hand side; the docstring says which. The test
asserts the zero-mean *property*, not the mechanism.

**Executed:** full suite 115 passed; the delivery path end to end on CPU
(3480 → 400 triangles, view-selected and levelled albedo + normal + AO on one
shared UV layout, OBJ/MTL/PNGs written and read back, `mesh_metrics.json`
round-tripped);
nine mutations checked (drop the clamp, drop the float64 promotion, blacken the
fallback, ignore the MRF labels, ignore the per-texel face id, disable view
selection, compare views at the vertex instead of along the edge, never apply
the seam correction, sample the seam metric at the shared vertex) — each fails
a test. Seam discontinuity on the shipped asset: 0.262 → 0.090.
**Reviewed only:** the `--texture_view_selection` /
`--texture_seam_smoothness` CLI guards and warnings, which need a real
checkpoint to reach.

### 2.18 Multi-page atlases

Past 8192 or 16384 texels a side an atlas stops being practical and the
evidence may still not fit — which §2.17's sizing now reports as `clamped`.
`bake_texture_atlas_pages` / `--texture_pages N` splits the surface across N
pages instead. Measured on a high-frequency pattern: 4 pages of 256² reach a
face error of **0.0318**, against **0.0310** for one 512² page of the same
total budget, where a single 256² manages only **0.0569**.

Faces are split by recursive median cuts on their centroids: deterministic,
exactly balanced for powers of two, and spatially compact (mean group spread
falls 0.996 → 0.890 → 0.770 → 0.476 from 1 to 8 pages), so each page unwraps
into few large charts. Geometry is untouched — only `triangle_uvs`,
`triangle_material_ids` and the texture list change — and open3d writes that as
a multi-material `.obj` + `.mtl` + N PNGs, verified through a disk round trip.

**The subtle part is the occluder.** A page ray-cast against only its own
geometry is blind to whatever the rest of the surface puts in front of it, and
textures the far wall of a room straight through the near one.
`_bake_points_from_views` gained an `occluder` argument for this, and pages
pass the whole mesh.

**A convex test shape cannot catch that**, which is worth recording: on a
sphere the bake is *identical* with and without the occluder, because every
face the cameras should not see is also facing away from them and back-face
rejection already removes it. The guard is pinned instead with a deliberately
non-convex scene — a small quad hiding the centre of a larger one, both facing
the cameras. Correct: the hidden point comes back with weight 0.0. With the
occluder dropped: weight 1.198 and the near quad's red painted onto it.

That took **two** tests, and mutation checking is what found the second. The
first proves the sampler honours an occluder; swapping the *call site* in
`bake_texture_atlas_pages` to `occluder=None` still left the suite green,
because that test supplies its own occluder either way. This is the third time
on this branch that a mechanism-level test passed while the call site went
unpinned (cf. §2.16's quality-vs-visibility cull, §2.17's max-over-views).

**Executed:** the measurements above; four mutations checked (bake each page
against itself, give every face material id 0, scatter page UVs to the wrong
faces, partition without sorting) — each fails a test. **Reviewed only:** the
`--texture_pages` atlas-mode guard. Combining pages with view selection is
**refused rather than half-supported**: the MRF labels faces across the whole
mesh and its seam levelling would have to run across page boundaries, which is
not implemented.

### 2.17 Sizing the atlas from the evidence

`--texture_size` was the same wrong question `--target_triangles` was, and gets
the same answer. `recommended_texture_size` / `--texture_texels_per_pixel`
chooses it from how much photographic evidence exists: the source pixels
covering each patch of surface, at **the best look the capture ever got at it**
— the maximum over views, not the sum. Photographing a wall twenty times is not
more detail than photographing it twice from the same distance. (Measured:
tripling the views raises the evidence 1.26x, not 3x.)

Two numbers are measured rather than assumed, and both changed the design:

- **Packing efficiency.** I was about to hardcode 0.75, measured 44% on one
  sphere, then found the real range is **42.7%–73.2%** across meshes and not
  monotonic in density. It *is* stable per mesh (five repeated unwraps spread
  by ≤2.8%, by 0.0% on two of three), so it is now measured with one probe
  unwrap rather than guessed.
- **Rounding.** Rounding *up* to a power of two quadruples the atlas for an
  arbitrarily small overshoot: an exact size of 518.1 rounded to 1024 and baked
  **3.88x** more texels than there were pixels to fill them. Nearest lands on
  512 and covers 0.98x. Found by running it, not by reading it.

The projection is checked against closed form: the faces one view sees tile a
silhouette disc of area `π(f·r/√(d²−r²))²` — measured 18850.9 against an
analytic 18877.5, 0.1%.

**A test premise I got wrong, and what replaced it.** I predicted the evidence
would fall with camera distance by the disc law, `√((d₂²−r²)/(d₁²−r²))` = 2.07,
and asserted it. It measures 2.27. The disc law describes a *silhouette*; this
sums each face's *best* view, and a face seen head-on projects like a patch at
range `(d−r)`, giving 2.40 as the other bound. The truth sits between because
faces spread across the visible cap rather than sitting at its closest point.
The test now asserts that bracket, with both ends derived.

**Executed:** the measurements above; full suite 144 passed; the complete
delivery path on CPU (6960 faces → cull 3480 → fit-decimate to 592 → atlas 512
chosen from evidence → view-selected, seam-levelled albedo + 16-bit normal + AO
on one shared UV layout, OBJ/MTL/PNGs written and read back). Four mutations
checked (round up instead of nearest, sum instead of max over views, ignore
packing efficiency, drop the absolute value in the triangle area) — each fails
a test. The max-over-views one needed a second test: every scaling test uses
*ratios*, and a consistent over-count cancels out of a ratio, so nothing caught
it until a test pinned absolute evidence against view count. **Reviewed only:**
the `--texture_texels_per_pixel` atlas-mode guard and clamp warning.

### 2.16 Culling geometry no camera observed

TSDF fusion returns a **closed** surface. That is what makes it watertight, and
it also means it invents geometry: the underside of anything resting on the
ground, the back of an object the capture only circled halfway, the inner shell
of a volume sealed off from every camera. None of it can be textured — those
faces carry the seam-dilation fill colour — and all of it costs triangles,
atlas area and file size. `cull_unobserved_faces` / `--cull_unobserved`
removes it, before decimation and texturing so neither spends its budget on
surface that will never be seen.

**Ground truth for the test needs no renderer:** a mesh built as an outer shell
plus a second shell sealed inside it. No camera outside can see any face of the
inner one, every face of the outer one is seen from somewhere, and the inner
faces are a known block of triangle indices — so "was exactly the right set
removed?" is answerable both ways. Measured: 360 of 720 faces culled, every
survivor's centroid out at radius > 0.9, and an observation histogram showing
exactly 360 faces seen by zero views.

**The subtle part is which question visibility asks.** It is deliberately *not*
`face_view_quality() == 0`. Quality is gradient energy over the projection, so
a face on a flat untextured surface scores zero however plainly it is in view —
there is no detail there to measure. Measured on a flat-shaded sphere: 215 of
653 visible (face, view) pairs score exactly zero, and **12 of 224 faces score
zero from every view while every camera sees them**. A cull reusing the quality
matrix deletes surface out of the middle of an observed object. `face_visibility`
was split out of `face_view_quality` for this, and both the matrix-level
distinction and its consequence at the call site are pinned by tests.

That second test exists because the first round of mutation checking **missed
this**: swapping the implementation to `quality > 0` passed the whole suite,
because the nested-spheres scene is textured and the two agree there. The
matrix-level test proved the two differ; nothing pinned the *cull* consulting
the right one until a flat-scene culling test was added.

Culling everything **raises** rather than returning an empty mesh: that means
the mesh and the dataset do not describe the same scene (wrong poses, wrong
scale, a different coordinate frame), and handing back an empty mesh at the end
of a long run would hide it. The CLI also warns when more than half the faces
were seen by no view at all.

**Executed:** the measurements above; the full delivery path end to end on CPU
(6960 faces → 3480 culled → 112 after fit-target decimation → view-selected and
seam-levelled albedo + 16-bit normal map + AO on one shared UV layout,
OBJ/MTL/PNGs written and read back). Four mutations checked (cull on quality
instead of visibility, mutate the caller's mesh in place, invert the keep mask,
drop the cull-everything guard) — each fails a test. **Reviewed only:** the
`--cull_unobserved` CLI guard and its histogram warning, which need a real
checkpoint to reach.

A bug caught in the CLI while wiring it, worth noting because `--help` is the
only place it shows: inserting the two new fields *above* `target_triangles`
left that field's docstring comment attached to `--cull_unobserved` and
`--target_triangles` with none at all. tyro takes the comment block immediately
above a field as its help, so field order and comment order have to move
together.

### 2.15 Bilinear source-image sampling

Every bake in `texturing.py` read its colours from the source images with
**nearest-neighbour** lookup. A surface point almost never lands on a pixel
centre, so that threw away up to half a pixel of the projection's accuracy —
and did so *differently in each view*, which is also what made two views
disagree about a point's colour by more than they needed to. Now bilinear.

Measured on the analytic sphere over 16 views:

| | nearest | bilinear |
|---|---|---|
| mean per-sample error vs ground truth | 0.0707 | 0.0572 |
| mean disagreement between two views of one vertex | 0.263 | 0.217 |
| per-vertex bake error vs ground truth | 0.0052 | **0.0027** |
| blended atlas error vs ground truth | 0.0052 | 0.0043 |
| seam levelling reduction | 1.97x | 1.94x |

The per-vertex bake gains most (1.9x) because it has no averaging to hide
behind; the blended atlas gains 18%; **seam levelling gains nothing**, because
averaging along each seam edge (§2.12) already does the same job. Worth
recording that the noise floor measured there — 0.288 pairwise disagreement,
larger than the 0.26 exposure signal it had to see past — is partly this, and
is now 0.217.

No public parameter: the whole subpackage is unreleased, bilinear is better on
every measure taken, and adding a `sampling=` argument to six bakers would be
API surface for a setting with one right answer.

**Executed:** the table above; three mutations checked (revert to
nearest-neighbour, forget the half-pixel offset, swap x and y in the lookup) —
each fails a test. **Not mutation-checked, because it is unreachable:** the
`minimum` clamping `x1`/`y1` to the last pixel. The clip on `x`/`y` already
forces the fractional weight to zero at the border, so wrapping instead
produces identical results; it is kept so the intent survives a later change
to the clip, and the docstring says so.

### 2.14 16-bit normal maps (§5.2 follow-on)

The 8-bit normal map's resolution floor was documented as a known sharp edge;
this removes it. Encoded as `0.5 + 0.5 * n`, the whole range is spent on
`[-1, 1]`, so 8 bits cannot represent a normal deviation finer than `2/255 ≈
0.0078` — about 0.45° of tilt — however dense the source mesh is.
`bake_normal_map(bits=16)` / `--normal_map_bits 16` drops that to `3.1e-5`.

Measured on a sphere decimated 6240 → 3000 triangles (a *light* decimation,
exactly the regime where 8 bits stops resolving anything), against the analytic
normal — on a unit sphere the true normal at a point is that point:

| bits | quantization floor | mean normal error |
|---|---|---|
| 8 | 0.0078 | 0.0033 |
| 16 | 0.000031 | 0.0013 |

The 8-bit error is *entirely* quantization, which is what makes this worth
doing rather than a nicety: uniform rounding over a step of `2/255` costs about
a quarter of a step per channel, ≈0.0034 in L2 over three channels, and that is
what the bake measures to three digits. At 16 bits what is left is the bake's
own geometric error. `bake_normal_map` now reports the `quantization_floor` it
used, so the same comparison is available on a real asset.

**A trap found while wiring it:** imageio's default PNG backend is Pillow,
which **cannot write a 16-bit RGB PNG at all** — it supports 16-bit grayscale
only, and raises `TypeError: Cannot handle this data type`. The CLI writes
those through OpenCV instead (already a dependency of the dataset loader).
OpenCV is **BGR**, so a normal map written without reversing the channels comes
back with X and Z swapped — an asset that loads without complaint and shades
wrong. `tests/test_extract_mesh_io.py` pins the round trip, with the channels
made deliberately distinguishable so a swap cannot pass.

**Executed:** the measurement above; three mutations checked (forget the BGR
reversal, quantise 16-bit output to 8-bit levels, drop the bit-depth guard) —
each fails a test; the full delivery path end to end on CPU with a 16-bit
normal map, written and read back byte-identical.

### 2.13 Decimation to a fit target (§5.2 follow-on)

`simplify_mesh` takes a triangle budget, which is the wrong question to have to
answer: how many triangles a scene needs depends on the scene, and the number
is only checked *afterwards* by measuring the cloud-to-mesh fit — the thing
actually cared about. `simplify_mesh_to_error` inverts it: give it the fit you
will accept and it binary-searches the triangle count, decimating and
re-measuring `point_to_mesh_distance` at each probe.

The target is given scale-free, as a multiple of the reference cloud's own mean
k-NN spacing — the same reading as the report's `mesh_fit_over_point_spacing`,
so "1.0" means the same thing on a tabletop scan and a city block. Measured on
an analytic sphere with a 20k-point reference cloud: ratio 0.25 → 1184
triangles (81% fewer), 1.0 → 142 (98%), 4.0 → 24 (99.6%), each independently
re-measured within its budget.

Two decisions worth keeping:

- **The mesh returned is one whose error was measured**, not one the search's
  final bracket implied was fine — quadric decimation is only *roughly*
  monotone in the triangle count and does not always land on the count it was
  asked for. Returning the *smallest* feasible probe rather than the last one
  is an optimality refinement, not a correctness guard (every candidate is
  feasible by measurement), and is **not mutation-checked**: on well-behaved
  input the two coincide, and no synthetic scene here separates them.
- **A target the input already misses has no solution below it**, since
  decimating only moves the surface further from the cloud. The input comes
  back unchanged with `target_met: false` and the CLI warns, rather than
  handing back a smaller mesh that misses by more.

The k-NN spacing is computed through open3d's own vectorised
`core.nns.NearestNeighborSearch` rather than `point_cloud_stats`'s
`scikit-learn`, so this does not pull a new hard dependency into the `mesh`
extra. Large clouds are subsampled to 20k query points — it is a density
estimate, and twenty thousand neighbourhoods is already far more than it needs.

**Executed:** the sweep above; four mutations checked (return an unmeasured
over-decimated mesh, decimate anyway when the input already misses the target,
include the self-match in the k-NN average, drop the empty-cloud guard) — each
fails a test. **Reviewed only:** the `--target_fit_ratio` / `--target_triangles`
mutual-exclusion guard and the miss warning, which need a real checkpoint to
reach.

### 2.10 Decimation + normal-map baking (the delivery path)

TSDF/Poisson extraction tessellates to the voxel grid rather than to the
scene's complexity, so a raw extraction is far heavier than the geometry
warrants. The standard photogrammetry answer is not a coarser extraction (that
loses detail) but decimate-and-bake, which this now implements:

- `simplify_mesh(mesh, target_triangles=...)` — Garland & Heckbert quadric
  error decimation, then the existing `_clean_mesh` pass.
- `bake_normal_map(high_mesh, low_mesh, space="tangent"|"object")` — casts a
  ray per texel from just outside the low surface inward along its own normal
  onto the dense mesh, and stores the dense mesh's barycentrically
  interpolated normal there. Returns `(mesh, normal_map, stats)`; `stats`
  carries `hit_fraction`, the diagnostic that says whether the map is doing
  anything (a low value means the ray cage doesn't span the gap).
- `extract_mesh.py --target_triangles/--normal_map/--normal_map_space`, which
  decimates before texturing, bakes the normal map after the albedo atlas,
  writes `mesh_normal.png`, and appends `norm`/`map_Bump` to the `.mtl`
  (open3d's OBJ writer emits `map_Kd` and nothing else, so the normal map
  would otherwise ship unreferenced).
- `_unwrap_and_rasterize` factors the unwrap + texel-frame rasterization out
  of `bake_texture_atlas`, so both bakes share one atlas construction.

**Robust multi-view texture fusion.** Both bakes previously blended every
view with a plain weighted mean, which on a real capture blurs and ghosts
wherever views disagree. `outlier_sigma` (CLI `--texture_outlier_sigma`) adds
iterative sigma clipping: discard observations far from a point's own mean,
re-estimate from the survivors, repeat. **A single pass is not enough**, and
this was worth measuring rather than assuming -- the bad samples inflate the
spread they are measured against, so at 25% contamination they sit right at a
1.5-sigma threshold and survive; the first implementation improved error by
only 4%. Iterating fixes it. Measured 3x error reduction (0.045 -> 0.015) in
the regime it is designed for, up to 9x with more views.

**A test-model trap, again worth recording:** the first version of the test
corrupted whole frames, which makes some surface points *majority*-wrong
(contaminated weight fraction up to 0.94, measured). No estimator centred on
the majority can recover those, so the feature looked broken when the test
was. The test now uses a partial-frame occluder and asserts its own premise --
that contamination is a per-point minority -- before measuring. The limitation
is real and documented: robust fusion complements `--mask_dir`, it does not
replace it.

**Ambient occlusion** completes the map set: `bake_ambient_occlusion` casts
`num_samples` cosine-weighted hemisphere rays per texel (Malley's method, so a
plain mean is unbiased) and stores the fraction that escaped, onto the same
shared atlas. `extract_mesh.py --ao_map/--ao_samples`.

**Bug found while wiring AO (§2.7's eleventh):** baking against a *different*
occluder mesh was silently wrong. Decimation cuts corners, so ~80% of a
decimated mesh's texels sit *inside* the dense mesh it came from (measured via
signed distance); with only the self-occlusion epsilon those rays start under
the occluder and hit it instantly, baking mean AO 0.204 instead of ~0.98 — a
map that looks like heavy occlusion and is pure artifact. Fixed with a `cage`
parameter defaulting to 2% of the bounding-box diagonal for the cross-mesh
case. Because that cage also erases occlusion finer than itself, the CLI now
bakes AO as *self*-occlusion on the shipped mesh; the dense bake stays
available through the Python API.

**Correctness issue this exposed (§2.7's tenth):** open3d's `compute_uvatlas`
is **not deterministic** — unwrapping the same mesh twice yields different UV
layouts (confirmed: four unwraps of one sphere, four different layouts). Since
`bake_texture_atlas` and `bake_normal_map` each unwrapped independently, an
asset's albedo and normal map would have had *different* UV layouts, so the
normal map would be sampled through the albedo's coordinates — a silently
broken asset. `_unwrap_and_rasterize` now reuses a mesh's existing
`triangle_uvs` when present (`reuse_uvs=False` forces a fresh unwrap). This
surfaced as an intermittently failing test before it was understood.

**A test-design trap worth remembering:** a plain sphere is useless for
validating normal-map baking. Its interpolated vertex normals are already
essentially the exact analytic normals, so a decimated sphere has no detail to
recover and an 8-bit map (quantization floor ~0.004, since `0.5` encodes to
byte 128) can only add noise — the first version of the test failed for
exactly that reason, and loosening the threshold would have hidden it. The
test now uses a radially displaced "bumpy" sphere, where the low-poly is
genuinely wrong (base error 0.14) and the map recovers it (0.025), with ground
truth from an *independent* closest-point query rather than the bake's own ray
cast.

### 2.9 Follow-on work from §5.2 (later session)

Three of §5.2's items are now done. All three were validated the same way as
everything before them: by hand, since CI still cannot run.

- **UV-atlas texture baking** (`bake_texture_atlas`). Per-vertex colors can
  only carry as much detail as the mesh is tessellated for. This UV-unwraps
  the mesh (open3d's `compute_uvatlas`), recovers each texel's 3D surface
  point/normal by baking vertex positions and normals into the atlas
  (`bake_vertex_attr_textures`), colors those points, and pads the result
  across UV seams. Returns `(mesh, texture)` with `triangle_uvs`/`textures`
  set, so `write_triangle_mesh("mesh.obj", mesh)` emits .obj + .mtl + .png.
  **No new dependency** — this uses open3d, already the `gsplat[mesh]` extra,
  so the `xatlas` §5.2 originally suggested turned out not to be needed.
  `_bake_points_from_views` factors the existing occlusion-aware,
  view-weighted blend out of `bake_texture` so both paths share one color
  signal; it also now chunks the ray cast, bounding memory for large atlases.
  `bake_mesh_texture(..., mode=...)` is the dispatcher both CLIs use, exposed
  as `extract_mesh.py --texture_mode/--texture_size`,
  `simple_trainer_2dgs.py --mesh_texture_mode/--mesh_texture_size`, and
  `run_pipeline.py --texture_mode/--texture_size` (which forwards them to its
  `extract_mesh` stage and records `mesh.obj` rather than `mesh.ply` as that
  stage's output on the atlas path), so the one-command pipeline reaches the
  feature too.
  **Sharp edge worth knowing:** open3d's `compute_uvatlas` requires a manifold
  mesh and **segfaults** rather than raising on non-manifold input (confirmed:
  exit 139). `bake_texture_atlas` therefore checks edge/vertex manifoldness up
  front and raises `ValueError`; the dispatcher warns and falls back to
  per-vertex colors, since losing the mesh after hours of training is worse.
- **`priors` mask/depth quality gate.** `pipeline.check_prior_quality(...)`
  (pure stdlib) turns the `mask_coverage_stats`/`depth_prior_stats` dicts into
  a list of concrete problems — an empty prior directory, masks excluding
  (almost) the whole frame or nothing at all, a mask excluding its entire
  frame, mostly-degenerate or mostly-non-finite depth maps. `run_pipeline.py`
  prints them, records them in the report, and fails the stage under
  `--strict`, so the run stops *before* the training stage rather than after.
  Thresholds are CLI-tunable (`--max_excluded_fraction`,
  `--max_degenerate_fraction`) and pass on equality.
- **Cross-stage derived metrics.** Each stage's metrics say what it produced;
  these say whether it *improved on* what came before, which is what a
  photogrammetry run is actually judged on. `pipeline.derive_cross_stage_metrics`
  reports `reprojection_error_reduction`,
  `points_retained_after_bundle_adjust`, `densification_ratio` (moved out of
  the ad-hoc inline computation in `run_pipeline.py`),
  `mesh_fit_over_point_spacing` and `mesh_edge_over_point_spacing`. The last
  two are the point of the exercise: a cloud-to-mesh distance in raw scene
  units means nothing on its own, but divided by the dense cloud's own k-NN
  spacing it becomes a scale-free verdict on whether the mesh fits within the
  evidence's noise floor, and whether `--voxel_size` matched the cloud's
  density. Every metric is omitted rather than guessed when an input stage was
  skipped or failed. `run_pipeline.py` prints the block and stores it in
  `pipeline_report.json` under `context.cross_stage_metrics`;
  `summarize_photogrammetry_stats.py` does the same via
  `cross_stage_metrics_from_artifacts`, sharing one derivation so a hand-run
  sequence is judged identically to an orchestrated one. **Note:**
  `stats_summary.json`'s shape changed to
  `{"artifact_metrics": ..., "cross_stage_metrics": ...}` (nothing else in the
  tree reads that file).
- **CI would exercise the photogrammetry suite** (§5.2's last item, which
  asked for exactly this double-check). Measured by simulating
  `core_tests.yml`'s dependency set: `pytest tests/` would **pass but be
  hollow** — 26 of 58 tests run, four of the six photogrammetry files skipping
  wholesale on their `importorskip` guards. `core_tests.yml` now also installs
  `pycolmap`, `open3d`, `scikit-learn`, `opencv-python-headless` and `piexif`,
  which takes it to all 58. Adding `scikit-learn` also un-skips the
  pre-existing `tests/test_init_multiframe.py` (checked: 10 passed, 1
  skipped). `opencv` does **not** newly enable
  `tests/sensors/models/cameras/test_fisheye.py` — that file imports
  `gsplat.sensors` at module scope and so already fails collection on a CPU
  runner with or without cv2 (pre-existing, unrelated to this PR, untouched).

- **AI-prior recipes, partially verified** (§5.2's "real-world AI-prior
  recipes"). Pretrained weights cannot be downloaded here -- the environment's
  network policy allows package registries but denies other hosts, so both
  `download.pytorch.org` and HuggingFace return 403 through the proxy -- so
  neither recipe can be run against real model output. What *was* executed:
  the Mask R-CNN recipe's entire API path, verbatim except for a
  randomly-initialized model of the same architecture, confirming
  `decode_image(path)`, the weights' `transforms()`, the prediction dict's
  keys, `masks` being `(N, 1, H, W)` so `obj_mask[0]` is `(H, W)`, and the
  written PNG being single-channel uint8. Its output was then fed through
  gsplat's real `Dataset(mask_dir=...)`, which loaded it with the excluded
  block landing exactly where the recipe put it (excluded fraction
  0.1953125, matching to the bit). `tests/test_colmap_dataset.py` now pins
  that recipe's exact output format (PIL, single-channel uint8 from a bool
  array -- a different writer from the `imageio` masks the other tests use).

**Bugs found by probing degenerate inputs** (§2.7's eighth and ninth): an
adversarial pass over the metrics surface -- feeding each function empty,
single-element and non-finite input -- found two plausible inputs that failed
opaquely rather than reporting anything useful.
`point_to_mesh_distance(points, empty_mesh)` died with open3d's
`IndexError: _Map_base::at`, and `extract_mesh.py` calls it unconditionally
after writing the mesh, so a TSDF run that produced nothing usable (bad
`--voxel_size`/`--sdf_trunc`) crashed opaquely at the very end of a long GPU
run. It now raises a `ValueError` naming the likely cause, and
`extract_mesh.py` checks first and prints what to adjust instead of failing.
`point_cloud_stats(empty_cloud)` raised numpy's bare "zero-size array to
reduction operation minimum"; an empty cloud is a real outcome (dense MVS
fusing nothing) and the rest of the module reports empty input as zeros, so it
now does too. Relatedly, `point_to_mesh_distance` with no points returns
`None` distances rather than `0.0` -- "nothing was measured" and "the fit is
perfect" must not look alike, least of all to
`mesh_fit_over_point_spacing`, which divides by them (confirmed: a null
`point_to_mesh` omits that metric rather than fabricating one).

**Bug found while verifying the depth recipe** (§2.7's seventh): a `(1, H, W)`
depth map -- the shape a transformers depth-estimation pipeline's
`predicted_depth` commonly has, and what the documented recipe produced --
was silently corrupted rather than rejected. `Dataset` compares
`mono_depth.shape[:2]` to the image's, so a `(1, H, W)` map takes the resize
path, and `cv2.resize` reads it as a one-row image with W channels and returns
`(H, W, W)` **without raising**. Training would have been supervised against
reshaped noise. The loader now squeezes singleton axes and raises a
`ValueError` naming the file for anything still not 2D; `depth_prior_stats`
reports `num_not_2d_maps` and the `priors` gate flags even one, so a whole
directory of them is caught before training starts; and the documented recipe
squeezes explicitly and asserts the shape.

**Bug found by running the new gate** (§2.7's sixth): without
`--continue_on_error`, a failing stage re-raised out of `run_pipeline.py`'s
`main()` *before the report was ever written* — so the `status="failed"`
record `run_stage` had just built was lost to the traceback, and any
`pipeline_report.json` from an earlier run stayed on disk still claiming
success. `main()` now runs the stages in a `try`/`finally` around the report
write. This affected every stage, not just the new gate.

**What was actually executed for these** (same sandbox: no GPU, no CUDA
extension, no capture data):

- Full suite 72 passed. Every new guard was mutation-checked rather than just
  asserted: removing the depth-map squeeze/ndim guard fails its test;
  vertically flipping the baked atlas fails the UV-convention test;
  removing the seam-fill edge-blanking fails the wrap test; calling
  `compute_uvatlas` on the non-manifold test mesh without the guard dies with
  SIGSEGV; reverting the `try`/`finally` fails the report-on-failure test;
  and for the cross-stage metrics, letting skipped/failed stages contribute,
  inverting the reprojection-reduction sign, and swapping the mesh-fit ratio
  each fail their tests.
- Real, non-dry-run runs of `run_pipeline.py --stages priors` against
  synthetic prior directories: warns and continues by default, fails the stage
  under `--strict` (exit 1, report written with `status="failed"`), clean on
  healthy priors.
- A real, non-dry-run CPU run of `run_pipeline.py --stages sfm_input
  bundle_adjust` against a synthetic COLMAP model with deliberately perturbed
  3D points, which produced genuine cross-stage numbers rather than mocked
  ones: `reprojection_error_reduction=0.9646`,
  `points_retained_after_bundle_adjust=1`. `summarize_photogrammetry_stats.py`
  was likewise run against real `stats/*.json` artifacts and produced the
  matching block.
- `extract_mesh.py --help` and `simple_trainer_2dgs.py --help` parse the new
  flags; `black --check --required-version 22.3.0` and `py_compile` on every
  changed file.

Still **not** executed: the atlas path against a real trained checkpoint and
real imagery (needs a GPU), and the `core_tests.yml` change itself (needs
Actions). The atlas correctness tests use analytic ground truth — a unit
sphere shaded by a known position-dependent function, viewed from 24
ray-traced cameras — and check the atlas *as addressed by the mesh's own
`triangle_uvs`*, which pins the OBJ v-up convention external tools rely on.

## 3. What has **not** been verified (sandbox limitations)

This work was done in a sandbox with **no GPU, no compiled CUDA extension,
no real capture data, and no CUDA-enabled `colmap` build**. Specifically
unverified beyond code review + the checks above:

- The actual **training runtime path** (`simple_trainer_2dgs.py` with
  `--mono_depth_loss`/`--mask_dir`/`--extract_mesh`/`--colmap_dir`/
  `--dense_points_path`) has never executed on a GPU.
- `extract_mesh.py`'s TSDF/Poisson extraction has never run against a real
  trained checkpoint (only against synthetic depth maps / point clouds in
  unit tests). The same goes for `--texture_mode atlas` (§2.9): its
  correctness is pinned against analytic ground truth, but it has never baked
  a real capture's imagery onto a real mesh.
- `dense_mvs.py` has never run at all beyond `--help` — needs a real
  CUDA-enabled `colmap` CLI install.
- No run has ever happened against a real dataset (e.g. Mip-NeRF 360
  "garden") — all validation used small synthetic `pycolmap` reconstructions
  and analytic shapes (spheres, grids).
- CI (`.github/workflows/*.yml` exist in the tree but Actions is disabled at
  the repo level) has never run on this branch — including the
  `core_tests.yml` dependency change made in §2.9, which was validated by
  simulating that dependency set locally, not by a CI run.

---

## 4. Outstanding blockers

- **GitHub Actions is disabled** on `Cyruskraad/gsplat` at the repository
  level. Confirmed via the GitHub API returning zero registered workflows
  despite workflow files existing in the tree — **re-checked in the §2.9
  session and still zero.** Neither this session nor the
  repo owner has the admin access to flip **Settings → Actions → General →
  Allow all actions**. Until someone with org/repo admin rights does that,
  there is no automated CI signal on this PR — every round of changes has
  been validated by manual code review instead.
- **PR #3 is still open and in draft state**, with no reviews, no review
  comments and no check runs as of the §2.9 session. It needs a human to mark
  it ready for review and merge it (or request changes). A ~60-minute
  self-scheduled check-in has been monitoring it for CI/mergeability/review
  activity throughout this work and will keep doing so.

---

## 5. Plan / what should happen next

Roughly in priority order:

### 5.1 Must happen before merge

**All three below were re-checked in the §2.9 session and are still blocked**
— they need repo-admin access and GPU hardware that no session so far has
had. Re-check them each time, but don't wait on them: §5.2 is where the
work is.

1. **Get this PR reviewed and merged**, or get feedback on scope/direction.
   It's currently a draft; someone needs to mark it ready and approve it.
2. **Enable GitHub Actions** on the repo (needs org/repo admin access this
   session doesn't have) so the existing `.github/workflows/*.yml` CI
   actually runs against this branch — this is the single highest-value
   remaining step, since it would give real, automated confirmation of
   everything that's currently "verified by code review" instead of a real
   test run.
3. **Run the full pipeline once on real GPU hardware against a real
   dataset** (e.g. Mip-NeRF 360 `garden`, or any COLMAP-processed capture):
   `python examples/run_pipeline.py --data_dir <real_dir> --result_dir
   <out>` end-to-end, including `dense_mvs` (needs a CUDA `colmap` build) and
   `--mono_depth_dir`/`--mask_dir` with real AI-model output. This would
   catch anything a synthetic/CPU-only sandbox structurally cannot —
   real-world numerical stability, actual runtime/memory behavior, and
   whether the produced mesh/metrics look sane on a real scene.

### 5.2 Natural follow-on work

**Done** (see §2.9 for what was executed vs. code-reviewed):
- ~~UV-atlas texture baking~~ — `bake_texture_atlas` / `--texture_mode atlas`.
  Used open3d's own `compute_uvatlas` rather than `xatlas`, so no new
  dependency was needed.
- ~~`priors` stage's mask-quality gate~~ — `pipeline.check_prior_quality`,
  wired into `run_pipeline.py` (warns by default, fails under `--strict`).
- ~~CI should exercise the photogrammetry suite~~ — the workflows do glob
  `tests/` broadly, but `core_tests.yml` didn't install the suite's
  dependencies, so 4 of its 6 files skipped wholesale (26 of 58 tests ran).
  Its install step now adds them. **Still needs Actions on to confirm.**

- ~~Dense-MVS metrics wired into the report more richly~~ —
  `derive_cross_stage_metrics`, shared by `run_pipeline.py` and
  `summarize_photogrammetry_stats.py`.

**Not started**, roughly in the order worth picking them up:
- **Appearance-embedding checkpoint support for mesh extraction.**
  `extract_mesh_tsdf`/`bake_texture` currently only support SH-color
  checkpoints (trained without `--app_opt`), since per-image appearance
  variation doesn't map onto one canonical mesh texture — resolving that
  (e.g. baking at a canonical appearance embedding) is unexplored. Needs a
  GPU and a real `--app_opt` checkpoint to evaluate, so it is effectively
  blocked alongside §5.1.
- **Real-world AI-prior recipes — partially done (§2.9).** The Mask R-CNN
  recipe's full API path has now been executed and its output format pinned by
  a test, and verifying the depth recipe turned up a real silent-corruption
  bug. What remains needs weights this environment cannot fetch (the network
  policy denies non-registry hosts): running either recipe against a real
  model on real images, and confirming the depth prior actually improves a
  trained result. Do this together with the §5.1 GPU run.
- **UV-atlas / normal-map polish, once a real capture has been through it.**
  `--texture_size` defaults to 2048, the seam dilation to 4 texels, and the
  normal-map ray cage to 2% of the bounding-box diagonal; none has been tuned
  against a real scene. Multi-material output (one atlas per chart group), a
  non-square atlas, and 16-bit normal maps (past the 8-bit ~0.004 floor) are
  unimplemented.
- ~~**Decimation driven by a quality target rather than a triangle count.**~~
  Done — `simplify_mesh_to_error` / `--target_fit_ratio`, see §2.13.
- ~~16-bit normal maps~~ (part of the UV-atlas/normal-map polish item) —
  `bake_normal_map(bits=16)` / `--normal_map_bits 16`, see §2.14. The rest of
  that item (multi-material output, non-square atlases, tuning the defaults
  against a real scene) still wants a capture this environment does not have.

### 5.3 Explicitly out of scope (by design, not oversight)
- gsplat will **never** bundle code that runs a neural network itself for
  depth estimation, segmentation, or neural-SfM — this is a deliberate,
  documented repo convention (see `docs/source/examples/dynamic_surgical.rst`,
  `docs/source/proposals/gsharp_v0_2_port.rst`), consistently followed
  throughout this work.
- `dense_mvs.py` will not reimplement patch-match stereo — it shells out to
  the real `colmap` CLI, matching how the rest of the pipeline treats COLMAP
  as the ground-truth SfM/MVS tool rather than something to reinvent.

---

## 6. File inventory (everything touched)

```
New:
  gsplat/photogrammetry/__init__.py
  gsplat/photogrammetry/texturing.py
  gsplat/photogrammetry/_open3d.py
  gsplat/photogrammetry/bundle_adjustment.py
  gsplat/photogrammetry/dense_mvs.py
  gsplat/photogrammetry/mesh_extraction.py
  gsplat/photogrammetry/neural_sfm.py
  gsplat/photogrammetry/metrics.py
  gsplat/photogrammetry/pipeline.py
  examples/bundle_adjust.py
  examples/dense_mvs.py
  examples/extract_mesh.py
  examples/summarize_photogrammetry_stats.py
  examples/run_pipeline.py
  tests/test_bundle_adjustment.py
  tests/test_mesh_extraction.py
  tests/test_neural_sfm.py
  tests/test_colmap_dataset.py
  tests/test_photogrammetry_metrics.py
  tests/test_photogrammetry_pipeline.py
  docs/photogrammetry.md
  docs/photogrammetry_status.md    (this file)
  docs/photogrammetry_texturing_plan.md  (approved plan for the in-progress
                                      view-selection texturing work)
  docs/source/apis/photogrammetry.rst
  docs/source/examples/photogrammetry.rst

Modified:
  examples/datasets/colmap.py       (colmap_dir/dense_points_path/dense_mode
                                      Parser overrides; mono_depth_dir/mask_dir
                                      Dataset additions)
  examples/simple_trainer_2dgs.py   (--extract_mesh, --mono_depth_loss,
                                      --mask_dir, --colmap_dir,
                                      --dense_points_path, --dense_mode)
  setup.py                          (gsplat[mesh] extra: open3d)
  examples/requirements.txt         (open3d)
  README.md                         (news bullet)
  .github/workflows/core_tests.yml  (install the photogrammetry suite's test
                                      deps so CI actually runs it -- see §2.9)
```

---

## 7. How to pick this back up

```bash
# See the PR
gh pr view 3 --repo Cyruskraad/gsplat   # or open the URL above

# Run the full test suite
cd gsplat
python -m pytest tests/test_bundle_adjustment.py tests/test_mesh_extraction.py \
    tests/test_neural_sfm.py tests/test_colmap_dataset.py \
    tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py \
    tests/test_texturing.py -v

# Try the one-command pipeline against a real capture. --strict now also
# fails the run if the AI-prior directories look unusable (see 2.9).
python examples/run_pipeline.py \
    --data_dir data/360_v2/garden --result_dir results/garden_pipeline \
    --mono_depth_dir <your_depth_dir> --mask_dir <your_mask_dir>

# Extract a mesh with a real UV texture atlas rather than vertex colors
# (writes mesh.obj + mesh.mtl + mesh_0.png)
python examples/extract_mesh.py \
    --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
    --data_dir data/360_v2/garden --result_dir results/garden_2dgs \
    --texture_mode atlas --texture_size 4096

# Read the full guide
cat docs/photogrammetry.md
```

---

*Generated by Claude Code, summarizing all work done on branch
`claude/photogrammetry-techniques-plan-jb0pod` across sessions
`session_01FfVDvERXP1ppdzKd63waP7` (§1-§2.8) and
`session_01Y7eAgYXjp1zBZCC2iMdTgU` (§2.9 and the §5 updates).*
