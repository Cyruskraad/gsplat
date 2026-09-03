# gsplat Photogrammetry Pipeline — Project Status

**Repo:** `Cyruskraad/gsplat` (fork of `nerfstudio-project/gsplat`)
**Branch:** `claude/photogrammetry-techniques-plan-jb0pod`
**PR:** [#3 — Add state-of-the-art photogrammetry pipeline](https://github.com/Cyruskraad/gsplat/pull/3) (open, draft)
**Diff size:** 26 files changed, +5,027 / −10 lines, 8 commits since branching from `main`

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
       tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py -q
   ```
   Expect **42 passed**. Needs `pycolmap`, `open3d`, `scikit-learn`,
   `opencv-python-headless`, `imageio`, `piexif`, `pytest-check` installed.
4. **Then start from §5.1** — the highest-value remaining steps are getting
   CI enabled, getting PR #3 reviewed/merged, and running the pipeline once
   on real GPU hardware against a real capture.

**Ground rules carried over from the work so far:**

- Develop on branch `claude/photogrammetry-techniques-plan-jb0pod`;
  everything lands in PR #3. Don't start a new branch/PR without being asked.
- **CI cannot run** (§4), so every change must be validated by hand before
  committing: `python -m black --check --required-version 22.3.0 <files>`,
  `python -m py_compile <files>`, and the full test suite above.
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
CI. Five real bugs were found and fixed this way, **each verified by
reverting the fix and confirming a test genuinely fails without it** (not
just re-asserting the buggy behavior):

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
| `tests/test_mesh_extraction.py` | 2 | TSDF fusion + Poisson reconstruction against an analytic sphere |
| `tests/test_neural_sfm.py` | 4 | Track merging correctness/non-chaining, COLMAP round-trip, composition with bundle adjustment |
| `tests/test_colmap_dataset.py` | 11 | `Parser`/`Dataset` overrides, `mono_depth_dir` and `mask_dir` alignment (including under real lens distortion and patch cropping), fisheye-ROI combination |
| `tests/test_photogrammetry_metrics.py` | 6 | Geometry metrics against known analytic ground truth |
| `tests/test_photogrammetry_pipeline.py` | 16 | Orchestration (timing/status/failure handling), artifact collection, the four new per-stage metric functions |
| **Total** | **42** | **All passing** in an isolated venv with real `pycolmap`/`open3d`/`scikit-learn`/`opencv` installed |

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

## 3. What has **not** been verified (sandbox limitations)

This work was done in a sandbox with **no GPU, no compiled CUDA extension,
no real capture data, and no CUDA-enabled `colmap` build**. Specifically
unverified beyond code review + the checks above:

- The actual **training runtime path** (`simple_trainer_2dgs.py` with
  `--mono_depth_loss`/`--mask_dir`/`--extract_mesh`/`--colmap_dir`/
  `--dense_points_path`) has never executed on a GPU.
- `extract_mesh.py`'s TSDF/Poisson extraction has never run against a real
  trained checkpoint (only against synthetic depth maps / point clouds in
  unit tests).
- `dense_mvs.py` has never run at all beyond `--help` — needs a real
  CUDA-enabled `colmap` CLI install.
- No run has ever happened against a real dataset (e.g. Mip-NeRF 360
  "garden") — all validation used small synthetic `pycolmap` reconstructions
  and analytic shapes (spheres, grids).
- CI (`.github/workflows/*.yml` exist in the tree but Actions is disabled at
  the repo level) has never run on this branch.

---

## 4. Outstanding blockers

- **GitHub Actions is disabled** on `Cyruskraad/gsplat` at the repository
  level. Confirmed via the GitHub API returning zero registered workflows
  despite workflow files existing in the tree. Neither this session nor the
  repo owner has the admin access to flip **Settings → Actions → General →
  Allow all actions**. Until someone with org/repo admin rights does that,
  there is no automated CI signal on this PR — every round of changes has
  been validated by manual code review instead.
- **PR #3 is still open and in draft state.** It needs a human to mark it
  ready for review and merge it (or request changes). A ~60-minute
  self-scheduled check-in has been monitoring it for CI/mergeability/review
  activity throughout this work and will keep doing so.

---

## 5. Plan / what should happen next

Roughly in priority order:

### 5.1 Must happen before merge
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

### 5.2 Natural follow-on work (not started)
- **UV-atlas texture baking.** `bake_texture` currently only produces
  per-vertex colors (explicitly documented as a deliberate scope boundary,
  not a bug) — a proper UV-unwrapped texture atlas (e.g. via `xatlas`) would
  be a natural next step for anyone wanting textures usable in standard DCC
  tools/game engines.
- **Appearance-embedding checkpoint support for mesh extraction.**
  `extract_mesh_tsdf`/`bake_texture` currently only support SH-color
  checkpoints (trained without `--app_opt`), since per-image appearance
  variation doesn't map onto one canonical mesh texture — resolving that
  (e.g. baking at a canonical appearance embedding) is unexplored.
- **`priors` stage's mask-quality gate.** `run_pipeline.py`'s `priors` stage
  currently just *reports* `mask_coverage_stats`/`depth_prior_stats`; it
  could optionally fail/warn loudly (or auto-adjust `--strict` behavior) when
  it detects a mask directory that excludes almost the entire frame, or a
  depth-prior directory that's mostly degenerate — turning the sanity check
  into an actual gate rather than an informational stat.
- **Dense-MVS metrics wired into `run_pipeline.py`'s report table more
  richly** — currently `densification_ratio` (dense vs. sparse point count)
  is the only derived cross-stage metric; comparable cross-stage deltas
  (e.g. mesh cloud-to-mesh fit vs. dense-cloud density) could be added.
- **CI, once enabled**, should be extended to actually exercise the
  photogrammetry test suite (`tests/test_bundle_adjustment.py`,
  `test_mesh_extraction.py`, `test_neural_sfm.py`, `test_colmap_dataset.py`,
  `test_photogrammetry_metrics.py`, `test_photogrammetry_pipeline.py`) if
  the existing workflow files don't already glob `tests/` broadly — worth
  double-checking once Actions is live.
- **Real-world AI-prior recipes.** The Depth Anything V2 and Mask R-CNN
  recipes in the docs are illustrative, tested only for API correctness
  against their respective libraries' documented interfaces — not run
  end-to-end against real images in this sandbox (no GPU). Worth a real
  pass once hardware is available.

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
    tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py -v

# Try the one-command pipeline against a real capture
python examples/run_pipeline.py \
    --data_dir data/360_v2/garden --result_dir results/garden_pipeline \
    --mono_depth_dir <your_depth_dir> --mask_dir <your_mask_dir>

# Read the full guide
cat docs/photogrammetry.md
```

---

*Generated by Claude Code, summarizing all work done in session
`session_01FfVDvERXP1ppdzKd63waP7` on branch
`claude/photogrammetry-techniques-plan-jb0pod`.*
