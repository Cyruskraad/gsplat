# Scaffolding

## File inventory

Everything below is new on this branch unless marked *(modified)*.

### Library — `gsplat/photogrammetry/` (7,235 lines)

| File | Lines | Holds |
|---|---:|---|
| `texturing.py` | 2960 | Everything that samples training views or writes an atlas |
| `mesh_extraction.py` | 1593 | Surface reconstruction, refinement, level sets, culling, decimation |
| `photometric_alignment.py` | 531 | Camera refinement against the surface (Zhou & Koltun 2014) |
| `metrics.py` | 669 | Automatic quality metrics for every stage |
| `pipeline.py` | 545 | Pure-stdlib orchestration layer (timing, status, report) |
| `bundle_adjustment.py` | 311 | Torch-native reprojection-error BA over COLMAP tracks |
| `neural_sfm.py` | 299 | Adapter for externally-run DUSt3R/MASt3R/VGGT-style tools |
| `__init__.py` | 158 | Re-exports; 47 public names |
| `dense_mvs.py` | 138 | Shells out to the `colmap` CLI for patch-match stereo + fusion |
| `_open3d.py` | 31 | The shared `_require_open3d()` guard, so extraction and texturing need not depend on each other |

### CLIs — `examples/` (2,292 lines)

| File | Lines | Purpose |
|---|---:|---|
| `extract_mesh.py` | 963 | The delivery stage; 43 flags, the richest CLI. Runs without a checkpoint or a GPU since `--mesh_path` |
| `run_pipeline.py` | 583 | Orchestrator — runs the per-stage scripts as subprocesses |
| `make_synthetic_capture.py` | 453 | Writes a complete multi-view-consistent capture to disk, so the CLIs are runnable with no GPU and no data |
| `summarize_photogrammetry_stats.py` | 105 | Reads `stats/*.json` written by hand-run stages |
| `bundle_adjust.py` | 98 | |
| `dense_mvs.py` | 90 | |

Also modified: `examples/simple_trainer_2dgs.py` (`--extract_mesh`,
`--mono_depth_loss`, `--mask_dir`), `examples/datasets/colmap.py`
(`mono_depth_dir`, `mask_dir`), `.github/workflows/core_tests.yml` (installs
the suite's deps).

### Tests — 194 across 10 files

| File | Tests | Lines |
|---|---:|---:|
| `tests/test_mesh_extraction.py` | 54 | 1957 |
| `tests/test_texturing.py` | 47 | 1806 |
| `tests/test_photogrammetry_pipeline.py` | 36 | 966 |
| `tests/test_photogrammetry_metrics.py` | 14 | 228 |
| `tests/test_colmap_dataset.py` | 13 | 568 |
| `tests/test_extract_mesh_cli.py` | 13 | 718 |
| `tests/test_extract_mesh_io.py` | 5 | 152 |
| `tests/test_photometric_alignment.py` | 5 | 440 |
| `tests/test_neural_sfm.py` | 4 | 243 |
| `tests/test_bundle_adjustment.py` | 3 | 166 |

### Docs

`docs/photogrammetry.md` (957) — user-facing feature doc ·
`docs/photogrammetry_status.md` (1145) — long-form running log ·
`docs/photogrammetry_texturing_plan.md` (368) — texturing design record ·
`docs/source/apis/photogrammetry.rst`, `docs/source/examples/photogrammetry.rst`
— Sphinx · `CLAUDE.md` — points a new session here.

---

## Module boundaries, and why they are where they are

`texturing.py` was split out of `mesh_extraction.py` when the latter hit 1431
lines holding two unrelated jobs. The split is along a real seam:

- **`mesh_extraction.py` owns the surface**: `extract_mesh_tsdf`,
  `extract_mesh_poisson`, `extract_level_set`, `refine_mesh_photometric`,
  `derive_reconstruction_parameters`, `_clean_mesh`, `cull_unobserved_faces`,
  `simplify_mesh`, `simplify_mesh_to_error`, `_point_spacing`.
- **`photometric_alignment.py` owns the cameras**, and is the only module that
  moves them after SfM. It imports `_view_samples` from `texturing` (lazily,
  inside functions) rather than reimplementing visibility.
- **`texturing.py` owns appearance**: everything that projects into a training
  view or writes into an atlas.
- **`_open3d.py`** holds only `_require_open3d()` so neither imports the other
  for it.
- `mesh_extraction.py` **re-exports** the moved names, because the example CLIs
  and the test suite import bakers from that path. Keep that working.

Cross-module imports that would be cycles are done **inside functions**:
`mesh_extraction` imports `point_to_mesh_distance`/`face_visibility` lazily;
`texturing` imports `atlas_sharpness`/`seam_discontinuity` lazily.

---

## Data flow through the texturing core

One function is the hub. Understand this and the rest follows:

```python
_view_samples(scene, o3d, dataset, points, normals, max_views, chunk_size)
    # yields (indices, colors, weights, view_index) per visible (point, view)
```

One pass over the dataset: project every point into each camera, drop
out-of-frame ones, **ray-cast away occluded ones**, weight the rest by
view/normal alignment and inverse distance. Colours are read with **bilinear**
interpolation (`_bilinear`).

Everything else is a consumer of it, so "visible" means one thing across the
whole module rather than three subtly different things:

| Consumer | Samples at | Produces |
|---|---|---|
| `bake_texture` | mesh vertices | per-vertex colours |
| `bake_texture_atlas` | atlas texel positions | blended UV atlas |
| `face_visibility` | face centroids | `(F, V)` bool — who can see what |
| `face_view_quality` | face centroids | `(F, V)` gradient energy over the projection |
| `face_projected_areas` | face corners | `(F,)` best-view pixel area = *evidence* |
| `level_seams` | points along seam edges | per-(vertex, label) colour corrections |
| `bake_texture_atlas_super_resolved` | atlas texel positions | deconvolved UV atlas |
| `refine_camera_poses_photometric` | surface sample points | refined `camtoworlds` |
| `refine_mesh_photometric` | mesh vertices | per-vertex photoconsistency |

**One caveat that has bitten twice**: `_view_samples` ray-casts each sample
against the mesh, so it answers "can this view see this *surface* point". A
point deliberately displaced *off* the surface reads as occluded by the surface
it came from. `refine_mesh_photometric` therefore resolves visibility once, at
the surface, and reuses it for every candidate offset (`ISSUES.md` § 4.26).

`_bake_points_from_views` wraps `_view_samples` with the weighted accumulation
and the optional iterative sigma-clipping, and takes an **`occluder`** argument
(defaulting to the mesh) — required so that one page of a multi-page atlas is
cast against the *whole* mesh, not against itself.

## The atlas path

`_unwrap_and_rasterize(mesh, texture_size, ...) -> _AtlasTexels` is shared by
every map, so albedo, normal and AO all land on **one UV layout**. It:

- **reuses `mesh.triangle_uvs` when present** (open3d's `compute_uvatlas` is
  non-deterministic — see [`ISSUES.md`](ISSUES.md));
- refuses non-manifold input up front (`compute_uvatlas` *segfaults*, it does
  not raise);
- does **not** mutate the mesh, so probing it is safe.

## The GPU/CPU seam

Nothing in this package has ever run on a GPU, so every stage that needs one is
split at a seam that leaves the majority testable. `_tsdf_fuse` set the
precedent — pure open3d/NumPy consuming `{"color", "depth", "K", "extrinsic"}`
dicts, so the fusion half is exercised with synthetic depth maps while the
rendering half is not. Two more follow it:

| Stage | GPU side (never executed) | CPU side (verified against analytic truth) |
|---|---|---|
| TSDF | rendering depth from the splats | `_tsdf_fuse`, and the parameter derivation inside it |
| Level sets | `gaussian_density_field` | `extract_level_set` — Kuhn decomposition, marching tetrahedra, shared-vertex identification, cleanup |

Put new GPU-dependent work on the same pattern: one thin function that touches
the field or the renderer, everything else consuming plain arrays.

## Measured, not guessed

The distinguishing design idea. Four decisions that used to be magic numbers
are now derived from the capture itself:

| Decision | Old | Now |
|---|---|---|
| How many triangles | `--target_triangles` | `--target_fit_ratio` — binary-search the count, re-measuring cloud-to-mesh fit at each probe; return a mesh whose error was **measured** |
| How big an atlas | `--texture_size` | `--texture_texels_per_pixel` — sum the source pixels covering the surface; measure this mesh's UV packing with a probe unwrap |
| Which faces to keep | (all) | `--cull_unobserved` — `face_visibility` over the training cameras |
| Which view textures a face | (blend all) | `--texture_view_selection` — MRF over face adjacency, then seam levelling |
| Where the cameras are | (whatever SfM said) | `--photometric_align` — move each camera until its image agrees with the baked surface colour |
| Where the vertices are | (whatever extraction said) | `--refine_mesh` — move each vertex along its normal to maximise multi-view photoconsistency |
| TSDF voxel / truncation / depth cap | `0.01` / `0.04` / `10.0` in scene units | derived from the point spacing and extent of the depth being fused |
| Poisson normal radius | `0.1` in scene units | 3 × the cloud's own point spacing |

---

## Public API — 47 names

```
Reconstruction   refine_reconstruction  run_dense_mvs
                 merge_point_maps_to_tracks  write_colmap_reconstruction

Surface          extract_mesh_tsdf  extract_mesh_poisson
                 cull_unobserved_faces  simplify_mesh  simplify_mesh_to_error
                 refine_mesh_photometric  derive_reconstruction_parameters
                 extract_level_set  gaussian_density_field

Registration     refine_camera_poses_photometric

Texturing        bake_texture  bake_texture_atlas  bake_texture_atlas_view_selected
                 bake_texture_atlas_super_resolved
                 bake_texture_atlas_pages  bake_mesh_texture
                 bake_normal_map  bake_ambient_occlusion
                 face_visibility  face_view_quality  face_projected_areas
                 recommended_texture_size  partition_faces
                 select_views_mrf  level_seams  NO_VIEW

Metrics          point_to_mesh_distance  mesh_quality_stats  point_cloud_stats
                 reconstruction_stats  track_stats
                 mask_coverage_stats  depth_prior_stats
                 atlas_sharpness  seam_discontinuity

Orchestration    PipelineReport  StageResult  run_stage  record_skipped
                 collect_artifact_metrics  check_prior_quality
                 derive_cross_stage_metrics  format_cross_stage_metrics
```

## Orchestration contract

`run_pipeline.py` runs `sfm_input → bundle_adjust → dense_mvs → priors → train
→ extract_mesh` as **subprocesses of the per-stage scripts**. A stage needing
something the machine lacks is recorded `skipped` with a reason rather than
failing (`--strict` to fail instead). The report is written **even when a stage
fails** — via `try`/`finally`, because losing it to a traceback is exactly when
it matters most.

Because the runner builds argv by hand, two tests guard the seam: every long
option it forwards must be one `extract_mesh.Config` accepts, and the delivery
flags must appear in the dry-run command. `--extract_mesh_extra_args` is the
escape hatch for options the runner does not name — **bind its first element
with `=`** (`--extract_mesh_extra_args=--texture_seam_smoothness 0.25`) or tyro
reads the leading `--` as a new option and rejects the call.
