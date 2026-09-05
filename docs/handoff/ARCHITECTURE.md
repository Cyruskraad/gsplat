# Scaffolding

## File inventory

Everything below is new on this branch unless marked *(modified)*.

### Library — `gsplat/photogrammetry/` (5,422 lines)

| File | Lines | Holds |
|---|---:|---|
| `texturing.py` | 2551 | Everything that samples training views or writes an atlas |
| `mesh_extraction.py` | 732 | Surface reconstruction, culling, decimation |
| `metrics.py` | 669 | Automatic quality metrics for every stage |
| `pipeline.py` | 545 | Pure-stdlib orchestration layer (timing, status, report) |
| `bundle_adjustment.py` | 311 | Torch-native reprojection-error BA over COLMAP tracks |
| `neural_sfm.py` | 299 | Adapter for externally-run DUSt3R/MASt3R/VGGT-style tools |
| `__init__.py` | 146 | Re-exports; 41 public names |
| `dense_mvs.py` | 138 | Shells out to the `colmap` CLI for patch-match stereo + fusion |
| `_open3d.py` | 31 | The shared `_require_open3d()` guard, so extraction and texturing need not depend on each other |

### CLIs — `examples/` (1,434 lines)

| File | Lines | Purpose |
|---|---:|---|
| `extract_mesh.py` | 612 | The delivery stage; 32 flags, the richest CLI |
| `run_pipeline.py` | 529 | Orchestrator — runs the per-stage scripts as subprocesses |
| `summarize_photogrammetry_stats.py` | 105 | Reads `stats/*.json` written by hand-run stages |
| `bundle_adjust.py` | 98 | |
| `dense_mvs.py` | 90 | |

Also modified: `examples/simple_trainer_2dgs.py` (`--extract_mesh`,
`--mono_depth_loss`, `--mask_dir`), `examples/datasets/colmap.py`
(`mono_depth_dir`, `mask_dir`), `.github/workflows/core_tests.yml` (installs
the suite's deps).

### Tests — 156 across 9 files

| File | Tests | Lines |
|---|---:|---:|
| `tests/test_texturing.py` | 42 | 1487 |
| `tests/test_mesh_extraction.py` | 36 | 1368 |
| `tests/test_photogrammetry_pipeline.py` | 36 | 949 |
| `tests/test_photogrammetry_metrics.py` | 14 | 228 |
| `tests/test_colmap_dataset.py` | 13 | 568 |
| `tests/test_extract_mesh_io.py` | 5 | 152 |
| `tests/test_extract_mesh_cli.py` | 3 | 170 |
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
  `extract_mesh_poisson`, `_clean_mesh`, `cull_unobserved_faces`,
  `simplify_mesh`, `simplify_mesh_to_error`, `_point_spacing`.
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

## Measured, not guessed

The distinguishing design idea. Four decisions that used to be magic numbers
are now derived from the capture itself:

| Decision | Old | Now |
|---|---|---|
| How many triangles | `--target_triangles` | `--target_fit_ratio` — binary-search the count, re-measuring cloud-to-mesh fit at each probe; return a mesh whose error was **measured** |
| How big an atlas | `--texture_size` | `--texture_texels_per_pixel` — sum the source pixels covering the surface; measure this mesh's UV packing with a probe unwrap |
| Which faces to keep | (all) | `--cull_unobserved` — `face_visibility` over the training cameras |
| Which view textures a face | (blend all) | `--texture_view_selection` — MRF over face adjacency, then seam levelling |

---

## Public API — 41 names

```
Reconstruction   refine_reconstruction  run_dense_mvs
                 merge_point_maps_to_tracks  write_colmap_reconstruction

Surface          extract_mesh_tsdf  extract_mesh_poisson
                 cull_unobserved_faces  simplify_mesh  simplify_mesh_to_error

Texturing        bake_texture  bake_texture_atlas  bake_texture_atlas_view_selected
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
