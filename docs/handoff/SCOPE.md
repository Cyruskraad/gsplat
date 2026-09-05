# Scope

## The problem this solves

gsplat already implements most state-of-the-art Gaussian-splatting *training*
techniques — 3DGS adaptive density control, MCMC densification, 2DGS surfels,
3DGUT, Mip-splatting antialiasing, pose and appearance optimisation,
bilateral-grid post-processing, depth losses. What it did not have was the
classic **photogrammetry loop around** that training: refining the SfM poses
that training starts from, densifying the sparse point cloud, and turning a
trained radiance field into an actual deliverable *surface asset* that loads in
Blender, Unreal or Unity.

This project adds that loop as `gsplat.photogrammetry`, wiring into the
existing COLMAP `Parser`/`Dataset` and the 2DGS/3DGS renderers with no changes
required elsewhere.

## The pipeline

```
SfM (COLMAP or neural-SfM)
  └─> bundle adjustment          refine poses + points against reprojection error
       └─> dense MVS             densify the cloud (COLMAP patch-match stereo)
            └─> 2DGS/3DGS train  optionally with mono-depth priors + transient masks
                 └─> mesh extraction        TSDF fusion, or Poisson from the dense cloud
                      └─> cull unobserved   drop faces no camera ever saw
                           └─> decimate     to a triangle budget or a measured fit target
                                └─> texture albedo atlas (blended or per-face view-selected)
                                     └─> maps  normal (8/16-bit) + ambient occlusion
                                          └─> mesh.obj + .mtl + PNGs
```

Every stage writes automatic quality metrics into `stats/*.json`, and the
orchestrator collects them plus **cross-stage** comparisons into one
`pipeline_report.json`.

## Design goals

1. **Each stage is independently useful.** You can run bundle adjustment alone,
   or texture a mesh you already have. The orchestrator composes the per-stage
   CLIs as subprocesses rather than reimplementing them, so each stage's own CLI
   stays the source of truth for its options.
2. **Every decision is measured, not guessed.** Where the pipeline once took a
   magic number (`--target_triangles`, `--texture_size`), it now takes the
   *outcome you want* and measures its way there. See
   [`ARCHITECTURE.md`](ARCHITECTURE.md) § "Measured, not guessed".
3. **A number in scene units means nothing on its own.** Quality metrics are
   scale-free wherever possible — cloud-to-mesh distance is reported over the
   cloud's own k-NN spacing, so "1.0" means the same thing on a tabletop scan
   and a city block.
4. **Failures are legible.** A stage that cannot run is recorded `skipped` with
   a reason; the report is written even when a stage fails; degenerate input
   raises naming the likely cause rather than dying in a library.

## Explicit non-goals

These are deliberate, not oversights. Do not "fix" them without being asked.

- **No bundled model-running code.** gsplat never runs a neural network itself.
  Monocular depth priors, transient-object masks and neural-SfM output are all
  consumed as *precomputed files*. This is an existing, documented repo
  convention (`docs/source/examples/dynamic_surgical.rst`: "gsplat does not ship
  code for estimating depth"; `docs/source/proposals/gsharp_v0_2_port.rst` lists
  bundling Depth Anything V2 / VGGT as a non-goal). `dense_mvs.py` follows the
  same pattern for COLMAP's dense stereo — it shells out to the real `colmap`
  CLI rather than reimplementing patch-match.
- **External tools stay external.** No reimplementation of COLMAP, no bundled
  glTF writer (open3d cannot write textured glTF; converting the OBJ with
  trimesh/Blender/gltfpack is the user's job).
- **No new required dependencies.** `gsplat[mesh]` is deliberately just
  `open3d` + `imageio`. The MRF optimiser and the linear solver are hand-rolled
  pure NumPy specifically to avoid a scipy hard dependency. scipy *is*
  installed and is in the `lidar` extra — acceptable as an optional
  accelerator, never as a hard import.
- **Appearance-embedding checkpoints are out of scope** for mesh extraction:
  per-image appearance variation does not map onto one canonical mesh texture.
  Resolving it (e.g. baking at a canonical embedding) is unexplored and needs
  a GPU to evaluate.

## Optional dependencies

| Module | Needs |
|---|---|
| `mesh_extraction`, `texturing`, `metrics` (geometry) | `open3d` (`pip install gsplat[mesh]`) |
| `neural_sfm` merge step, `metrics.point_cloud_stats` | `scikit-learn` |
| `dense_mvs` | a **CUDA-enabled** `colmap` CLI (`pycolmap` alone does not expose patch-match stereo) |
| `pipeline` | pure stdlib, always importable |
| 16-bit normal-map writing (CLI) | `opencv` — Pillow cannot write 16-bit RGB PNG |
