# Features: Photogrammetry Pipeline

`gsplat.photogrammetry` closes the classic photogrammetry loop -- **SfM ->
bundle adjustment -> dense MVS -> Gaussian-splat training -> mesh
extraction/texturing** -- on top of the existing COLMAP data loader
(`examples/datasets/colmap.py`) and 2DGS/3DGS renderers. Each stage is
independently useful and wires into the existing pipeline with no other
changes required:

- **Bundle adjustment** (`gsplat.photogrammetry.bundle_adjustment`) refines
  COLMAP poses and 3D points via a differentiable, reprojection-error-based
  joint optimization over the SfM point tracks.
- **Dense MVS** (`gsplat.photogrammetry.dense_mvs`) densifies the sparse
  COLMAP point cloud via COLMAP's own patch-match stereo + fusion pipeline.
- **Mesh extraction** (`gsplat.photogrammetry.mesh_extraction`) extracts a
  cleaned, colored triangle mesh from a trained 2DGS/3DGS scene, via TSDF
  fusion of rendered depth/normal maps or Poisson reconstruction from a dense
  point cloud, with vertex-color texture baking from the training images.

`mesh_extraction` requires the optional `open3d` dependency: `pip install
gsplat[mesh]`. `dense_mvs` requires a CUDA-enabled `colmap` command-line
install (see https://colmap.github.io/install.html) -- `pycolmap` alone does
not expose patch-match stereo.

## How to Use

### For users directly running `examples` in gsplat:

Starting from a standard COLMAP-processed capture (`<data_dir>/images/` +
`<data_dir>/sparse/0/`, as used by `examples/simple_trainer.py`):

```bash
# 1. Bundle adjustment: refine poses + points, writing sparse/refined.
python examples/bundle_adjust.py --data_dir data/360_v2/garden

# 2. Dense MVS: densify the (refined) sparse point cloud.
python examples/dense_mvs.py --data_dir data/360_v2/garden \
    --colmap_dir data/360_v2/garden/sparse/refined

# 3. Train as usual, pointing the trainer at the refined poses via
#    Parser(..., colmap_dir=...) / dense-augmented init via
#    Parser(..., dense_points_path=...) -- see "For users using gsplat's API"
#    below for wiring these into examples/simple_trainer_2dgs.py's Config.
python examples/simple_trainer_2dgs.py \
    --data_dir data/360_v2/garden --data_factor 4 \
    --result_dir results/garden_2dgs

# 4. Mesh extraction: TSDF fusion of the trained 2DGS scene's rendered
#    depth maps, with texture baking from the training images.
python examples/extract_mesh.py \
    --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
    --data_dir data/360_v2/garden --result_dir results/garden_2dgs
```

This writes `data/360_v2/garden/sparse/refined/` (refined COLMAP model),
`data/360_v2/garden/dense/dense.ply` (dense point cloud), and
`results/garden_2dgs/mesh.ply` (textured mesh).

To reconstruct a mesh via Poisson reconstruction over the dense MVS cloud
instead of TSDF fusion:

```bash
python examples/extract_mesh.py --method poisson \
    --dense_points data/360_v2/garden/dense/dense.ply \
    --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
    --data_dir data/360_v2/garden --result_dir results/garden_2dgs
```

Mesh extraction currently only supports SH-color checkpoints (the default,
i.e. trained without `--app_opt`) -- appearance-embedding checkpoints are out
of scope, since per-image appearance variation doesn't map onto a single
canonical mesh texture.

### For users using gsplat's API:

- `gsplat.photogrammetry.bundle_adjustment.refine_reconstruction(colmap_dir,
  output_dir, ...)` reads a COLMAP sparse model and writes a refined one --
  point `examples.datasets.colmap.Parser` at the output via its new
  `colmap_dir` argument (`Parser(data_dir, colmap_dir=output_dir, ...)`).
- `gsplat.photogrammetry.dense_mvs.run_dense_mvs(data_dir, colmap_dir,
  output_dir, ...)` returns the path to a fused `dense.ply`. Pass it to
  `Parser`'s new `dense_points_path` argument
  (`Parser(data_dir, dense_points_path=dense_ply, dense_mode="augment", ...)`)
  to densify Gaussian initialization -- `create_splats_with_optimizers`'s
  existing `init_type="sfm"` path picks up the denser `parser.points`
  automatically, with no trainer changes needed.
- `gsplat.photogrammetry.mesh_extraction.extract_mesh_tsdf(splats, dataset,
  ...)` / `extract_mesh_poisson(points_xyz, points_rgb, ...)` /
  `bake_texture(mesh, dataset, ...)` operate on a loaded checkpoint's
  `"splats"` state dict and an `examples.datasets.colmap.Dataset`, returning
  `open3d.geometry.TriangleMesh` objects.
