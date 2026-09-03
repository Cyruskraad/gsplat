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
- **Automatic metrics** (`gsplat.photogrammetry.metrics`) reports
  quantitative quality stats -- mesh watertightness/connected-components,
  cloud-to-mesh fit, point-cloud density -- for the stages above, written to
  `stats/*.json` files next to the trainer's existing PSNR/SSIM/LPIPS
  render-quality reports.
- **Transient/dynamic-object masking** (`Dataset(..., mask_dir=...)`,
  `--mask_dir`) excludes externally-segmented moving content (people,
  vehicles, ...) from training supervision and mesh fusion via
  `gsplat.losses.masked_l1`/`masked_ssim`.

`mesh_extraction`/`metrics` require the optional `open3d` dependency: `pip
install gsplat[mesh]`. `dense_mvs` requires a CUDA-enabled `colmap`
command-line install (see https://colmap.github.io/install.html) --
`pycolmap` alone does not expose patch-match stereo.

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
`results/garden_2dgs/mesh.ply` (textured mesh) -- and, alongside each,
automatic quality stats: `.../sparse/refined/bundle_adjust_stats.json`
(reprojection error before/after), `.../dense/dense_stats.json` (point
count, k-NN density), and `results/garden_2dgs/mesh_metrics.json`
(watertightness, connected components, cloud-to-mesh fit against the sparse
or dense cloud).

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

Steps 3-4 can also be collapsed into one command: pass `--extract_mesh` to
`simple_trainer_2dgs.py` to run TSDF extraction + texture baking
automatically at the end of training (writing `mesh_<step>.ply` alongside the
checkpoint), instead of a separate `extract_mesh.py` call:

```bash
python examples/simple_trainer_2dgs.py \
    --data_dir data/360_v2/garden --data_factor 4 \
    --result_dir results/garden_2dgs --extract_mesh
```

This shortcut only covers the TSDF path (`--mesh_bake_texture`,
`--mesh_voxel_size`, `--mesh_sdf_trunc` mirror `extract_mesh.py`'s options) --
Poisson reconstruction and the dense-MVS point cloud path still need the
standalone `examples/extract_mesh.py` script, since the trainer has no dense
point cloud of its own to reconstruct from. It writes mesh quality stats to
`results/garden_2dgs/stats/mesh_step<step>.json`, next to `eval()`'s own
`stats/val_step<step>.json` render-quality reports.

### Automatic metrics & the consolidated report

Every stage above that produces a geometric artifact (bundle adjustment,
dense MVS, mesh extraction) now writes a `stats/*.json` file of automatic
quality metrics (`gsplat.photogrammetry.metrics`) next to its output, using
the same convention the trainer's `eval()` already uses for render quality
(PSNR/SSIM/LPIPS). To pull everything for one run together into a single
report:

```bash
python examples/summarize_photogrammetry_stats.py \
    --result_dir results/garden_2dgs --data_dir data/360_v2/garden
```

This finds and aggregates whichever of `bundle_adjust_stats.json`,
`dense_stats.json`, `mesh_metrics.json`/`mesh_step*.json`, and
`val_step*.json` are present under `--result_dir`/`--data_dir`, prints a
summary table, and writes `results/garden_2dgs/pipeline_report.json` --
mirroring `examples/benchmarks/compression/summarize_stats.py`'s
read-then-write pattern.

### Monocular depth-prior supervision (AI-assisted)

gsplat does not run any depth-estimation model itself -- consistent with how
`dense_mvs.py` shells out to `colmap` rather than reimplementing MVS, this is
"bring your own precomputed depth maps." Run a monocular depth model
externally over your training images, save one `<image_stem>.npy` (float32,
any resolution -- it's resized to match automatically) per image into a
directory, then pass `--mono_depth_loss --mono_depth_dir <that directory>` to
`simple_trainer_2dgs.py`. This supervises the *full* rendered depth map
against the prior via `gsplat.losses.pearson_depth_loss` -- scale/shift
invariant, since monocular predictions are only *relative*-depth accurate --
additive to (not a replacement for) `--depth_loss`'s existing sparse
COLMAP-point supervision.

Example using Depth Anything V2 via HuggingFace `transformers` (a separate
install -- `pip install transformers`; not a gsplat dependency) to produce
compatible `.npy` files:

```python
import os
import numpy as np
from PIL import Image
from transformers import pipeline

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image_dir, out_dir = "data/360_v2/garden/images", "data/360_v2/garden/mono_depth"
os.makedirs(out_dir, exist_ok=True)
for fname in os.listdir(image_dir):
    depth = pipe(Image.open(os.path.join(image_dir, fname)))["predicted_depth"]
    stem = os.path.splitext(fname)[0]
    np.save(os.path.join(out_dir, f"{stem}.npy"), depth.numpy().astype(np.float32))
```

```bash
python examples/simple_trainer_2dgs.py \
    --data_dir data/360_v2/garden --data_factor 4 \
    --result_dir results/garden_2dgs \
    --mono_depth_loss --mono_depth_dir data/360_v2/garden/mono_depth
```

### Transient/dynamic-object masking (AI-assisted)

Real captures often contain moving people, vehicles, or pets that corrupt a
static-scene reconstruction. As with monocular depth, gsplat does not run
any segmentation model itself -- run one externally over your training
images and save one `<image_stem>.png` mask per image into a directory
(any resolution -- resized/warped to match automatically), where **nonzero
= keep (static content)**, **0 = exclude (transient content)**. This
matches the convention of the existing fisheye undistortion ROI mask, which
this feature composes with (both are combined, so a fisheye capture with
moving people gets both border cropping and transient exclusion). Pass
`--mask_dir <that directory>` to `simple_trainer_2dgs.py`, `extract_mesh.py`,
or `Parser`'s `Dataset(..., mask_dir=...)` directly.

Excluded pixels are dropped from the photometric loss
(`gsplat.losses.masked_l1`/`masked_ssim` -- mean taken over the kept region
only, rather than diluting the loss by zeroing pixels and averaging over the
full frame) and from `--mono_depth_loss`'s depth-prior supervision, and (via
`Runner.extract_mesh()`/`extract_mesh.py --mask_dir`) from TSDF mesh fusion,
so transient content isn't baked into an extracted mesh either. There is no
separate enable flag -- masks are used whenever `--mask_dir` is given.

Example using `torchvision`'s pretrained Mask R-CNN (COCO instance
segmentation; not a gsplat dependency) to exclude common movable classes:

```python
import os
import numpy as np
import torch
from PIL import Image
from torchvision.io import decode_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# COCO category ids for commonly-moving objects (person, bicycle, car,
# motorcycle, bus, train, truck, boat, cat, dog).
MOVABLE_CATEGORY_IDS = {1, 2, 3, 4, 6, 7, 8, 9, 17, 18}

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).eval()
preprocess = weights.transforms()

image_dir, out_dir = "data/360_v2/garden/images", "data/360_v2/garden/masks"
os.makedirs(out_dir, exist_ok=True)
for fname in os.listdir(image_dir):
    image = decode_image(os.path.join(image_dir, fname))
    with torch.no_grad():
        pred = model([preprocess(image)])[0]
    keep_mask = np.ones(image.shape[1:], dtype=bool)  # (H, W), True = keep
    for label, score, obj_mask in zip(pred["labels"], pred["scores"], pred["masks"]):
        if score > 0.7 and int(label) in MOVABLE_CATEGORY_IDS:
            keep_mask &= obj_mask[0].numpy() < 0.5  # exclude this instance
    stem = os.path.splitext(fname)[0]
    Image.fromarray((keep_mask * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"{stem}.png")
    )
```

```bash
python examples/simple_trainer_2dgs.py \
    --data_dir data/360_v2/garden --data_factor 4 \
    --result_dir results/garden_2dgs --mask_dir data/360_v2/garden/masks
```

### Starting from a neural SfM tool instead of COLMAP (AI-assisted)

Feed-forward neural SfM tools (DUSt3R/MASt3R/VGGT-style) predict per-image
camera poses and a dense 3D point *per pixel* directly, without COLMAP's
incremental matching/triangulation. gsplat doesn't run any such tool itself
(same convention as above) -- `gsplat.photogrammetry.neural_sfm` is a
tool-agnostic adapter that takes plain arrays and produces a normal COLMAP
model, so the rest of the pipeline (most usefully, bundle adjustment --
neural-SfM poses are typically less precise than classical SfM and benefit
from it) works unchanged:

```python
from gsplat.photogrammetry.neural_sfm import (
    merge_point_maps_to_tracks,
    write_colmap_reconstruction,
)
from gsplat.photogrammetry.bundle_adjustment import refine_reconstruction

# 1. Run your neural SfM tool externally, and extract, per image:
#    - a (N_i, 3) array of 3D points (already in one shared world frame,
#      as these tools' own global alignment produces)
#    - a matching (N_i, 2) array of the pixel each point came from
#    - (recommended) a (N_i,) confidence array
# points_per_image, pixel_xy_per_image, confidence_per_image = ...  # your adapter code

# 2. Merge each image's independent per-pixel points into cross-view tracks.
#    min_track_length=2 is important -- a point only one image "saw" gives
#    bundle adjustment no cross-view constraint.
merged = merge_point_maps_to_tracks(
    points_per_image, pixel_xy_per_image,
    confidence_per_image=confidence_per_image, confidence_threshold=0.5,
    merge_radius=0.01, min_track_length=2, max_points_per_image=2000,
)

# 3. Write it as a COLMAP model.
write_colmap_reconstruction(
    image_names=image_names,        # must match files under <data_dir>/images/
    camtoworlds=camtoworlds,        # (N, 4, 4), from the neural SfM tool
    Ks=Ks,                          # (N, 3, 3) or (3, 3)
    image_sizes=(width, height),
    points_xyz=merged["points_xyz"],
    tracks=merged["tracks"],
    output_dir="data/my_scene/sparse/0",
)

# 4. Refine the (typically approximate) neural-SfM poses -- exactly the same
#    bundle adjustment step used after COLMAP.
refine_reconstruction(
    colmap_dir="data/my_scene/sparse/0",
    output_dir="data/my_scene/sparse/refined",
)
```

From there, `examples/dense_mvs.py`, `examples/simple_trainer_2dgs.py`
(`--colmap_dir data/my_scene/sparse/refined` via
`examples.datasets.colmap.Parser`'s `colmap_dir` argument), and
`examples/extract_mesh.py` all work exactly as they would starting from
COLMAP. gsplat provides the merge + COLMAP-writing primitives above; a
tool-specific script converting DUSt3R/MASt3R/VGGT's own output format into
the plain `points_per_image`/`pixel_xy_per_image`/`camtoworlds` arrays is
something you write yourself (or find in that tool's own repo) -- not
something gsplat ships.

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
- `gsplat.photogrammetry.metrics.point_to_mesh_distance(points, mesh, ...)` /
  `mesh_quality_stats(mesh)` / `point_cloud_stats(points, ...)` return plain
  dicts of quality stats for a mesh or point cloud, independent of how it was
  produced -- the functions the CLIs above write to `stats/*.json` with.
