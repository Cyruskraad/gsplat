# Fixed-light object reconstruction pipeline

## Purpose and scope

This branch extends gsplat's COLMAP example into a reproducible fixed-light
object-reconstruction workflow. It targets novel-view rendering from calibrated
photographs and binary object masks. Relighting, UV material recovery, mesh
extraction, and claims of watertight geometry are outside its scope.

The final representation is a set of anisotropic 3D Gaussians with opacity,
orientation, scale, and spherical-harmonic appearance. COLMAP supplies camera
intrinsics, camera poses, and sparse initialization; differentiable Gaussian
rasterization fits the dense appearance representation to the photographs.

## Pipeline changes

### Dataset contract and preprocessing

- `--object-mask-dir` matches white-foreground masks to images by relative stem.
- `--split-manifest` loads explicit train, validation, and sealed-test view lists.
- `--train-all-views` supports production retraining after scientific selection.
- Lens-valid masks and object masks remain separate.
- Images and masks receive the same resize, undistortion, and crop transforms;
  masks use nearest-neighbour sampling.
- Sparse COLMAP points are retained only when their observations fall inside the
  object masks.
- `tools/prepare_gsplat_hq.py` validates filenames, inspects mask fill/connectivity
  and border contact, stages the dataset without copying it, and produces a
  deterministic camera-pose farthest-point split.

For the 115-view AngelDemon capture, the deterministic split is 92 training,
12 validation, and 11 sealed-test cameras.

### Mask-aware objective

When object masks are present, the trainer composites foreground RGB and target
RGB onto the same randomized background. It optimizes the standard
`0.8 * L1 + 0.2 * (1 - SSIM)` appearance objective on a 10%-padded foreground
crop and adds full-valid-image binary cross-entropy between rendered alpha and
the object mask. The default alpha-loss weight is `0.1`.

The correction controls `--rgb-mask-erosion-pixels` and
`--alpha-boundary-band-pixels` support a one-time silhouette-focused rerun when
the baseline misses its alpha gate. They are explicit options rather than an
unrecorded preprocessing change.

### Evaluation and model selection

Evaluation returns data programmatically and writes aggregate plus per-view
records. The object-aware report includes:

- foreground PSNR, SSIM, and LPIPS;
- alpha intersection-over-union and boundary F-score;
- mean, median, and worst-view summaries;
- Gaussian population, render time/FPS, and peak PyTorch VRAM;
- full-frame metrics for backwards comparison only.

`--save-best` uses the declared ordering: require the minimum alpha IoU, minimize
validation LPIPS, prefer PSNR within the LPIPS tie, then prefer fewer Gaussians
within the PSNR tie. Early stopping counts an evaluation as progress only when
LPIPS improves by at least `0.005` or PSNR by at least `0.1 dB`; patience is
configurable and was four evaluations in the production protocol.

`tools/export_gsplat_metrics.py` exports checkpoint and per-view CSV files from
the JSON evidence without changing the underlying measurements.

### Resume and artifact safety

- `--resume` restores splats and the completed step.
- `--resume-optimizer` restores optimizer, scheduler, and strategy state for an
  interrupted run.
- `--resume-lr-scale` supports low-rate native-resolution refinement.
- Checkpoints sanitize and serialize resumable strategy state.
- PLY export, trajectory video output, run logs, configuration, provenance,
  telemetry, and terminal `RUNNING`, `COMPLETED`, or `FAILED` markers are kept in
  non-overwriting run directories.
- `tools/run_gsplat_job.py` enforces GPU, free-disk, and projected-artifact
  preflight checks before launching a job.

### Bounded strategies and rendering

The workflow supports the stock strategy, AbsGS, and 3DGS-MCMC. Production uses
packed antialiased rasterization. MCMC population caps and explicit refinement
and noise-injection stop iterations prevent late unbounded growth. AbsGS remains
available for fine-detail experiments using absolute screen-space gradients.

Pose, appearance, and bilateral optimization remain opt-in. They were disabled
for the fixed-light AngelDemon run because its COLMAP geometry was already strong
and a pose ablation was not justified by repeatable double edges.

## AngelDemon production configuration

The production HQ model used all 115 registered views at factor 2
(4128 x 2752), capped MCMC at 600,000 Gaussians, stopped refinement/noise
injection at 15,000 steps, and trained to step 15,999. The compact model used a
300,000-Gaussian cap and 15,000 steps. Both use SH degree 3, packed tensors,
antialiased rasterization, randomized backgrounds, and mask/alpha supervision.

Scientific quality claims belong only to the frozen 92-train-view models:

| Model | Sealed-test masked PSNR | SSIM | LPIPS | Alpha IoU | Gaussians | FPS |
|---|---:|---:|---:|---:|---:|---:|
| HQ | 20.912 dB | 0.9112 | 0.1405 | 0.9732 | 600,000 | 91.7 |
| Compact | 20.804 dB | 0.9046 | 0.1547 | 0.9725 | 300,000 | 98.7 |

The production all-view retrains are delivery models and must not inherit those
held-out metrics. The compact model is 50% smaller and met the predefined
quality/efficiency rule. Native-resolution HQ refinement was rejected because
it worsened perceptual quality and silhouette boundaries.

## Known limitations

The preregistered mean alpha-IoU threshold of `0.98` was not reached. The
prescribed stronger silhouette correction was run once and rejected because it
worsened IoU and image quality. The frozen models retain small exterior remnants
in difficult oblique views but do not reproduce the severe late-training floater
growth seen in the old RedObject model.

This is a view-synthesis reconstruction, not a triangle mesh. Unseen or weakly
observed surfaces are not guaranteed to be complete, metrically exact, or
watertight, and the fixed capture lighting is baked into the appearance.

## Principal implementation files

- `examples/datasets/colmap.py` — mask/split loading, transforms, and sparse
  point filtering.
- `examples/simple_trainer.py` — losses, evaluation, checkpoint selection,
  resume, early stopping, telemetry, and exports.
- `tools/colmap_scripts/gsplat_hq.py` — deterministic split, mask audit, and
  selection helpers.
- `tools/prepare_gsplat_hq.py` — dataset preflight and staging.
- `tools/run_gsplat_job.py` — immutable launch and resource controls.
- `tools/export_gsplat_metrics.py` — reproducible JSON-to-CSV reporting.
- `tools/generate_sam2_masks.py` and `tools/generate_mesh_silhouettes.py` —
  optional mask-generation paths with resumable output handling.

