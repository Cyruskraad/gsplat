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
  quantitative quality stats for every stage -- SfM/track quality, mesh
  watertightness/connected-components, cloud-to-mesh fit, point-cloud
  density, AI-prior coverage -- written to `stats/*.json` files next to the
  trainer's existing PSNR/SSIM/LPIPS render-quality reports.
- **Transient/dynamic-object masking** (`Dataset(..., mask_dir=...)`,
  `--mask_dir`) excludes externally-segmented moving content (people,
  vehicles, ...) from training supervision and mesh fusion via
  `gsplat.losses.masked_l1`/`masked_ssim`.
- **Pipeline orchestration** (`gsplat.photogrammetry.pipeline`,
  `examples/run_pipeline.py`) chains every stage above into one command,
  wiring each stage's output into the next and recording per-stage status,
  timing and metrics into a single `pipeline_report.json`.

`mesh_extraction`/`metrics` require the optional `open3d` dependency: `pip
install gsplat[mesh]`. `dense_mvs` requires a CUDA-enabled `colmap`
command-line install (see https://colmap.github.io/install.html) --
`pycolmap` alone does not expose patch-match stereo. `pipeline` itself is
pure stdlib and always importable.

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

# 3. Train, pointing the trainer at the refined poses (--colmap_dir) and
#    densifying Gaussian initialization from the dense cloud
#    (--dense_points_path); both are optional -- omit either to train on
#    the un-refined/un-densified input instead.
python examples/simple_trainer_2dgs.py \
    --data_dir data/360_v2/garden --data_factor 4 \
    --result_dir results/garden_2dgs \
    --colmap_dir data/360_v2/garden/sparse/refined \
    --dense_points_path data/360_v2/garden/dense/dense.ply

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

#### Texture: per-vertex colors or a UV atlas

By default the extracted mesh carries **per-vertex colors**, whose effective
resolution is the mesh's own vertex density: detail finer than the spacing
between vertices is averaged away, no matter how sharp the input images are.
Pass `--texture_mode atlas` to instead UV-unwrap the mesh and bake one color
per *texel*:

```bash
python examples/extract_mesh.py \
    --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
    --data_dir data/360_v2/garden --result_dir results/garden_2dgs \
    --texture_mode atlas --texture_size 4096
```

`run_pipeline.py` takes the same two flags and forwards them to this stage,
so the one-command path can produce an atlas too:

```bash
python examples/run_pipeline.py \
    --data_dir data/360_v2/garden --result_dir results/garden_pipeline \
    --texture_mode atlas --texture_size 4096
```

This writes `mesh.obj`, `mesh.mtl` and `mesh_0.png` (instead of `mesh.ply`,
which cannot carry UVs or a texture image), so the result opens with its
texture already attached in Blender, MeshLab, Unreal, Unity and anything else
that reads OBJ. Texture detail is then set by `--texture_size` rather than by
how finely the mesh happens to be tessellated, which usually means a smaller,
cheaper mesh can carry the same visual detail.

Both paths use the same occlusion-aware, view-weighted blend across training
images (each sample weighted by view-direction/surface-normal alignment and
inverse distance, with points occluded in a given view ray-cast away), so
switching modes changes the *resolution* of the texture, not its colors.
Baked colors are grown a few texels outward across UV seams and into patches
no camera observed, so bilinear sampling and mipmapping don't bleed
background into the surface.

#### Robust multi-view fusion

Both texture paths blend every view that sees a point, weighted by
view-direction alignment and distance. On a real capture the views *disagree*:
something walks through the scene, a surface goes specular, one camera is
slightly misregistered. A plain mean has no way to prefer the majority, so it
blends the disagreement in as ghosting and blur.

`--texture_outlier_sigma 1.5` discards observations more than that many
standard deviations from a point's own mean colour, re-estimates from the
survivors, and repeats. One pass is deliberately not enough: the bad samples
inflate the very spread they are measured against, so they sit right at the
threshold and mostly survive. Re-centring on the survivors shrinks the spread
until they separate cleanly. On a synthetic capture with an occluder covering
part of a sixth of the frames, this cut mean colour error against ground truth
by **3x** (0.045 to 0.015); with more views the gap widens.

It only recovers points whose bad observations are a **minority**. A surface
hidden behind something in most of the views that see it has no majority to
fall back on, and clipping will happily converge on the occluder instead --
that case is what `--mask_dir` transient masking is for. The two are
complements: masking removes the wholesale occlusions, robust fusion cleans up
the residual disagreement masking doesn't catch. Each extra round costs one
more pass over the dataset.

#### Per-face view selection (sharper, but a tradeoff)

Blending is the wrong operator for *sharpness*, and robust fusion cannot fix
that. Two views of the same surface point are never registered to sub-pixel
accuracy after real SfM, so averaging them is a low-pass filter: the atlas
comes out systematically blurrier than any single source photograph, and
sigma-clipping only removes *outliers* -- the surviving inliers are still
averaged, and averaging slightly-misaligned inliers is exactly what destroys
high-frequency detail.

`--texture_view_selection` does what production photogrammetry texturers do
(Waechter et al., *Let There Be Color!*, ECCV 2014): choose **one** view per
face, so each face is textured from a single un-averaged photograph. The
choice is an MRF over the mesh's face-adjacency graph, minimising

```
E(labels) = sum_f -log(quality[f, label_f])  +  lambda * (number of seams)
```

where `quality[f, v]` is the gradient energy of view `v` over face `f`'s
projection -- one number that rewards being close, fronto-parallel *and* in
focus, so a motion-smeared view loses to a sharper one with worse geometry.
`--texture_mrf_lambda` is that lambda: raise it for fewer, larger single-view
regions.

**This is a genuine tradeoff, not a strict improvement, which is why it is off
by default.** Measured on a synthetic sphere whose cameras were rotated to
simulate residual pose error, at 45 arcminutes:

| | detail retained (gradient) | pointwise error (L1) |
|---|---|---|
| blended | 59% | **0.171** |
| view-selected | **106%** | 0.199 |

Blending *attenuates* detail; single-view sampling *displaces* it. A
displaced-but-sharp texture scores worse pointwise than a blurred one even
though it looks far better. Choose view selection for an asset that will be
*looked at*, and blending for one that will be *measured*. The
`view_selection` block in `mesh_metrics.json` reports both numbers
(`atlas_sharpness` vs `blended_atlas_sharpness`) for the actual capture, and
the CLI warns if view selection came out *less* sharp -- which means the poses
were already well registered and blending is the better choice for that scene.

Faces no single view can texture, and texels their face's chosen view cannot
see, keep the **blended** colour: averaging a handful of views is the right
answer where one view is unavailable. So `--texture_outlier_sigma` and
`--texture_view_selection` are complements, not alternatives -- robust
blending still governs the fallback regions.

#### Seam levelling

One view per face means neighbouring faces textured from different photographs
meet at a visible step, because the two cameras disagree about exposure and
white balance. `--texture_seam_smoothness` controls the correction that removes
it: an additive colour offset solved per **(vertex, label)** pair — not per
vertex, since a single per-vertex offset provably cannot close a discontinuity
*at* that vertex (both sides would receive it and the step would survive).

```
E(g) = Σ_seam edges, at each endpoint v  ‖(g[v,l1] − g[v,l2]) + (mean_l1 − mean_l2)‖²
     + λ_s Σ_edges of a face labelled l  ‖g[v,l] − g[w,l]‖²
```

`mean_l1` and `mean_l2` are each view's **mean colour along the shared edge**,
over the same sample points. That detail is load-bearing, not incidental:
comparing the two views *at the shared vertex* instead gives a target dominated
by noise. Measured on the synthetic sphere, two views of one vertex disagree by
0.288 (L2 over RGB) from pixel quantisation and silhouette bleed alone, where
the exposure difference being corrected is 0.26 — the solve then fits noise
larger than the signal and makes the atlas *worse*. Averaging along the edge is
what separates them.

The result is a linear least squares, solved through its normal equations with
a hand-rolled conjugate gradient (~50 lines) that applies `AᵀA` as a matvec and
never materialises the matrix — scipy's `cg` would do, but is not a hard
dependency here. On the synthetic sphere with ±0.15 per-view exposure this
halves the measured `seam_discontinuity` (0.26 → 0.12) *and* brings the atlas
closer to the mean-exposure ground truth (L1 0.078 → 0.052), so it is removing
real error rather than hiding a boundary. The result is robust across λ_s from
0.003 to 1; the default is 0.1.

`seam_discontinuity` never reaches zero — two samples either side of a border
are different surface points, so the texture's own detail sets a floor. Read
the before/after pair, which `mesh_metrics.json` reports, rather than the
absolute number.

The labelling uses ICM (iterated conditional modes) rather than the
alpha-expansion graph cut the paper uses, because alpha-expansion needs a
max-flow solver and `gsplat[mesh]` is deliberately just `open3d` + `imageio`.
ICM has no optimality bound where alpha-expansion is within a known factor of
the global optimum; it is run from several seeds (the per-face best, and
"every face takes view alpha" for the strongest few views) because a single
greedy sweep is badly seed-dependent under a strong seam penalty. The
practical gap is a few extra seams.

#### Decimation + normal maps (the delivery path)

TSDF and Poisson extraction tessellate to the voxel grid, not to the scene's
actual complexity -- routinely millions of triangles for a scene a few hundred
thousand would describe. The standard photogrammetry answer is not to extract
a coarser mesh (that loses the detail) but to **decimate and bake the removed
detail into a normal map**, so the light mesh still *shades* like the dense
one:

```bash
python examples/extract_mesh.py \
    --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
    --data_dir data/360_v2/garden --result_dir results/garden_2dgs \
    --texture_mode atlas --texture_size 4096 \
    --target_triangles 200000 --normal_map
```

`--target_triangles` decimates via Garland & Heckbert quadric error metrics
(collapsing the cheapest edges first, so flat regions lose triangles and
detailed ones keep them), and `--normal_map` bakes the *pre-decimation* mesh's
normals onto the decimated mesh's UV atlas.

##### Decimating to a fit target instead of a triangle count

A triangle budget is the wrong question to have to answer: how many triangles a
scene needs depends on the scene, and the number is only checked *afterwards*,
by measuring the cloud-to-mesh fit — which is the thing actually cared about.
`--target_fit_ratio` inverts that. Give it the fit you are willing to accept
and it finds the smallest mesh that still delivers it:

```bash
python examples/extract_mesh.py --ckpt ... --data_dir ... --result_dir ...     --texture_mode atlas --target_fit_ratio 1.0 --normal_map
```

The target is scale-free — cloud-to-mesh distance measured **in units of the
reference cloud's own k-NN spacing**, the same reading as the pipeline report's
`mesh_fit_over_point_spacing`. At or below ~1 the mesh tracks the cloud to
within its own sampling noise, and that means the same thing on a tabletop scan
and on a city block. On an analytic sphere with a 20k-point reference cloud:

| `--target_fit_ratio` | triangles | reduction |
|---|---|---|
| 0.25 | 1184 | 81% |
| 1.0 | 142 | 98% |
| 4.0 | 24 | 99.6% |

It works by binary search over the triangle count, decimating and re-measuring
at each probe (~12–16 probes resolves a million-triangle mesh). Two details
matter:

- **The mesh it returns is one whose error was measured**, not one the search
  assumed was fine. Quadric decimation is only *roughly* monotone in the
  triangle count and does not always land on the count it was asked for, so a
  binary search's final bracket is not by itself a guarantee. Among the probes
  that met the target, the smallest is returned — that part is about not
  handing back a needlessly large mesh rather than about correctness, since
  every candidate is feasible by measurement.
- **A target the input mesh already misses has no solution**, since decimating
  can only move the surface further from the cloud. The input comes back
  unchanged with `target_met: false` and the CLI warns, rather than handing
  back a smaller mesh that misses by more. That reads as either a poor
  extraction (check `--voxel_size` / `--poisson_depth`) or a target tighter
  than the reconstruction can be.

`--target_triangles` and `--target_fit_ratio` are two ways of asking the same
question and are rejected together. The result is
`mesh.obj` + `mesh.mtl` + `mesh_0.png` (albedo) + `mesh_normal.png`, with the
`.mtl` referencing both.

Add `--ao_map` for the third map of the standard set: **ambient occlusion**,
how much of the sky each point of the surface can actually see, so creases,
cavities and contact points darken. Neither the albedo nor the normal map
carries that cue, and without it a scanned asset reads flat under ambient
light. Each texel casts `--ao_samples` rays over the cosine-weighted
hemisphere about its normal (noise falls as `1/sqrt(n)`; 64 previews, a few
hundred is smooth), and the value stored is the fraction that escaped. It is
written as `mesh_ao.png` and noted in the `.mtl` as a comment — there is no
standard Wavefront key for an AO map, since it is an engine-side input rather
than a material property.

The CLI bakes AO as *self*-occlusion on the mesh that ships, unlike the normal
map's dense-vs-decimated bake. Casting against the dense mesh needs a ray cage
big enough to clear the decimation gap — most of a simplified surface sits
*inside* the mesh it came from (measured at 80% of texels on a decimated test
shape), and a ray starting under the occluder hits it immediately, baking a
uniformly dark map that looks like heavy occlusion and is pure artifact. That
cage then erases occlusion detail finer than itself, so it costs the very cues
it was meant to add. AO's real signal is large cavities and creases, which
survive decimation. `bake_ambient_occlusion(mesh, occluder_mesh=dense, cage=...)`
is available from the Python API if you want the dense bake anyway.

`--normal_map_space` selects `tangent` (the default -- what engines expect,
and valid under transforms) or `object` (simpler, immune to UV-seam tangent
artifacts, and a fine choice for a static scanned asset). Both are reported
with a `hit_fraction`: the share of texels whose ray actually reached the
dense mesh. A low value means the map is mostly flat and is doing nothing,
almost always because the ray cage is too small to span the gap between the
two meshes; the CLI warns when it drops below 50%.

Two ordering details matter, and the CLI handles both: decimation happens
*before* texturing, so the atlas is built on the mesh that ships; and the
normal map is baked *after* the albedo atlas so it reuses those UVs.
open3d's unwrapper is **not deterministic** -- unwrapping the same mesh twice
gives different layouts -- so a second unwrap would leave the normal map
addressed by different coordinates than the albedo, silently breaking the
asset. Baking onto a mesh that already has `triangle_uvs` always reuses them.

Note the resolution floor of an 8-bit normal map. Encoded as `0.5 + 0.5 * n`,
the whole range is spent on `[-1, 1]`, so the smallest representable deviation
is `2/255 ≈ 0.0078` — about 0.45° of tilt, and a hard ceiling on what the map
can carry however dense the source mesh is. On a surface whose low-poly
normals are already that accurate there is nothing to recover and the map
stores quantization noise.

Two ways out. Decimate far enough that the detail you are baking actually
exceeds the floor — or raise the floor with `--normal_map_bits 16`, which
drops it to `3.1e-5`. Measured on a sphere decimated 6240 → 3000 triangles
(a light decimation, exactly the regime where 8 bits stops resolving
anything), against the analytic normal:

| bits | quantization floor | mean normal error |
|---|---|---|
| 8 | 0.0078 | 0.0033 |
| 16 | 0.000031 | 0.0013 |

The 8-bit error is *entirely* quantization: uniform rounding over a step of
`2/255` costs about a quarter of a step per channel, ≈0.0034 in L2 over three
channels, which is what it measures. At 16 bits what remains is the bake's own
geometric error. The costs are a file twice the size and that not every
downstream tool reads 16-bit PNGs. `bake_normal_map` reports the
`quantization_floor` it used in its stats, so the comparison is available on a
real asset rather than only here.

(Writing that file does not go through imageio: Pillow, its default PNG
backend, cannot write 16-bit *RGB* PNGs at all — only 16-bit grayscale — so
the CLI writes them with OpenCV, which is already a dependency of the dataset
loader. OpenCV is BGR, and a normal map written without reversing the channels
loads fine and shades wrong, so `tests/test_extract_mesh_io.py` pins the round
trip.)

UV unwrapping requires a **manifold** mesh. TSDF and Poisson output has
already been through `remove_non_manifold_edges()`, so this normally holds;
if it doesn't, the CLIs warn and fall back to per-vertex colors rather than
failing at the end of a long run. (The underlying
`bake_texture_atlas` raises instead -- open3d's unwrapper *segfaults* on
non-manifold input rather than raising, so the check happens up front.)

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
`--mesh_texture_mode`, `--mesh_texture_size`, `--mesh_voxel_size`,
`--mesh_sdf_trunc` mirror `extract_mesh.py`'s options; `--mesh_texture_mode
atlas` writes `mesh_<step>.obj` rather than `mesh_<step>.ply`) --
Poisson reconstruction and the dense-MVS point cloud path still need the
standalone `examples/extract_mesh.py` script, since the trainer has no dense
point cloud of its own to reconstruct from. It writes mesh quality stats to
`results/garden_2dgs/stats/mesh_step<step>.json`, next to `eval()`'s own
`stats/val_step<step>.json` render-quality reports.

### One-command end-to-end pipeline

Steps 1-4 above (plus a baseline stats pass on the input SfM model, and
validating any AI-assisted priors) can also be run as a single command,
which wires each stage's output into the next exactly the way the manual
steps do -- the refined poses and dense point cloud both feed the trainer,
whose checkpoint feeds mesh extraction:

```bash
python examples/run_pipeline.py \
    --data_dir data/360_v2/garden --result_dir results/garden_pipeline
```

Add `--mono_depth_dir`/`--mask_dir` to run the AI-assisted stages too. Each
stage is invoked as a subprocess of its own standalone script (the same way
`dense_mvs.py` itself shells out to `colmap`), so `run_pipeline.py` stays
dependency-light and every stage keeps its own CLI as the source of truth
for its options; a stage that needs something the machine doesn't have (a
CUDA `colmap` build, a GPU) is recorded as `skipped` with the reason rather
than failing the run (pass `--strict` to fail instead). `--stages` selects a
subset (e.g. `--stages sfm_input bundle_adjust` to only refine poses and
report on it), and `--dry_run` prints the commands it would run without
running them. This writes `results/garden_pipeline/pipeline_report.json`:
per-stage status/timing/metrics plus every `stats/*.json` artifact found
under `--result_dir`/`--data_dir`, from
`gsplat.photogrammetry.pipeline.PipelineReport`. The report is written even
when a stage fails, since that is when it matters most.

#### The `priors` quality gate

Training is the expensive stage, and a bad prior directory is silent: a
segmenter run with an inverted keep/exclude convention, a depth model that
emitted constant maps, or a `--mask_dir` that simply matched no files all
look like a normal run until hours later. The `priors` stage judges the
`mask_coverage_stats`/`depth_prior_stats` numbers before that happens
(`gsplat.photogrammetry.pipeline.check_prior_quality`), printing each problem
it finds and recording them in the report as
`stages[priors].metrics.problems`:

```
[priors] WARNING: masks exclude 100.0% of the average frame (> 90.0%): little
photometric signal would be left to train on.
[priors] WARNING: 3/3 depth maps (100.0%) are constant or entirely
non-finite, and carry no gradient for the depth loss.
```

It flags a prior directory with no files at all, masks that exclude more than
`--max_excluded_fraction` (default 0.9) of the average frame or that exclude
nothing whatsoever, any mask that excludes its entire frame, a depth
directory more than `--max_degenerate_fraction` (default 0.5) of whose maps
are constant or entirely non-finite, and depth maps that are mostly
non-finite. Every check is on the directory as a whole -- one odd frame is
normal, a directory-wide pattern is a setup mistake.

By default this warns and the pipeline continues; `--strict` turns the
warnings into a stage failure, so the run stops before training instead of
after. Raise `--max_excluded_fraction` if your capture really is mostly
transient content.

### Automatic metrics & the consolidated report

Every stage above now reports automatic quality metrics
(`gsplat.photogrammetry.metrics`) -- bundle adjustment/dense MVS/mesh
extraction write a `stats/*.json` file next to their output, using the same
convention the trainer's `eval()` already uses for render quality
(PSNR/SSIM/LPIPS); `run_pipeline.py`'s `sfm_input`/`priors` stages compute
theirs directly (`reconstruction_stats`, `mask_coverage_stats`,
`depth_prior_stats`) since those don't otherwise write files. Running
`run_pipeline.py` already collects all of this into its
`pipeline_report.json` (above); to instead aggregate `stats/*.json` files
from a run you drove manually (steps 1-4), or to refresh a report after
extra runs of the individual scripts:

```bash
python examples/summarize_photogrammetry_stats.py \
    --result_dir results/garden_2dgs --data_dir data/360_v2/garden
```

This finds and aggregates whichever of `bundle_adjust_stats.json`,
`dense_stats.json`, `mesh_metrics.json`/`mesh_step*.json`, and
`val_step*.json` are present under `--result_dir`/`--data_dir`, prints a
summary table, and writes `results/garden_2dgs/stats_summary.json` (leaving
any `pipeline_report.json` from `run_pipeline.py` untouched) -- both share
`gsplat.photogrammetry.pipeline.collect_artifact_metrics`, mirroring
`examples/benchmarks/compression/summarize_stats.py`'s read-then-write
pattern.

#### Cross-stage metrics

Each stage's own metrics answer "what did this stage produce?". Both writers
also report a handful of numbers that only exist by *comparing* two stages --
"did it actually improve on, or agree with, what came before?", which is what
a photogrammetry run is really judged on and what no single stage can see.
An illustrative block (a run whose stages all completed):

```
CROSS-STAGE METRICS
-----------------------------------------------
reprojection_error_reduction         0.9646
points_retained_after_bundle_adjust  1
densification_ratio                  53.3
mesh_fit_over_point_spacing          0.9286
mesh_edge_over_point_spacing         1.333
```

| Metric | Compares | Reading it |
|---|---|---|
| `reprojection_error_reduction` | bundle adjustment before vs. after | Fraction of the input model's mean reprojection error removed. `0.1` = 10% better; **negative means it made the fit worse**. |
| `points_retained_after_bundle_adjust` | refined vs. input SfM model | Well under `1.0` means bundle adjustment discarded much of the sparse cloud. |
| `densification_ratio` | dense MVS vs. sparse SfM | How many times denser the MVS cloud is than the SfM points it started from. |
| `mesh_fit_over_point_spacing` | mesh vs. dense cloud | **The headline end-to-end number.** Mean cloud-to-mesh distance divided by the cloud's own mean k-NN spacing. At or below ~1 the mesh tracks the cloud to within its own sampling noise; well above 1 it genuinely misses geometry the cloud captured. |
| `mesh_edge_over_point_spacing` | mesh vs. dense cloud | Mean mesh edge length over that same spacing. Much below 1 means the mesh is tessellated finer than the evidence supports (`--voxel_size` too small); much above 1 means it discards detail the cloud has. |

The last two are why this matters: a raw cloud-to-mesh distance is in
arbitrary scene units and means nothing on its own, but measured against the
point cloud's own sample spacing it becomes a scale-free verdict on the mesh.

`run_pipeline.py` prints these after its stage table and stores them in
`pipeline_report.json` under `context.cross_stage_metrics`;
`summarize_photogrammetry_stats.py` prints the same block and writes
`stats_summary.json` as `{"artifact_metrics": ..., "cross_stage_metrics":
...}`. Both go through one shared derivation
(`gsplat.photogrammetry.pipeline.derive_cross_stage_metrics` /
`cross_stage_metrics_from_artifacts`), so a hand-run sequence is judged
identically to an orchestrated one. Any metric whose input stages didn't run
is **omitted rather than guessed** -- so the summarizer, which has no
`sfm_input` baseline to read, simply reports fewer of them.

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
    # Save a bare (H, W) array: depending on the transformers version,
    # `predicted_depth` can carry a leading batch axis, and an unsqueezed
    # (1, H, W) map is not loadable as a depth prior.
    depth = np.squeeze(depth.numpy()).astype(np.float32)
    assert depth.ndim == 2, f"expected one (H, W) depth map, got {depth.shape}"
    stem = os.path.splitext(fname)[0]
    np.save(os.path.join(out_dir, f"{stem}.npy"), depth)
```

Each `.npy` must be a single 2D `(H, W)` array (any resolution). Loading one
that isn't fails with a message naming the file, and
`run_pipeline.py`'s [`priors` gate](#the-priors-quality-gate) reports
`num_not_2d_maps` for the whole directory before training starts.

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
# merged["stats"] (gsplat.photogrammetry.metrics.track_stats, plus
# num_input_points/merge_radius) -- check multi_view_track_fraction before
# writing anything: a low value means most points didn't merge into a
# cross-view track, so bundle adjustment will have little to work with.
print(merged["stats"])

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
  `bake_texture_atlas(mesh, dataset, texture_size=...)` returns
  `(mesh, texture)` -- the mesh with `triangle_uvs`/`textures` set (write it
  with `open3d.io.write_triangle_mesh("mesh.obj", mesh)` to emit the `.obj`,
  `.mtl` and `.png` together) and the `uint8` atlas as a numpy array.
  `bake_mesh_texture(mesh, dataset, mode="vertex"|"atlas", ...)` is the
  dispatching entry point the CLIs use, returning `(mesh, texture_or_None)`
  and falling back to per-vertex colors if the mesh can't be unwrapped; pass
  `view_selection=True` (and a `stats_out={}` dict to receive the numbers) to
  route it through the single-view path.
- `gsplat.photogrammetry.bake_texture_atlas_view_selected(mesh, dataset,
  texture_size=..., mrf_smoothness=...)` returns `(mesh, texture, stats)`,
  texturing each face from one chosen view instead of a blend. `stats` carries
  the labelling (`mrf`), the texel accounting, and `atlas_sharpness` for both
  this atlas and the blended one it replaced. The pieces are usable on their
  own: `face_view_quality(mesh, dataset)` returns the `(F, V)` score matrix and
  `select_views_mrf(quality, adjacency, smoothness=...)` returns
  `(labels, stats)` with `NO_VIEW` for faces no view can texture.
- `gsplat.photogrammetry.atlas_sharpness(texture, covered_mask=None)` measures
  how much high-frequency detail a baked atlas carries. This is the metric that
  distinguishes view selection from blending -- pointwise error does not, and
  goes the other way.
- `gsplat.photogrammetry.level_seams(mesh, dataset, labels, smoothness=...)`
  returns the per-(vertex, label) colour corrections that close the seams,
  ready to interpolate across each face;
  `seam_discontinuity(mesh, texture, labels, triangle_uvs)` measures how
  visible those seams are in the atlas as shipped.
- `gsplat.photogrammetry.bake_ambient_occlusion(mesh, occluder_mesh=None,
  num_samples=..., cage=...)` returns `(mesh, ao_map, stats)`; `stats` reports
  `mean_ao`/`min_ao` and the `cage`/`max_distance` used. A `mean_ao` of
  essentially 1.0 means nothing occluded anything — correct for a convex
  shape, and otherwise a sign the occlusion distance is too small.
- `gsplat.photogrammetry.simplify_mesh(mesh, target_triangles=...)` returns a
  quadric-decimated copy, and
  `simplify_mesh_to_error(mesh, points, error_over_spacing=...)` returns
  `(mesh, stats)` — the smallest decimation whose *measured* cloud-to-mesh fit
  still meets the target, with every probe's (triangles, error) pair in
  `stats["probes"]`; `bake_normal_map(high_mesh, low_mesh, bits=8|16,
  texture_size=..., space="tangent"|"object")` returns
  `(low_mesh, normal_map, stats)`, baking the dense mesh's normals onto the
  decimated one's atlas (reusing its existing UVs when it has them).
- `gsplat.photogrammetry.metrics.point_to_mesh_distance(points, mesh, ...)` /
  `mesh_quality_stats(mesh)` / `point_cloud_stats(points, ...)` return plain
  dicts of quality stats for a mesh or point cloud, independent of how it was
  produced -- the functions the CLIs above write to `stats/*.json` with.
  `reconstruction_stats(colmap_dir)` measures a COLMAP model itself (image/
  point/observation counts, mean track length, mean reprojection error);
  `track_stats(tracks)` summarizes a
  `neural_sfm.merge_point_maps_to_tracks(...)["stats"]`-style track-length
  distribution; `mask_coverage_stats(mask_dir)` / `depth_prior_stats(dir)`
  sanity-check `--mask_dir`/`--mono_depth_dir` inputs before a long run.
- `gsplat.photogrammetry.pipeline.derive_cross_stage_metrics(report)` /
  `cross_stage_metrics_from_artifacts(collect_artifact_metrics(...))` return
  the derived comparisons above as a plain dict, and
  `format_cross_stage_metrics(...)` renders the block; `check_prior_quality(
  depth_stats=..., mask_stats=...)` returns the `priors` gate's problem list.
- `gsplat.photogrammetry.pipeline` -- `PipelineReport`/`run_stage`/
  `record_skipped` build a per-stage timing+metrics report like
  `examples/run_pipeline.py` does; `collect_artifact_metrics(result_dir,
  data_dir)` reads back every `stats/*.json` file the stages above wrote.
