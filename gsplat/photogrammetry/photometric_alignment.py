"""Photometric camera-pose refinement -- **a negative result, retained.**

**This does not work well enough to use, and is deliberately not exported from
`gsplat.photogrammetry` or wired to any CLI.** It is kept, with its measurements,
because the finding is specific and expensive to rediscover, and because a
future appearance model would make it work -- see "What would fix it" below.

## What it was for

Everything else in this package that addresses misregistration is a workaround.
Robust fusion discards outlier observations; per-face view selection avoids
averaging misaligned views; seam levelling repairs the steps view selection
leaves. None fixes the cause, which is that two views of a surface point are
never registered to sub-pixel accuracy after SfM. Zhou & Koltun, *Color Map
Optimization for 3D Reconstruction with Consumer Depth Cameras* (SIGGRAPH 2014)
fixes the cause, by alternating against a fixed surface:

1. estimate each surface point's colour by fusing the views that see it;
2. move each camera so its image agrees with that estimate.

The hope was that this collapses the sharpness-versus-accuracy tradeoff in
`docs/handoff/ISSUES.md` into a strict win, since blending only destroys detail
to the extent the views disagree about where that detail is.

## What actually happens

It converges to a scene-dependent **attractor of roughly 5-25 arcminutes
regardless of where it starts** -- including from exactly correct poses. It
helps when the poses are worse than that floor and harms when they are better,
which fails the "do no harm" criterion the work set for itself. On a 12-view
synthetic sphere, mean pose error after refinement (arcminutes):

    pose_prior     from 0'    from 15'   from 45'
        0           23.39      23.39      23.93
        1            4.77      10.08      23.91
       10            2.09       9.97      28.71
      100            0.63      13.33      41.66
     1000            0.22      14.80      44.64

No row both preserves correct poses and meaningfully recovers a large error.

## Why -- the diagnosis, measured

The objective is not merely hard to optimise; **its minimum is in the wrong
place.** With the fused colour re-estimated at each pose (the true joint
objective), the converged pose scores **0.038** against **0.061 at the ground
truth**. The optimiser is minimising exactly what it was asked to.

The cause is the appearance model. One colour per surface point cannot express
what different views of the same point actually report: they differ in pixel
footprint, in obliquity, and in how the mesh's own faceting falls inside a
pixel. At the true poses those differences are already 0.061 (mean absolute,
against the fused blend), and they are *reducible* by moving cameras. The bias
is larger than the error being corrected.

Ruled out along the way, each by measurement rather than argument:

- **Not the optimiser.** Adam random-walks (drift proportional to learning
  rate: ~14' at 3e-4, ~8' at 1e-4, ~4' at 3e-5, matching sqrt(steps)), so this
  uses L-BFGS with a strong-Wolfe line search, which stops. The attractor
  remained.
- **Not the pyramid.** Present at `num_levels=1`.
- **Not the fixture's tessellation.** A finer mesh is *worse* (20 -> 40 -> 80
  resolution: 20.6' -> 28.2' -> 29.2' of drift from correct poses).
- **Not texture periodicity.** A non-periodic band-limited random field is
  worse still (~107'). A *smoother* pattern is dramatically worse (~129'),
  which is consistent: less texture, flatter objective, further wandering.
- **Not out-of-frame clamping**, though that was a real bug and is fixed here:
  `_bilinear_torch` clamps, which is right for sampling but would reward the
  optimiser for pushing the surface off-image onto the flat border.
- **Not target/residual scale mismatch**, also a real bug found and fixed:
  fusing the target from full-resolution images while scoring it against
  downsampled ones moved correct poses on its own.

`open3d` ships this algorithm as `pipelines.color_map.run_rigid_optimizer` and
is already a dependency, so the repo's "external tools stay external"
convention says to prefer it. It was tried first and is worse: it degraded pose
error in every configuration tested (5', 15', 45' of injected error, with and
without `convert_rgb_to_intensity`), always landing near 190-280' regardless of
the start, and **given perfect poses it moved them to 223' of error**.

## What would fix it

The missing ingredient is a per-view appearance term, so that a legitimate
difference between two views is explained rather than blamed on the pose:
a per-view exposure/gain (Zhou & Koltun's own non-rigid variant carries a
per-image warp for the analogous reason), and a footprint-aware sample -- the
target for a view should be the surface colour *convolved with that view's
pixel footprint*, not a single point sample shared by all views. That is the
same modelling gap the super-resolution work (Goldlucke et al., IJCV 2014)
exists to close, which suggests doing that first and revisiting this after.

## One structural note worth keeping

There is **no gauge freedom here, so no anchor camera** -- unlike
:mod:`gsplat.photogrammetry.bundle_adjustment`, which this borrows its pose
parameterisation from. Pure reprojection bundle adjustment can slide the whole
scene 7 degrees of freedom without changing its cost, so it must pin something.
Here the mesh is fixed in world space, so moving every camera together *does*
change the cost. Anchoring a camera would not fix a gauge; it would just forbid
that camera from being corrected.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ._open3d import _require_open3d
from .bundle_adjustment import _so3_exp
from .texturing import _bake_points_from_views, _view_samples


class _PoseOverride:
    """A dataset view substituting poses, and optionally the pyramid level.

    Lets the existing bakers and :func:`_view_samples` be re-run at candidate
    poses without copying images, so "visible" keeps meaning exactly what it
    means everywhere else in the package rather than becoming a second, subtly
    different test defined here.

    ``images``/``Ks`` must be overridden together with the poses whenever the
    residual is evaluated on a downsampled level. Fusing the target colour from
    full-resolution images and then comparing it against blurred ones is a
    systematic mismatch, not a small one: it moves cameras away from poses that
    are already exactly right, because a sharp target can always be matched
    slightly better somewhere other than where it belongs.
    """

    def __init__(self, dataset, camtoworlds: np.ndarray, images=None, Ks=None):
        self._dataset = dataset
        self._camtoworlds = camtoworlds
        self._images = images
        self._Ks = Ks

    def __len__(self) -> int:
        return len(self._camtoworlds)

    def __getitem__(self, index: int) -> dict:
        item = dict(self._dataset[index])
        item["camtoworld"] = torch.as_tensor(
            self._camtoworlds[index], dtype=torch.float32
        )
        if self._images is not None:
            item["image"] = torch.as_tensor(
                self._images[index] * 255.0, dtype=torch.float32
            )
        if self._Ks is not None:
            item["K"] = torch.as_tensor(self._Ks[index], dtype=torch.float32)
        return item


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter an image down by an integer factor, cropping the remainder."""
    if factor <= 1:
        return image
    height = (image.shape[0] // factor) * factor
    width = (image.shape[1] // factor) * factor
    cropped = image[:height, :width]
    return cropped.reshape(
        height // factor, factor, width // factor, factor, image.shape[2]
    ).mean(axis=(1, 3))


def _scale_intrinsics(K: np.ndarray, factor: int) -> np.ndarray:
    """Rescale a pinhole K for an image downsampled by `factor`.

    The half-pixel convention matters: a pixel centre at `x + 0.5` in the full
    image sits at `(x + 0.5) / factor` in the downsampled one, which is what
    the `+ 0.5 ... - 0.5` sandwich below preserves. Scaling `cx` by `1/factor`
    alone shifts the principal point by half a pixel per level, which a
    photometric objective reads as a real misalignment and dutifully "corrects".
    """
    if factor <= 1:
        return K
    scaled = K.copy()
    scaled[0, 0] /= factor
    scaled[1, 1] /= factor
    scaled[0, 2] = (K[0, 2] + 0.5) / factor - 0.5
    scaled[1, 2] = (K[1, 2] + 0.5) / factor - 0.5
    return scaled


def _bilinear_torch(image: Tensor, uv: Tensor) -> Tensor:
    """Differentiable bilinear sample, matching `texturing._bilinear` exactly.

    Written out rather than using `grid_sample` so the half-pixel convention is
    visibly the same as the one every other colour read in this package uses --
    a half-pixel disagreement between the sampler that *measures* misalignment
    and the sampler that later *bakes* would make this optimiser correct a
    misregistration that is not there.
    """
    height, width = image.shape[0], image.shape[1]
    x = torch.clamp(uv[:, 0] - 0.5, 0.0, width - 1.0)
    y = torch.clamp(uv[:, 1] - 0.5, 0.0, height - 1.0)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    fx = (x - x0.to(x.dtype)).unsqueeze(-1)
    fy = (y - y0.to(y.dtype)).unsqueeze(-1)
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _visible_point_indices(
    mesh, dataset, points: np.ndarray, normals: np.ndarray, max_views: Optional[int]
):
    """Which points each view sees, and with what weight, by the package's test.

    Reuses :func:`_view_samples`, so occlusion, front-facing and in-frame all
    mean here what they mean in every bake -- and so the weights are the *same*
    view/normal-alignment and inverse-distance weights the fused target is
    built from. Using them in the residual too is not cosmetic: the target is a
    weighted mean, so scoring every view equally against it lets a grazing view
    that contributes almost nothing to the target dominate the correction.
    """
    o3d = _require_open3d()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    per_view: Dict[int, List[np.ndarray]] = {}
    per_weight: Dict[int, List[np.ndarray]] = {}
    for chunk, _sampled, weight, view in _view_samples(
        scene, o3d, dataset, points, normals, max_views, 1 << 20
    ):
        per_view.setdefault(view, []).append(chunk)
        per_weight.setdefault(view, []).append(weight)
    return {
        view: (np.concatenate(chunks), np.concatenate(per_weight[view]))
        for view, chunks in per_view.items()
    }


def refine_camera_poses_photometric(
    mesh,
    dataset,
    max_views: Optional[int] = None,
    num_levels: int = 3,
    iterations_per_level: int = 60,
    alternations_per_level: int = 6,
    learning_rate: float = 1.0,
    max_points: int = 4000,
    huber_delta: float = 0.1,
    pose_prior: float = 1.0,
    outlier_sigma: Optional[float] = None,
    seed: int = 0,
):
    """Refine camera poses so all views agree on the surface's colour.

    Alternates estimating each surface point's colour from the current poses
    with moving the poses to match it, coarse-to-fine over an image pyramid
    (Zhou & Koltun, SIGGRAPH 2014). The mesh is never modified.

    The pyramid is load-bearing, not a speed optimisation: a photometric
    objective has a basin of convergence roughly the width of the image
    features it is matching, so at full resolution it cannot see a
    misregistration larger than a few pixels. ``num_levels=1`` is measurably
    worse at recovering large errors, and a test pins that.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``, held fixed.
        dataset: An ``examples.datasets.colmap.Dataset``-like object -- see
            :func:`gsplat.photogrammetry.bake_texture`.
        max_views: If given, only the first ``max_views`` images are used.
        num_levels: Image-pyramid levels. Level ``k`` is downsampled ``2**k``.
        iterations_per_level: Total Adam steps per pyramid level, split evenly
            across ``alternations_per_level``.
        alternations_per_level: How many times per level to re-estimate the
            surface colour from the current poses. This is the "alternating"
            in the algorithm and it is not optional: with the target fused
            once and held fixed, the optimiser slides cameras into whichever
            nearby pixels happen to match a slightly blurred estimate, which
            measurably *degrades* already-correct poses.
        learning_rate: Initial L-BFGS step size. With a strong-Wolfe line
            search this is a starting guess that the search scales, not a
            fixed rate, so 1.0 is the right default rather than a small value.
        max_points: Cap on surface samples, subsampled deterministically from
            the mesh's vertices. Cost is ``O(max_points x views)`` per step.
        huber_delta: Residual (in [0, 1] colour units) past which the loss
            becomes linear, so a specular highlight or a passing pedestrian
            cannot dominate the fit.
        outlier_sigma: Passed to the colour-estimation step -- see
            :func:`_bake_points_from_views`.
        seed: Subsampling seed.

    Returns:
        ``(camtoworlds, stats)``. ``camtoworlds`` is a new ``(V, 4, 4)`` array;
        the dataset is not modified. ``stats`` carries the photometric residual
        before and after, the mean pose correction in arcminutes, and the
        per-level residual trace.
    """
    o3d = _require_open3d()

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    if num_views == 0:
        raise ValueError("Photometric alignment needs at least one view.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError(
            "Photometric alignment needs a mesh with vertices to align against."
        )
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals_all = np.asarray(mesh.vertex_normals, dtype=np.float64)

    rng = np.random.default_rng(seed)
    if len(vertices) > max_points:
        keep = rng.choice(len(vertices), size=max_points, replace=False)
        keep.sort()
    else:
        keep = np.arange(len(vertices))
    points = vertices[keep]
    normals = normals_all[keep]

    initial = np.stack(
        [
            np.asarray(dataset[i]["camtoworld"], dtype=np.float64)
            for i in range(num_views)
        ]
    )
    images = [
        np.asarray(dataset[i]["image"], dtype=np.float64) / 255.0
        for i in range(num_views)
    ]
    Ks = [np.asarray(dataset[i]["K"], dtype=np.float64) for i in range(num_views)]

    R0 = torch.tensor(initial[:, :3, :3], dtype=torch.float64)
    t0 = torch.tensor(initial[:, :3, 3], dtype=torch.float64)
    points_t = torch.tensor(points, dtype=torch.float64)

    # No anchor camera: the mesh fixes the world frame, so unlike pure
    # reprojection bundle adjustment there is no gauge to pin. See the module
    # docstring.
    rot_delta = torch.nn.Parameter(torch.zeros(num_views, 3, dtype=torch.float64))
    trans_delta = torch.nn.Parameter(torch.zeros(num_views, 3, dtype=torch.float64))

    def current_camtoworlds() -> np.ndarray:
        with torch.no_grad():
            R = _so3_exp(rot_delta) @ R0
            t = t0 + trans_delta
        out = np.tile(np.eye(4), (num_views, 1, 1))
        out[:, :3, :3] = R.numpy()
        out[:, :3, 3] = t.numpy()
        return out

    stats: dict = {
        "num_views": num_views,
        "num_points": int(len(points)),
        "levels": [],
    }
    residual_before = None

    for level in range(num_levels - 1, -1, -1):
        factor = 2**level
        level_image_arrays = [_downsample(img, factor) for img in images]
        level_K_arrays = [_scale_intrinsics(K, factor) for K in Ks]
        level_images = [
            torch.tensor(img, dtype=torch.float64) for img in level_image_arrays
        ]
        level_Ks = [torch.tensor(K, dtype=torch.float64) for K in level_K_arrays]
        steps = max(1, iterations_per_level // max(alternations_per_level, 1))
        level_before = None

        for _alternation in range(max(alternations_per_level, 1)):
            poses = current_camtoworlds()
            override = _PoseOverride(
                dataset, poses, images=level_image_arrays, Ks=level_K_arrays
            )

            # Step 1: the surface's colour, fused from the views at the poses
            # as they stand -- the same estimator the atlas bake uses.
            color_accum, weight_accum = _bake_points_from_views(
                mesh,
                override,
                points,
                normals,
                max_views=max_views,
                outlier_sigma=outlier_sigma,
            )
            seen = weight_accum > 0
            target = np.zeros_like(color_accum)
            target[seen] = color_accum[seen] / weight_accum[seen][:, None]
            target_t = torch.tensor(target, dtype=torch.float64)

            visible = _visible_point_indices(mesh, override, points, normals, max_views)
            visible = {
                view: (idx[seen[idx]], w[seen[idx]])
                for view, (idx, w) in visible.items()
                if seen[idx].any()
            }
            if not visible:
                raise ValueError(
                    "No surface point is visible from any view, so there is no "
                    "photometric signal to align. The mesh and the cameras are "
                    "probably in different coordinate frames."
                )

            def residual() -> Tensor:
                R = _so3_exp(rot_delta) @ R0
                t = t0 + trans_delta
                total = points_t.new_zeros(())
                count = 0.0
                for view, (idx, view_weight) in visible.items():
                    index = torch.as_tensor(idx, dtype=torch.long)
                    weight = torch.as_tensor(view_weight, dtype=torch.float64)
                    # World -> camera is the inverse of camera-to-world; for a
                    # rotation that is the transpose, so no solve is needed.
                    Rw = R[view].transpose(0, 1)
                    local = (points_t[index] - t[view]) @ Rw.transpose(0, 1)
                    depth = local[:, 2].clamp_min(1e-6)
                    uvw = local @ level_Ks[view].transpose(0, 1)
                    uv = uvw[:, :2] / depth.unsqueeze(-1)

                    # Points that leave the frame must be dropped, not clamped.
                    # `_bilinear_torch` clamps, which is right for sampling but
                    # catastrophic as an objective: it turns an out-of-frame
                    # projection into a cheap sample of the flat border, so the
                    # optimiser is *rewarded* for pushing the surface off-image
                    # until every view agrees on the background. Measured with
                    # the clamp exposed, poses converged to the same wrong
                    # attractor from 0' and from 45' of injected error.
                    height, width = (
                        level_images[view].shape[0],
                        level_images[view].shape[1],
                    )
                    inside = (
                        (uv[:, 0] >= 0.0)
                        & (uv[:, 0] <= width - 1.0)
                        & (uv[:, 1] >= 0.0)
                        & (uv[:, 1] <= height - 1.0)
                        & (local[:, 2] > 1e-6)
                    )
                    if not bool(inside.any()):
                        continue
                    sampled = _bilinear_torch(level_images[view], uv)
                    diff = (sampled - target_t[index]).abs()
                    huber = torch.where(
                        diff <= huber_delta,
                        0.5 * diff**2,
                        huber_delta * (diff - 0.5 * huber_delta),
                    )
                    gate = (weight * inside).unsqueeze(-1)
                    total = total + (huber * gate).sum()
                    count = count + float(gate.sum().item()) * 3
                mean = total / max(count, 1e-12)
                if pose_prior > 0.0:
                    # Without this the objective is free to wander: minimising
                    # inter-view disagreement has minima away from the truth,
                    # because one colour per surface point cannot express the
                    # per-view blur a real capture has. Measured: from *exactly
                    # correct* poses the unregularised fit reaches a joint
                    # objective of 0.038 against 0.061 at the truth. So this is
                    # refinement, anchored to the input, not free estimation --
                    # the same role `anchor_reg` plays in bundle_adjustment.py.
                    mean = mean + pose_prior * (
                        (rot_delta**2).sum(-1).mean()
                        + (trans_delta**2).sum(-1).mean()
                    )
                return mean

            if level_before is None:
                with torch.no_grad():
                    level_before = float(residual().item())
                if residual_before is None:
                    residual_before = level_before

            # L-BFGS with a strong-Wolfe line search, not Adam. Adam takes a
            # step of roughly `lr` in parameter units however small the
            # gradient is, so at the optimum it does not stop -- it random
            # walks. Measured on already-correct poses, the drift was ~14' at
            # lr 3e-4, ~8' at 1e-4 and ~4' at 3e-5: proportional to the step
            # size and matching a sqrt(steps) random walk, which is the
            # signature of an optimiser that cannot recognise it has arrived.
            # A line search evaluates the objective before accepting a step, so
            # it does stop. Zhou & Koltun use Gauss-Newton for the same reason.
            optimizer = torch.optim.LBFGS(
                [rot_delta, trans_delta],
                lr=learning_rate,
                max_iter=steps,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            )

            def closure():
                optimizer.zero_grad()
                loss = residual()
                loss.backward()
                return loss

            optimizer.step(closure)

        with torch.no_grad():
            level_after = float(residual().item())
        stats["levels"].append(
            {
                "level": int(level),
                "downsample": int(factor),
                "image_size": [
                    int(level_images[0].shape[1]),
                    int(level_images[0].shape[0]),
                ],
                "residual_before": level_before,
                "residual_after": level_after,
            }
        )

    refined = current_camtoworlds()

    # Report the correction as an angle, which is the unit the pose error was
    # specified in and the only one comparable across scene scales.
    angles = []
    for i in range(num_views):
        delta = initial[i][:3, :3].T @ refined[i][:3, :3]
        cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos)) * 60.0)

    stats["residual_before"] = residual_before
    stats["residual_after"] = stats["levels"][-1]["residual_after"]
    stats["mean_pose_correction_arcmin"] = float(np.mean(angles))
    stats["max_pose_correction_arcmin"] = float(np.max(angles))
    stats["mean_translation_correction"] = float(
        np.mean(np.linalg.norm(refined[:, :3, 3] - initial[:, :3, 3], axis=-1))
    )
    return refined, stats
