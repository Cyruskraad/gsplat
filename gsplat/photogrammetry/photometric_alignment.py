# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Refining camera poses so the photographs agree on the surface's colour.

Everything else this package does about misregistration is a *workaround*.
Robust fusion (:func:`~.texturing._bake_points_from_views`) discards views that
disagree; per-face view selection
(:func:`~.texturing.bake_texture_atlas_view_selected`) avoids averaging them;
seam levelling (:func:`~.texturing.level_seams`) repairs what view selection
then breaks. None of them fixes the cause.

The cause is that two views of a surface point are never registered to
sub-pixel accuracy after SfM. Bundle adjustment minimises *reprojection* error
over sparse feature tracks, which leaves a residual that is small in feature
terms and large in texture terms. That residual is the whole reason the
package's headline tradeoff exists: blending retains 59% of ground-truth
contrast but scores L1 0.171, while view selection retains 106% and scores
0.199. Blending attenuates detail; single-view sampling displaces it. Register
the views properly and the choice stops being a choice.

Method: Zhou & Koltun, *Color Map Optimization for 3D Reconstruction with
Consumer Depth Cameras*, SIGGRAPH 2014. Alternate between

1. baking the surface colour with the poses you currently have, and
2. moving each camera so its own image agrees with that colour,

coarse-to-fine over an image pyramid.

Requires the optional ``open3d`` dependency (``pip install gsplat[mesh]``) and
``torch``.

Why not open3d's implementation
-------------------------------
open3d is *already* a ``gsplat[mesh]`` dependency and ships this algorithm as
``open3d.pipelines.color_map.run_rigid_optimizer``, and "external tools stay
external" is a repo convention -- so reuse was tried first, and measured before
being rejected. On the analytic sphere with 10 views it made the poses **worse
in every configuration tested**:

===================  =============  ==============
Injected pose error  30 iterations  100 iterations
===================  =============  ==============
0'                   -> 72.1'       -> 84.9'
10'                  -> 117.1'      -> 153.5'
45'                  -> 280.8'      -> 368.8'
===================  =============  ==============

It fails the *do-no-harm* case outright: fed exact poses it walks 72' away from
them, which is not a tuning problem on the caller's side. The reason is
structural rather than a bug: open3d optimises **per-vertex colours** as its
surface proxy, so the objective can only see detail the mesh's vertex density
can carry. This package bakes into a *UV atlas* precisely because vertex
colours are far too coarse for that -- a 760-vertex sphere cannot represent the
texture whose misregistration is being measured, so the photometric gradient is
dominated by interpolation error. Its API is also built around RGBD scans with
shared intrinsics, where ``write_colmap_reconstruction`` and real captures are
per-image.

So this module keeps the algorithm and drops the proxy: the surface colour is
sampled at points dense enough to carry the texture, through the same
visibility and weighting hub every other bake in this package uses
(:func:`~.texturing._view_samples`).
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ._open3d import _require_open3d


class _PosedPyramidDataset:
    """A dataset view with substituted poses and optionally halved images.

    Both jobs belong together because they are the two things the alternation
    varies: the poses change every outer iteration, and the resolution changes
    every pyramid level. Wrapping rather than mutating means
    :func:`~.texturing._view_samples` -- and therefore the occlusion test, the
    weighting and the bilinear read -- works unchanged at every level, so
    "visible" keeps meaning exactly one thing across the whole package.

    Args:
        dataset: The wrapped ``Dataset``-like object.
        camtoworlds: ``(N, 4, 4)`` poses to report instead of the originals.
        levels: How many times to halve the images. 0 is the source resolution.
    """

    def __init__(self, dataset, camtoworlds: np.ndarray, levels: int = 0):
        import torch

        self._items = []
        for i in range(len(dataset)):
            data = dataset[i]
            image = np.asarray(data["image"].numpy(), dtype=np.float64)
            K = np.asarray(data["K"].numpy(), dtype=np.float64).copy()
            for _ in range(levels):
                # Crop an odd row/column before pooling rather than padding: a
                # padded edge invents a pixel, and the projection convention
                # (integer coordinates at pixel corners) then no longer lines
                # up with the intrinsics scaled below.
                height, width = image.shape[:2]
                image = image[: height - (height % 2), : width - (width % 2)]
                image = 0.25 * (
                    image[0::2, 0::2]
                    + image[0::2, 1::2]
                    + image[1::2, 0::2]
                    + image[1::2, 1::2]
                )
                # Corner-convention intrinsics scale linearly, principal point
                # included -- the same rule `Parser` uses for its `factor`.
                K[:2, :] *= 0.5
            self._items.append(
                {
                    "camtoworld": torch.from_numpy(
                        np.ascontiguousarray(camtoworlds[i], dtype=np.float64)
                    ),
                    "K": torch.from_numpy(np.ascontiguousarray(K)),
                    "image": torch.from_numpy(np.ascontiguousarray(image)),
                }
            )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def _torch_bilinear(images, view_idx, uv):
    """:func:`~.texturing._bilinear`, differentiable and batched over views.

    Deliberately a re-expression rather than a call: the numpy original cannot
    carry a gradient back to ``uv``, and the gradient with respect to the pixel
    coordinate *is* the photometric objective's entire signal. The conventions
    have to match exactly or the refinement optimises a slightly different
    image than the bake reads -- integer coordinates at pixel *corners* (so
    half a pixel comes off first) and clamping, never wrapping, at the border.
    ``tests/test_photometric_alignment.py`` pins the two against each other.
    """
    import torch

    height, width = images.shape[1], images.shape[2]
    x = (uv[:, 0] - 0.5).clamp(0.0, width - 1.0)
    y = (uv[:, 1] - 0.5).clamp(0.0, height - 1.0)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x0i = x0.long()
    y0i = y0.long()
    x1i = (x0i + 1).clamp(max=width - 1)
    y1i = (y0i + 1).clamp(max=height - 1)
    fx = (x - x0).unsqueeze(-1)
    fy = (y - y0).unsqueeze(-1)
    top = images[view_idx, y0i, x0i] * (1.0 - fx) + images[view_idx, y0i, x1i] * fx
    bottom = images[view_idx, y1i, x0i] * (1.0 - fx) + images[view_idx, y1i, x1i] * fx
    return top * (1.0 - fy) + bottom * fy


def _surface_samples(mesh, num_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Points and outward normals spread over the mesh, for the colour proxy.

    Sampled over the *surface* rather than taken at vertices so the proxy's
    density is set by what the texture needs, not by how the mesh happens to be
    tessellated -- which is the specific thing that makes open3d's per-vertex
    formulation unusable here.
    """
    o3d = _require_open3d()
    if len(mesh.triangles) == 0:
        raise ValueError(
            "Cannot photometrically align cameras against a mesh with no "
            "triangles: there is no surface for the views to agree about."
        )
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=int(num_points))
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(lengths, 1e-12, None)
    return points, normals


def _observations(mesh, dataset, points, normals, max_views):
    """Every visible (point, view) pair, through the package's one visibility hub.

    Returns ``(point_idx, view_idx, weight)``. Recomputed each alternation,
    because moving the cameras changes what each of them can see.
    """
    from .texturing import _view_samples

    o3d = _require_open3d()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    point_idx, view_idx, weights = [], [], []
    for chunk, _sampled, weight, view in _view_samples(
        scene, o3d, dataset, points, normals, max_views, 1 << 20
    ):
        point_idx.append(chunk)
        view_idx.append(np.full(chunk.shape, view, dtype=np.int64))
        weights.append(weight)
    if not point_idx:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.concatenate(point_idx),
        np.concatenate(view_idx),
        np.concatenate(weights),
    )


def _bake_proxy(mesh, dataset, points, normals, max_views, outlier_sigma):
    """The surface colour the cameras are asked to agree with, per point."""
    from .texturing import _bake_points_from_views

    color, weight = _bake_points_from_views(
        mesh,
        dataset,
        points,
        normals,
        max_views=max_views,
        outlier_sigma=outlier_sigma,
    )
    seen = weight > 0
    proxy = np.zeros_like(color)
    proxy[seen] = color[seen] / weight[seen, None]
    return proxy, seen


def refine_camera_poses_photometric(
    mesh,
    dataset,
    num_levels: int = 3,
    iterations: int = 60,
    alternations: int = 2,
    num_points: int = 20000,
    max_views: Optional[int] = None,
    lr: float = 3e-4,
    huber_delta: float = 0.1,
    anchor_image_idx: int = 0,
    anchor_reg: float = 1e-3,
    refine_translation: bool = True,
    outlier_sigma: Optional[float] = None,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Move each camera so its image agrees photometrically with the surface.

    Alternates Zhou & Koltun's two steps -- bake the surface colour with the
    current poses, then optimise each pose against that colour -- coarse to
    fine over an image pyramid.

    **What the pyramid is actually for, measured.** The received wisdom is that
    a photometric objective has a tiny basin of convergence and single-scale
    cannot recover 45'. On the analytic-sphere fixture that is simply false, and
    the numbers say why: the detail's wavelength is 5.2 px in the image and 45'
    displaces a projection by 1.70 px, comfortably inside the half-wavelength
    (2.6 px) within which the objective is unambiguous. At *equal work* --
    9 optimisation rounds either way -- single-scale reaches 23.6' and the
    3-level pyramid 24.75'. The alternation's re-baked target, not the pyramid,
    carries that case. (The comparison has to be equal-work: 3 levels x 3
    rounds against 1 level x 3 rounds says nothing, and asserting it left a
    mutation that disabled the pyramid entirely undetected.)

    The pyramid earns its place further out, where the displacement passes half
    the detail's wavelength and the objective starts aliasing. At 90'
    (3.39 px) equal-work single-scale lands at 183' where the pyramid lands at
    133'. Neither *recovers* there -- both end worse than they started -- so the
    honest summary is that this method's working range is a displacement below
    half the detail's wavelength, and the pyramid widens the margin rather than
    extending the range. It is kept on by default because it costs little and
    is strictly more robust in that band.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh`` -- the surface the views are
            registered against. It is not modified.
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3) and ``"image"``
            ((H, W, 3) in [0, 255]).
        num_levels: Pyramid levels. Level ``num_levels - 1`` is the coarsest and
            runs first. 1 disables the pyramid.
        iterations: Optimiser steps per alternation.
        alternations: Bake/optimise rounds per level. The bake is what carries
            the improvement from one round into the next.
        num_points: Surface points the colour proxy is sampled at. This is the
            resolution of the objective, and the reason this does not reuse
            open3d's per-vertex formulation.
        max_views: If given, only the first ``max_views`` images are used.
        lr: Adam learning rate on the pose deltas, decayed to a tenth over each
            alternation (the schedule ``bundle_adjustment`` already uses).
        huber_delta: Robust threshold on the colour residual, in [0, 1] units.
            Occluders, specularities and the silhouette all produce residuals
            that a squared loss would let dominate.
        anchor_image_idx: The camera held exactly fixed, via a zero gradient
            mask. Photometric alignment determines the cameras only *relative*
            to each other -- rotating every camera together and the surface
            with it costs nothing -- so without an anchor the gauge is free and
            the solve drifts. Fixing one is also why the absolute pose error
            floors at the anchor's own error while the *relative* registration,
            which is what texturing suffers from, keeps improving.
        anchor_reg: Weight on a small pull back toward the initial poses.
        refine_translation: Whether to solve for translation as well as
            rotation.
        outlier_sigma: Robust fusion for the colour proxy -- see
            :func:`~.texturing._bake_points_from_views`.
        seed: For the surface point sampling.
        device: Torch device. There is no GPU requirement here; the whole solve
            is small.

    Returns:
        ``(camtoworlds, stats)``. ``camtoworlds`` is ``(N, 4, 4)`` refined
        camera-to-world matrices. ``stats`` reports
        ``mean_photometric_residual_before``/``_after``,
        ``mean_pose_correction_arcmin``, ``num_observations``, and a
        ``levels`` list with the per-level residuals.

    Raises:
        ValueError: If ``mesh`` has no triangles, the dataset is empty, or
            ``anchor_image_idx`` is out of range.
    """
    import torch

    from .bundle_adjustment import _so3_exp

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    if num_views == 0:
        raise ValueError("Cannot align an empty dataset.")
    if not (0 <= anchor_image_idx < num_views):
        raise ValueError(
            f"anchor_image_idx {anchor_image_idx} is outside the "
            f"{num_views} views being aligned."
        )
    if num_levels < 1:
        raise ValueError(f"num_levels must be at least 1, got {num_levels}.")

    points, normals = _surface_samples(mesh, num_points, seed)

    initial = np.stack(
        [
            np.asarray(dataset[i]["camtoworld"].numpy(), dtype=np.float64)
            for i in range(num_views)
        ]
    )
    camtoworlds = initial.copy()

    torch_device = torch.device(device)
    points_t = torch.from_numpy(points).to(torch_device)

    def _residual_now(level: int) -> float:
        """Mean |view colour - baked colour| over every visible (point, view)."""
        view = _PosedPyramidDataset(dataset, camtoworlds, levels=level)
        proxy, seen = _bake_proxy(mesh, view, points, normals, max_views, outlier_sigma)
        p_idx, v_idx, weight = _observations(mesh, view, points, normals, max_views)
        if p_idx.size == 0:
            return float("nan")
        images, Ks, w2cs = _level_tensors(view, num_views, torch_device, torch)
        with torch.no_grad():
            uv = _project(points_t[p_idx], w2cs[v_idx], Ks[v_idx])
            sampled = _torch_bilinear(
                images, torch.from_numpy(v_idx).to(torch_device), uv
            )
            target = torch.from_numpy(proxy[p_idx]).to(torch_device)
            keep = torch.from_numpy(seen[p_idx]).to(torch_device)
            err = (sampled - target).abs().mean(dim=-1)
            w = torch.from_numpy(weight).to(torch_device) * keep
            return float((err * w).sum().item() / max(float(w.sum().item()), 1e-12))

    stats_levels = []
    residual_before = _residual_now(0)
    total_observations = 0

    for level in range(num_levels - 1, -1, -1):
        for _ in range(alternations):
            view = _PosedPyramidDataset(dataset, camtoworlds, levels=level)
            proxy, seen = _bake_proxy(
                mesh, view, points, normals, max_views, outlier_sigma
            )
            p_idx, v_idx, weight = _observations(mesh, view, points, normals, max_views)
            keep = seen[p_idx]
            p_idx, v_idx, weight = p_idx[keep], v_idx[keep], weight[keep]
            if p_idx.size == 0:
                continue
            total_observations = int(p_idx.size)

            images, Ks, w2cs = _level_tensors(view, num_views, torch_device, torch)
            target = torch.from_numpy(proxy[p_idx]).to(torch_device)
            weight_t = torch.from_numpy(weight).to(torch_device)
            weight_t = weight_t / weight_t.sum().clamp_min(1e-12)
            p_idx_t = torch.from_numpy(p_idx).to(torch_device)
            v_idx_t = torch.from_numpy(v_idx).to(torch_device)

            R0 = w2cs[:, :3, :3].clone()
            t0 = w2cs[:, :3, 3].clone()
            mask = torch.ones(num_views, 1, dtype=torch.float64, device=torch_device)
            mask[anchor_image_idx] = 0.0

            rot_delta = torch.nn.Parameter(
                torch.zeros(num_views, 3, dtype=torch.float64, device=torch_device)
            )
            trans_delta = torch.nn.Parameter(
                torch.zeros(num_views, 3, dtype=torch.float64, device=torch_device)
            )
            params = [rot_delta]
            if refine_translation:
                params.append(trans_delta)
            else:
                trans_delta.requires_grad_(False)

            optimizer = torch.optim.Adam(params, lr=lr)
            gamma = 0.1 ** (1.0 / max(iterations, 1))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

            for _step in range(iterations):
                optimizer.zero_grad()
                # The rotation carries the translation with it, so `rot_delta`
                # is a pure rotation about the camera's own optical centre.
                # Leaving `t0` alone (which is what a reprojection BA can get
                # away with, because its translation is always free to
                # compensate) instead swings the centre through |t0| * delta --
                # 0.046 world units here, a third of the texture's wavelength.
                # Measured: without this the solve makes the registration
                # *worse*, 62' -> 155'.
                delta_R = _so3_exp(rot_delta * mask)
                R = delta_R @ R0
                t = torch.einsum("nij,nj->ni", delta_R, t0) + trans_delta * mask
                Xc = (
                    torch.einsum("kij,kj->ki", R[v_idx_t], points_t[p_idx_t])
                    + t[v_idx_t]
                )
                uvw = torch.einsum("kij,kj->ki", Ks[v_idx_t], Xc)
                uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
                sampled = _torch_bilinear(images, v_idx_t, uv)
                err = (sampled - target).abs().mean(dim=-1)
                huber = torch.where(
                    err <= huber_delta,
                    0.5 * err**2,
                    huber_delta * (err - 0.5 * huber_delta),
                )
                loss = (huber * weight_t).sum()
                loss = loss + anchor_reg * (
                    ((rot_delta * mask) ** 2).sum(-1).mean()
                    + ((trans_delta * mask) ** 2).sum(-1).mean()
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                delta_R = _so3_exp(rot_delta * mask)
                R = delta_R @ R0
                t = torch.einsum("nij,nj->ni", delta_R, t0) + trans_delta * mask
                w2c = (
                    torch.eye(4, dtype=torch.float64, device=torch_device)
                    .unsqueeze(0)
                    .repeat(num_views, 1, 1)
                )
                w2c[:, :3, :3] = R
                w2c[:, :3, 3] = t
                camtoworlds = np.linalg.inv(w2c.cpu().numpy())

        stats_levels.append({"level": int(level), "residual": _residual_now(level)})

    residual_after = _residual_now(0)

    corrections = []
    for i in range(num_views):
        delta = initial[i, :3, :3].T @ camtoworlds[i, :3, :3]
        cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        corrections.append(np.degrees(np.arccos(cos)) * 60.0)

    stats = {
        "mean_photometric_residual_before": residual_before,
        "mean_photometric_residual_after": residual_after,
        "mean_pose_correction_arcmin": float(np.mean(corrections)),
        "max_pose_correction_arcmin": float(np.max(corrections)),
        "num_observations": total_observations,
        "num_views": int(num_views),
        "num_points": int(points.shape[0]),
        "num_levels": int(num_levels),
        "anchor_image_idx": int(anchor_image_idx),
        "levels": stats_levels,
    }
    return camtoworlds, stats


def _level_tensors(view, num_views, torch_device, torch):
    """Images (N, H, W, 3) in [0, 1], intrinsics and world-to-camera, as tensors."""
    images = torch.stack(
        [
            torch.as_tensor(view[i]["image"].numpy() / 255.0, dtype=torch.float64)
            for i in range(num_views)
        ]
    ).to(torch_device)
    Ks = torch.stack(
        [
            torch.as_tensor(view[i]["K"].numpy(), dtype=torch.float64)
            for i in range(num_views)
        ]
    ).to(torch_device)
    c2w = np.stack(
        [
            np.asarray(view[i]["camtoworld"].numpy(), dtype=np.float64)
            for i in range(num_views)
        ]
    )
    w2cs = torch.as_tensor(np.linalg.inv(c2w), dtype=torch.float64).to(torch_device)
    return images, Ks, w2cs


def _project(points, w2c, K):
    """Pixel coordinates of ``points`` under per-observation ``w2c`` and ``K``."""
    import torch

    Xc = torch.einsum("kij,kj->ki", w2c[:, :3, :3], points) + w2c[:, :3, 3]
    uvw = torch.einsum("kij,kj->ki", K, Xc)
    return uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
