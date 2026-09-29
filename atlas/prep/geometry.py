# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The geometry a relighting capture needs on top of what SfM returns.

SfM hands back poses in an arbitrary similarity frame. Relighting cannot use
that frame as-is, for one reason: the near-field falloff ``(d_ref / d)^2`` and
the flash's offset from the lens are *metric*. So the scene is moved so the
object sits at the origin and scaled so the median camera is at a stated
distance -- in the same unit the flash offset is given in. Everything else is
bookkeeping between conventions:

* COLMAP: ``cam_from_world`` in OpenCV axes (x right, y down, z forward),
  pixel centres at half-integers.
* gsplat / the loader: the same OpenCV world-to-camera matrices, the same
  pixel convention.
* The manifest: NeRF ``transform_matrix`` (camera-to-world, OpenGL axes),
  converted by :func:`atlas.data.synthetic.viewmat_to_nerf`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "focus_point",
    "normalise_scene",
    "flash_positions",
    "undistortion_grid",
    "remap",
    "knn_log_scales",
    "write_gaussian_ply",
]


def focus_point(centres: Tensor, forward: Tensor) -> Tensor:
    """The point closest, in least squares, to every camera's optical axis.

    For an orbit around an object this is the object, independent of where the
    sparse points happen to be dense. When the axes are close to parallel (a
    forward-facing capture) the system is ill-conditioned and the camera
    centroid pushed one median distance along the mean view direction is
    returned instead.
    """
    centres = centres.to(torch.float64)
    forward = torch.nn.functional.normalize(forward.to(torch.float64), dim=-1)
    eye = torch.eye(3, dtype=torch.float64)
    projectors = eye - forward.unsqueeze(-1) * forward.unsqueeze(-2)
    system = projectors.sum(dim=0)
    rhs = (projectors @ centres.unsqueeze(-1)).sum(dim=0).squeeze(-1)
    eigenvalues = torch.linalg.eigvalsh(system)
    if float(eigenvalues[0]) < 1e-3 * float(eigenvalues[-1]):
        spread = float(torch.linalg.norm(centres - centres.mean(0), dim=-1).median())
        return centres.mean(0) + forward.mean(0) * max(spread, 1.0)
    return torch.linalg.solve(system, rhs)


def normalise_scene(
    viewmats: Tensor, focus: Tensor, camera_distance: float
) -> Tuple[Tensor, float]:
    """Move ``focus`` to the origin and scale so the median camera sits at
    ``camera_distance``.

    Args:
        viewmats: ``[N, 4, 4]`` OpenCV world-to-camera.
        focus: ``[3]`` point to put at the origin.
        camera_distance: The target median camera-to-object distance.

    Returns:
        ``(viewmats, scale)``: new world-to-camera matrices, and the factor
        applied to world lengths (``x_new = scale * (x_old - focus)``).
    """
    viewmats = viewmats.to(torch.float64)
    rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
    centres = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    median = float(torch.linalg.norm(centres - focus, dim=-1).median())
    if median <= 0.0:
        raise ValueError("every camera sits at the focus point")
    scale = camera_distance / median
    out = viewmats.clone()
    # x_cam = R x_old + t, x_old = x_new / s + f  =>  s x_cam = R x_new + s (R f + t)
    out[:, :3, 3] = scale * ((rotation @ focus.to(torch.float64)) + translation)
    return out, scale


def flash_positions(viewmats: Tensor, offset: Sequence[float]) -> Tensor:
    """World positions of a flash rigidly mounted at ``offset`` in camera axes.

    ``offset`` is in OpenCV camera axes, in scene units: a hotshoe flash 12 cm
    above the lens is ``(0, -0.12, 0)``.
    """
    viewmats = viewmats.to(torch.float64)
    rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
    local = torch.tensor(list(offset), dtype=torch.float64).view(1, 3, 1)
    return (rotation.transpose(-1, -2) @ (local - translation.unsqueeze(-1))).squeeze(
        -1
    )


def undistortion_grid(source_camera, target_camera) -> Tensor:
    """``[H, W, 2]`` sampling grid, for ``grid_sample``, that undistorts.

    For every pixel of the pinhole ``target_camera`` it asks COLMAP where that
    ray lands in the distorted ``source_camera``. Using COLMAP's own camera
    models means every model it can estimate is undistorted correctly, with
    no re-implementation of the distortion polynomials.
    """
    import numpy as np

    width, height = int(target_camera.width), int(target_camera.height)
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    rays = np.asarray(target_camera.cam_from_img(pixels), dtype=np.float64)
    rays = np.concatenate([rays, np.ones_like(rays[:, :1])], axis=-1)
    source = np.asarray(
        source_camera.img_from_cam(rays, check_cheirality=False), dtype=np.float64
    )
    grid = torch.from_numpy(source).view(height, width, 2)
    size = torch.tensor(
        [float(source_camera.width), float(source_camera.height)], dtype=torch.float64
    )
    return grid / size * 2.0 - 1.0


def remap(image: Tensor, grid: Tensor) -> Tensor:
    """Resample ``[H, W, C]`` through ``grid``; outside pixels become zero."""
    chw = image.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    out = torch.nn.functional.grid_sample(
        chw,
        grid.unsqueeze(0).to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0).contiguous()


def knn_log_scales(points: Tensor, k: int = 3, chunk: int = 4096) -> Tensor:
    """``log`` of the mean distance to the ``k`` nearest neighbours, per point.

    The standard 3DGS initial scale. Chunked so a few hundred thousand SfM
    points do not need an ``N x N`` distance matrix at once.
    """
    points = points.to(torch.float32)
    count = points.shape[0]
    if count < 2:
        return torch.full((count,), math.log(1e-2))
    k = min(k, count - 1)
    means = []
    for start in range(0, count, chunk):
        block = torch.cdist(points[start : start + chunk], points)
        nearest = block.topk(k + 1, dim=-1, largest=False).values[:, 1:]
        means.append(nearest.mean(dim=-1))
    return torch.cat(means).clamp_min(1e-7).log()


def write_gaussian_ply(path: Path | str, means: Tensor, linear_rgb: Tensor) -> Path:
    """An initial 3DGS in the layout ``gsplat.export_splats`` writes.

    Isotropic Gaussians at the points, sized by neighbour distance, at opacity
    0.1, with the colour in the SH DC band -- the initialisation every 3DGS
    trainer uses, readable by :meth:`atlas.model.RelightSplats.from_ply`.
    """
    from ..ply import SH_C0

    means = means.to(torch.float32).cpu()
    count = means.shape[0]
    log_scales = knn_log_scales(means).unsqueeze(-1).expand(-1, 3)
    dc = (linear_rgb.to(torch.float32).cpu().clamp(0, 1) - 0.5) / SH_C0
    opacity = torch.full((count, 1), math.log(0.1 / 0.9))
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1.0
    normals = torch.zeros(count, 3)
    columns = torch.cat([means, normals, dc, opacity, log_scales, rotation], dim=-1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(columns.contiguous().numpy().astype("<f4").tobytes())
    return path
