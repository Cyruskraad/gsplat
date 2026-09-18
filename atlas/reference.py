# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""A Gaussian rasteriser written to be obviously correct, and slow.

Three jobs, none of which the CUDA kernel can do:

**It generates the synthetic capture.** Ground-truth data needs a renderer, and
until now the only one in this project is a tiled CUDA kernel that will not run
in CI and has never run here at all.

**It is `make smoke-cpu`.** The whole trainer, end to end, on a machine with no
GPU -- which is every machine this code has been written on so far.

**It is an independent oracle for `gsplat.rasterization`.** The GPU parity test
shows Path A and Path B agree *with each other*; if the kernel were not linear
in the per-primitive feature, both paths would be wrong together and the test
would still pass. This renderer is derived from the projection algebra rather
than from the kernel, so agreeing with it means something the parity test
cannot mean on its own.

Everything is the plain formulation: one global depth sort, every Gaussian
evaluated against every pixel, no tiles, no bounding boxes, no early
termination. That is roughly a thousand times slower than the kernel and it is
the point -- there is nowhere for a scheduling bug to hide. Cost is
``O(pixels x primitives)``, so this is for scenes of hundreds of primitives at
tens of thousands of pixels, and the chunking keeps its working set bounded
rather than making it fast.

Conventions follow ``gsplat.rasterization`` exactly, because the oracle is
worthless if it and the kernel mean different things:

* ``viewmats`` are **world-to-camera**, ``[4, 4]``, OpenCV convention: ``+x``
  right, ``+y`` down, ``+z`` forward into the scene.
* ``scales`` are **log-scale** and ``opacities`` are **logits**, as they are
  stored on :class:`~atlas.model.RelightSplats`; ``exp`` and ``sigmoid`` are
  applied here, in the same place ``render()`` applies them.
* Quaternions are ``[w, x, y, z]`` and are normalised before use.
* The 2-D covariance is dilated by ``dilation`` on the diagonal, which is
  gsplat's antialiasing convention and is a real part of the image, not a
  numerical nicety.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from .functional.transport import compositing_weights

__all__ = [
    "quaternion_to_rotation",
    "covariance_3d",
    "Projection",
    "project_gaussians",
    "render_reference",
    "look_at",
    "pinhole_intrinsics",
]


# --- geometry ---------------------------------------------------------------


def quaternion_to_rotation(quats: Tensor) -> Tensor:
    """``[N, 4]`` quaternions in ``[w, x, y, z]`` order to ``[N, 3, 3]``."""
    if quats.ndim != 2 or quats.shape[-1] != 4:
        raise ValueError(f"quats must be [N, 4], got {tuple(quats.shape)}")
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def covariance_3d(quats: Tensor, log_scales: Tensor) -> Tensor:
    """``Sigma = R S S^T R^T`` from ``[N, 4]`` quaternions and log scales."""
    if log_scales.ndim != 2 or log_scales.shape[-1] != 3:
        raise ValueError(f"log_scales must be [N, 3], got {tuple(log_scales.shape)}")
    rotation = quaternion_to_rotation(quats)
    scaled = rotation * torch.exp(log_scales).unsqueeze(-2)  # R @ diag(s)
    return scaled @ scaled.transpose(-1, -2)


def look_at(eye: Tensor, target: Tensor, up: Optional[Tensor] = None) -> Tensor:
    """A ``[4, 4]`` world-to-camera matrix, OpenCV convention.

    Provided here because every synthetic camera in the project needs one and
    writing it twice is how the two disagree about which way ``y`` points.
    """
    eye = eye.to(torch.float64).reshape(3)
    target = target.to(torch.float64).reshape(3)
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    up = up.to(torch.float64).reshape(3)

    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-12)
    if float((forward * up / up.norm()).sum().abs()) > 1 - 1e-6:
        raise ValueError("the up vector is parallel to the view direction")
    right = torch.linalg.cross(forward, up)
    right = right / right.norm().clamp_min(1e-12)
    down = torch.linalg.cross(forward, right)

    rotation = torch.stack([right, down, forward])  # world -> camera
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ eye
    return viewmat


def pinhole_intrinsics(width: int, height: int, fov_degrees: float = 45.0) -> Tensor:
    """A ``[3, 3]`` pinhole matrix with the principal point at the centre.

    The tangent is taken in Python rather than through ``torch.tensor(...)``,
    whose default dtype is float32: a focal length rounded to float32 puts a
    3e-9 relative error into every projected coordinate, which is invisible in
    an image and ruins this renderer's only real job, which is to be compared
    against hand-computed numbers.
    """
    focal = 0.5 * height / math.tan(math.radians(fov_degrees) / 2.0)
    return torch.tensor(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


# --- projection -------------------------------------------------------------


class Projection(NamedTuple):
    """Visible Gaussians, sorted front to back."""

    index: Tensor  #: [M] into the original primitives, in depth order
    mean2d: Tensor  #: [M, 2] pixel coordinates
    conic: Tensor  #: [M, 3] upper triangle (a, b, c) of the inverse 2-D covariance
    depth: Tensor  #: [M] camera-space z, ascending
    opacity: Tensor  #: [M] after sigmoid


def project_gaussians(
    means: Tensor,
    quats: Tensor,
    log_scales: Tensor,
    opacity_logits: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    *,
    near: float = 1e-2,
    dilation: float = 0.3,
    margin_sigmas: float = 3.0,
) -> Projection:
    """Project to screen space and cull, returning what survives in depth order.

    The EWA projection: the 2-D covariance is ``J W Sigma W^T J^T``, where ``W``
    is the camera rotation and ``J`` the Jacobian of the perspective divide at
    the primitive's own centre. It is a local linearisation, so it is wrong far
    from the centre and increasingly wrong towards the edge of a wide frame --
    which is the same approximation the CUDA kernel makes, deliberately, since
    an oracle that is *more* correct than the thing it checks reports
    differences that are not defects.
    """
    means = means.to(torch.float64)
    viewmat = viewmat.to(torch.float64)
    K = K.to(torch.float64)

    rotation = viewmat[:3, :3]
    translation = viewmat[:3, 3]
    camera = means @ rotation.T + translation  # [N, 3]
    depth = camera[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    safe_z = depth.clamp_min(near)
    mean2d = torch.stack(
        [fx * camera[:, 0] / safe_z + cx, fy * camera[:, 1] / safe_z + cy], dim=-1
    )

    zero = torch.zeros_like(safe_z)
    jacobian = torch.stack(
        [
            torch.stack([fx / safe_z, zero, -fx * camera[:, 0] / safe_z**2], dim=-1),
            torch.stack([zero, fy / safe_z, -fy * camera[:, 1] / safe_z**2], dim=-1),
        ],
        dim=-2,
    )  # [N, 2, 3]

    covariance = covariance_3d(quats, log_scales).to(torch.float64)
    world_to_pixel = jacobian @ rotation  # [N, 2, 3]
    cov2d = world_to_pixel @ covariance @ world_to_pixel.transpose(-1, -2)
    cov2d = cov2d + dilation * torch.eye(2, dtype=cov2d.dtype).expand_as(cov2d)

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    determinant = a * c - b * b
    radius = margin_sigmas * torch.sqrt(torch.maximum(a, c).clamp_min(0.0))

    visible = (
        (depth > near)
        & (determinant > 1e-12)
        & (mean2d[:, 0] > -radius)
        & (mean2d[:, 0] < width + radius)
        & (mean2d[:, 1] > -radius)
        & (mean2d[:, 1] < height + radius)
    )
    keep = visible.nonzero().flatten()
    order = keep[torch.argsort(depth[keep])]

    inverse = 1.0 / determinant[order]
    return Projection(
        index=order,
        mean2d=mean2d[order],
        conic=torch.stack(
            [c[order] * inverse, -b[order] * inverse, a[order] * inverse], dim=-1
        ),
        depth=depth[order],
        opacity=torch.sigmoid(opacity_logits.to(torch.float64))[order],
    )


# --- rendering --------------------------------------------------------------

#: gsplat discards a contribution below this, and so does this renderer: at
#: 8 bits an alpha under 1/255 cannot change the image, and evaluating it costs
#: the same as evaluating one that can.
MIN_ALPHA = 1.0 / 255.0

#: An alpha of exactly 1 makes every weight behind it zero, which is correct,
#: but it also makes the gradient of everything behind it zero, which ends
#: training for those primitives. gsplat clamps; so does this.
MAX_ALPHA = 0.999


def render_reference(
    means: Tensor,
    quats: Tensor,
    log_scales: Tensor,
    opacity_logits: Tensor,
    colors: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    *,
    background: Optional[Tensor] = None,
    rows_per_chunk: int = 0,
    **projection_kwargs,
) -> Tuple[Tensor, Tensor]:
    """Render one view. Returns ``(image [H, W, C], alpha [H, W])``.

    ``colors`` is ``[N, C]`` for any ``C``: three for radiance, ``3B`` for a
    packed transport splat. Nothing here knows or cares which, which is what
    makes this usable as the oracle for both render paths.
    """
    if colors.ndim != 2 or colors.shape[0] != means.shape[0]:
        raise ValueError(
            f"colors must be [{means.shape[0]}, C], got {tuple(colors.shape)}"
        )
    if width < 1 or height < 1:
        raise ValueError(f"the frame must be at least 1x1, got {width}x{height}")

    projection = project_gaussians(
        means,
        quats,
        log_scales,
        opacity_logits,
        viewmat,
        K,
        width,
        height,
        **projection_kwargs,
    )
    channels = colors.shape[-1]
    image = torch.zeros(height, width, channels, dtype=torch.float64)
    alpha_map = torch.zeros(height, width, dtype=torch.float64)

    if projection.index.numel() == 0:
        return _with_background(image, alpha_map, background, channels)

    features = colors.to(torch.float64)[projection.index]  # [M, C], depth order
    visible = projection.index.numel()
    if rows_per_chunk <= 0:
        # One row of pixels costs width * M doubles; keep a chunk near 32 MiB.
        rows_per_chunk = max(1, int((32 << 20) / max(width * visible * 8, 1)))

    columns = torch.arange(width, dtype=torch.float64)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        rows = torch.arange(start, stop, dtype=torch.float64)
        dx = columns.view(1, width, 1) - projection.mean2d[:, 0].view(1, 1, -1)
        dy = rows.view(-1, 1, 1) - projection.mean2d[:, 1].view(1, 1, -1)

        power = -0.5 * (
            projection.conic[:, 0] * dx * dx
            + 2.0 * projection.conic[:, 1] * dx * dy
            + projection.conic[:, 2] * dy * dy
        )
        alphas = projection.opacity * torch.exp(power.clamp(max=0.0))
        alphas = torch.where(alphas < MIN_ALPHA, torch.zeros_like(alphas), alphas)
        alphas = alphas.clamp(max=MAX_ALPHA)

        weights = compositing_weights(alphas)  # [rows, width, M]
        # `composite` would build a [rows, width, M, C] product. The features are
        # the same for every pixel, so the same sum is a matmul -- and the part
        # that is actually subtle, the exclusive cumulative product, is still
        # `compositing_weights` rather than a second copy of it here.
        image[start:stop] = weights @ features
        alpha_map[start:stop] = weights.sum(dim=-1)

    return _with_background(image, alpha_map, background, channels)


def _with_background(
    image: Tensor, alpha: Tensor, background: Optional[Tensor], channels: int
) -> Tuple[Tensor, Tensor]:
    if background is None:
        return image, alpha
    background = background.to(torch.float64).reshape(-1)
    if background.numel() == 1:
        background = background.expand(channels)
    if background.numel() != channels:
        raise ValueError(
            f"background must be a scalar or {channels} values, got "
            f"{background.numel()}"
        )
    return image + (1.0 - alpha).unsqueeze(-1) * background, alpha
