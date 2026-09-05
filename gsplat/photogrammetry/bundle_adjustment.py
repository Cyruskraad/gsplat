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
"""Torch-native bundle adjustment over a COLMAP reconstruction.

Refines per-image camera poses and 3D point positions by minimizing a robust
(Huber) reprojection error over the SfM point tracks, using plain gradient-based
optimization (Adam) rather than a sparse Levenberg-Marquardt solver. This keeps
the implementation self-contained (no new heavy solver dependency) and lets the
refinement compose naturally with the rest of gsplat's torch-based stack.

The public entry point, :func:`refine_reconstruction`, reads and writes a real
COLMAP model via ``pycolmap`` so the result is a drop-in replacement for the
input: point ``examples.datasets.colmap.Parser`` at the output directory and
nothing else needs to change. The underlying optimization loop,
:func:`_optimize`, operates purely on tensors and is unit-tested independently
of any COLMAP I/O.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def _skew(v: Tensor) -> Tensor:
    """Build skew-symmetric matrices from 3-vectors. v: (..., 3) -> (..., 3, 3)."""
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _so3_exp(v: Tensor) -> Tensor:
    """so(3) tangent vector -> SO(3) rotation matrix via the matrix exponential.

    v: (..., 3) -> (..., 3, 3)
    """
    return torch.matrix_exp(_skew(v))


def _optimize(
    w2c_init: Tensor,  # (M, 4, 4) initial world-to-camera transforms
    Ks: Tensor,  # (M, 3, 3) per-image pinhole intrinsics
    points_init: Tensor,  # (P, 3) initial 3D point positions
    obs_image_idx: Tensor,  # (K,) long, index into [0, M)
    obs_point_idx: Tensor,  # (K,) long, index into [0, P)
    obs_xy: Tensor,  # (K, 2) observed pixel coordinates
    anchor_image_idx: int,
    num_iters: int = 2000,
    lr: float = 1e-3,
    huber_delta: float = 1.0,
    anchor_reg: float = 1e-4,
    refine_points: bool = True,
) -> Dict[str, object]:
    """Run the reprojection-error bundle adjustment optimization.

    Poses are parameterized as the fixed initial world-to-camera transform plus
    a learnable delta (translation + so(3) tangent rotation), matching the
    delta-pose convention already used by ``examples.utils.CameraOptModule``
    elsewhere in this codebase. The ``anchor_image_idx`` image's delta is kept
    at exactly zero (via a zero gradient mask) to fix the 7-DOF gauge freedom
    (rotation + translation + scale) of pure reprojection bundle adjustment.

    Returns a dict with the refined ``"w2c"`` (M, 4, 4), refined ``"points"``
    (P, 3), and the mean pixel reprojection error before/after optimization.
    """
    device = w2c_init.device
    M = w2c_init.shape[0]

    R0 = w2c_init[:, :3, :3].clone()  # (M, 3, 3), fixed
    t0 = w2c_init[:, :3, 3].clone()  # (M, 3), fixed

    mask = torch.ones(M, 1, device=device)
    mask[anchor_image_idx] = 0.0

    trans_delta = torch.nn.Parameter(torch.zeros(M, 3, device=device))
    rot_delta = torch.nn.Parameter(torch.zeros(M, 3, device=device))
    points = torch.nn.Parameter(points_init.clone())

    params: List[torch.nn.Parameter] = [trans_delta, rot_delta]
    if refine_points:
        params.append(points)
    else:
        points.requires_grad_(False)

    optimizer = torch.optim.Adam(params, lr=lr)
    gamma = 0.1 ** (1.0 / max(num_iters, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _current_pose() -> Dict[str, Tensor]:
        trans_eff = trans_delta * mask  # (M, 3)
        rot_eff = rot_delta * mask  # (M, 3)
        R = _so3_exp(rot_eff) @ R0  # (M, 3, 3)
        t = t0 + trans_eff  # (M, 3)
        return {"R": R, "t": t}

    def _reprojection_error(points_cur: Tensor, pose: Dict[str, Tensor]) -> Tensor:
        """Per-observation pixel reprojection error (K,), invalid ones set to nan."""
        Xw = points_cur[obs_point_idx]  # (K, 3)
        R_k = pose["R"][obs_image_idx]  # (K, 3, 3)
        t_k = pose["t"][obs_image_idx]  # (K, 3)
        Xc = torch.einsum("kij,kj->ki", R_k, Xw) + t_k  # (K, 3)
        K_k = Ks[obs_image_idx]  # (K, 3, 3)
        uvw = torch.einsum("kij,kj->ki", K_k, Xc)  # (K, 3)
        valid = uvw[:, 2] > 1e-6
        uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
        err = (uv - obs_xy).norm(dim=-1)
        return torch.where(valid, err, torch.full_like(err, float("nan")))

    with torch.no_grad():
        err_before = _reprojection_error(points, _current_pose())
        mean_err_before = float(torch.nanmean(err_before).item())

    for _ in range(num_iters):
        optimizer.zero_grad()
        pose = _current_pose()
        err = _reprojection_error(points, pose)
        valid = ~torch.isnan(err)
        err = torch.nan_to_num(err, nan=0.0)
        huber = torch.where(
            err <= huber_delta,
            0.5 * err**2,
            huber_delta * (err - 0.5 * huber_delta),
        )
        reproj_loss = (huber * valid).sum() / valid.sum().clamp_min(1)

        reg_loss = anchor_reg * (
            ((trans_delta * mask) ** 2).sum(-1).mean()
            + ((rot_delta * mask) ** 2).sum(-1).mean()
        )
        if refine_points:
            reg_loss = (
                reg_loss + anchor_reg * ((points - points_init) ** 2).sum(-1).mean()
            )

        loss = reproj_loss + reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        pose = _current_pose()
        err_after = _reprojection_error(points, pose)
        mean_err_after = float(torch.nanmean(err_after).item())

        w2c = torch.eye(4, device=device).unsqueeze(0).repeat(M, 1, 1)
        w2c[:, :3, :3] = pose["R"]
        w2c[:, :3, 3] = pose["t"]

    return {
        "w2c": w2c.detach(),
        "points": points.detach(),
        "mean_reprojection_error_before": mean_err_before,
        "mean_reprojection_error_after": mean_err_after,
    }


def refine_reconstruction(
    colmap_dir: str,
    output_dir: str,
    num_iters: int = 2000,
    lr: float = 1e-3,
    huber_delta: float = 1.0,
    anchor_reg: float = 1e-4,
    refine_points: bool = True,
    device: str = "cuda",
) -> Dict[str, float]:
    """Refine a COLMAP reconstruction's poses (and, optionally, 3D points).

    Loads ``colmap_dir`` via ``pycolmap.Reconstruction``, runs
    reprojection-error bundle adjustment (see :func:`_optimize`), and writes a
    new, valid COLMAP model to ``output_dir``. Point at ``output_dir`` with
    ``examples.datasets.colmap.Parser(..., colmap_dir=output_dir)`` (or
    ``Parser(data_dir, ...)`` with ``output_dir`` placed at
    ``<data_dir>/sparse/refined``) to train on the refined poses/points with no
    other changes.

    Args:
        colmap_dir: Path to the input COLMAP sparse model (containing
            cameras/images/points3D files).
        output_dir: Path to write the refined COLMAP sparse model to.
        num_iters: Number of Adam optimization steps.
        lr: Initial learning rate (decayed 10x over ``num_iters``).
        huber_delta: Huber loss transition point, in pixels.
        anchor_reg: Weight keeping the refined poses/points close to their
            COLMAP initialization, preventing the reconstruction from drifting
            beyond the anchor's gauge fix.
        refine_points: If True, also refine the 3D point positions. If False,
            only camera poses are refined (points are kept at their COLMAP
            positions).
        device: Torch device to run the optimization on.

    Returns:
        A dict of summary stats: number of images/points/observations, and the
        mean pixel reprojection error before and after refinement.
    """
    import os

    import pycolmap

    reconstruction = pycolmap.Reconstruction(colmap_dir)
    cameras = {int(k): v for k, v in reconstruction.cameras.items()}
    images = {int(k): v for k, v in reconstruction.images.items()}
    points3D = {int(k): v for k, v in reconstruction.points3D.items()}

    image_ids = sorted(images.keys())
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    point3D_ids = sorted(points3D.keys())
    point3D_id_to_idx = {pid: i for i, pid in enumerate(point3D_ids)}

    w2c = np.zeros((len(image_ids), 4, 4), dtype=np.float64)
    Ks = np.zeros((len(image_ids), 3, 3), dtype=np.float64)
    track_lengths = np.zeros(len(image_ids), dtype=np.int64)
    for iid, idx in image_id_to_idx.items():
        im = images[iid]
        cam_from_world = im.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        mat = np.eye(4)
        mat[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        w2c[idx] = mat
        cam = cameras[int(im.camera_id)]
        Ks[idx] = np.asarray(cam.calibration_matrix(), dtype=np.float64)

    points_init = np.array(
        [points3D[pid].xyz for pid in point3D_ids], dtype=np.float64
    ).reshape(-1, 3)

    obs_image_idx: List[int] = []
    obs_point_idx: List[int] = []
    obs_xy: List[np.ndarray] = []
    for pid in point3D_ids:
        p_idx = point3D_id_to_idx[pid]
        for elem in points3D[pid].track.elements:
            iid = int(elem.image_id)
            if iid not in image_id_to_idx:
                continue
            img_idx = image_id_to_idx[iid]
            xy = np.asarray(
                images[iid].points2D[int(elem.point2D_idx)].xy, dtype=np.float64
            )
            obs_image_idx.append(img_idx)
            obs_point_idx.append(p_idx)
            obs_xy.append(xy)
            track_lengths[img_idx] += 1

    if len(obs_xy) == 0:
        raise ValueError(f"No point-track observations found in {colmap_dir}.")

    anchor_image_idx = int(np.argmax(track_lengths))

    result = _optimize(
        w2c_init=torch.from_numpy(w2c).float().to(device),
        Ks=torch.from_numpy(Ks).float().to(device),
        points_init=torch.from_numpy(points_init).float().to(device),
        obs_image_idx=torch.tensor(obs_image_idx, dtype=torch.long, device=device),
        obs_point_idx=torch.tensor(obs_point_idx, dtype=torch.long, device=device),
        obs_xy=torch.from_numpy(np.stack(obs_xy)).float().to(device),
        anchor_image_idx=anchor_image_idx,
        num_iters=num_iters,
        lr=lr,
        huber_delta=huber_delta,
        anchor_reg=anchor_reg,
        refine_points=refine_points,
    )

    w2c_refined = result["w2c"].cpu().numpy()
    points_refined = result["points"].cpu().numpy()

    for iid, idx in image_id_to_idx.items():
        R_refined = w2c_refined[idx, :3, :3]
        t_refined = w2c_refined[idx, :3, 3]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(R_refined), t_refined)
        image = reconstruction.images[iid]
        # Newer pycolmap (rig/frame model): Image.cam_from_world is
        # read-only, poses live on the image's Frame. Older pycolmap:
        # Image.cam_from_world is a plain settable attribute.
        frame_id = getattr(image, "frame_id", None)
        if frame_id is not None:
            reconstruction.frame(frame_id).set_cam_from_world(image.camera_id, pose)
        else:
            image.cam_from_world = pose
    if refine_points:
        for pid in point3D_ids:
            reconstruction.points3D[pid].xyz = points_refined[point3D_id_to_idx[pid]]

    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write(output_dir)

    return {
        "num_images": len(image_ids),
        "num_points": len(point3D_ids),
        "num_observations": len(obs_xy),
        "mean_reprojection_error_before": result["mean_reprojection_error_before"],
        "mean_reprojection_error_after": result["mean_reprojection_error_after"],
    }
