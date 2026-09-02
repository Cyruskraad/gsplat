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
"""Refine a COLMAP reconstruction's poses (and points) via bundle adjustment.

Example:

    python examples/bundle_adjust.py --data_dir data/360_v2/garden

writes a refined COLMAP model to ``data/360_v2/garden/sparse/refined``, which
can then be used in place of ``sparse/0`` by pointing
``examples.datasets.colmap.Parser`` at it via its ``colmap_dir`` argument.
"""

import os
from dataclasses import dataclass

import tyro

from gsplat.photogrammetry.bundle_adjustment import refine_reconstruction


@dataclass
class Config:
    # Dataset root directory.
    data_dir: str = "data/360_v2/garden"
    # Sub-directory (relative to data_dir) containing the input COLMAP sparse
    # model. Defaults to sparse/0 if it exists, else sparse.
    colmap_subdir: str = ""
    # Sub-directory (relative to data_dir) to write the refined COLMAP sparse
    # model to.
    output_subdir: str = "sparse/refined"
    # Number of Adam optimization steps.
    num_iters: int = 2000
    # Initial learning rate (decayed 10x over num_iters).
    lr: float = 1e-3
    # Huber loss transition point, in pixels.
    huber_delta: float = 1.0
    # Weight keeping refined poses/points close to their COLMAP initialization.
    anchor_reg: float = 1e-4
    # Whether to also refine 3D point positions (else only camera poses).
    refine_points: bool = True
    # Torch device to run the optimization on.
    device: str = "cuda"


def main(cfg: Config) -> None:
    colmap_dir = cfg.colmap_subdir
    if not colmap_dir:
        colmap_dir = os.path.join(cfg.data_dir, "sparse/0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(cfg.data_dir, "sparse")
    else:
        colmap_dir = os.path.join(cfg.data_dir, colmap_dir)
    output_dir = os.path.join(cfg.data_dir, cfg.output_subdir)

    print(f"[bundle_adjust] refining {colmap_dir} -> {output_dir}")
    stats = refine_reconstruction(
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        num_iters=cfg.num_iters,
        lr=cfg.lr,
        huber_delta=cfg.huber_delta,
        anchor_reg=cfg.anchor_reg,
        refine_points=cfg.refine_points,
        device=cfg.device,
    )
    print(
        f"[bundle_adjust] {stats['num_images']} images, {stats['num_points']} points, "
        f"{stats['num_observations']} observations"
    )
    print(
        "[bundle_adjust] mean reprojection error: "
        f"{stats['mean_reprojection_error_before']:.4f}px -> "
        f"{stats['mean_reprojection_error_after']:.4f}px"
    )
    print(f"[bundle_adjust] wrote refined reconstruction to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
