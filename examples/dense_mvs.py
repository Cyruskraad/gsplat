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
"""Densify a COLMAP sparse point cloud via COLMAP's own dense MVS pipeline.

Requires a CUDA-enabled `colmap` binary on PATH (the `pycolmap` Python
bindings alone are not sufficient -- patch-match stereo is only exposed
through the COLMAP CLI). See https://colmap.github.io/install.html.

Example:

    python examples/dense_mvs.py --data_dir data/360_v2/garden \\
        --colmap_dir data/360_v2/garden/sparse/refined

writes `data/360_v2/garden/dense/dense.ply`, plus per-view depth maps, which
can then be passed to `examples.datasets.colmap.Parser`'s `dense_points_path`
argument to densify Gaussian initialization, or to
`gsplat.photogrammetry.mesh_extraction.extract_mesh_poisson`.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import tyro

from gsplat.photogrammetry.dense_mvs import run_dense_mvs
from gsplat.photogrammetry.metrics import point_cloud_stats


@dataclass
class Config:
    # Dataset root directory (used to locate images/ if image_dir is not set).
    data_dir: str = "data/360_v2/garden"
    # COLMAP sparse model to densify. Defaults to data_dir/sparse/0 (or
    # data_dir/sparse) if not given.
    colmap_dir: Optional[str] = None
    # Workspace directory for undistorted images, depth maps, and dense.ply.
    output_dir: Optional[str] = None
    # Directory of source images. Defaults to data_dir/images.
    image_dir: Optional[str] = None
    # Maximum image dimension used during dense stereo.
    max_image_size: int = 2000


def main(cfg: Config) -> None:
    colmap_dir = cfg.colmap_dir
    if colmap_dir is None:
        colmap_dir = os.path.join(cfg.data_dir, "sparse/0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(cfg.data_dir, "sparse")
    output_dir = cfg.output_dir or os.path.join(cfg.data_dir, "dense")

    dense_ply = run_dense_mvs(
        data_dir=cfg.data_dir,
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        image_dir=cfg.image_dir,
        max_image_size=cfg.max_image_size,
    )
    print(f"[dense_mvs] wrote fused dense point cloud to {dense_ply}")

    pcd = o3d.io.read_point_cloud(dense_ply)
    stats = point_cloud_stats(np.asarray(pcd.points))
    print(
        f"[dense_mvs] {stats['num_points']} points, "
        f"mean k-NN spacing {stats['mean_knn_distance']:.4f}"
    )
    stats_path = os.path.join(output_dir, "dense_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[dense_mvs] wrote stats to {stats_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
