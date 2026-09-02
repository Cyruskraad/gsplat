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
"""Extract a textured triangle mesh from a trained 2DGS/3DGS checkpoint.

Requires the optional `open3d` dependency: `pip install gsplat[mesh]`.

Example:

    python examples/extract_mesh.py \\
        --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \\
        --data_dir data/360_v2/garden --result_dir results/garden_2dgs

writes `results/garden_2dgs/mesh.ply`. Pass `--method poisson
--dense_points data/360_v2/garden/dense/dense.ply` to instead run Poisson
reconstruction over a dense MVS point cloud (see `examples/dense_mvs.py`).
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import torch
import tyro
from datasets.colmap import Dataset, Parser

from gsplat.photogrammetry.mesh_extraction import (
    bake_texture,
    extract_mesh_poisson,
    extract_mesh_tsdf,
)


@dataclass
class Config:
    # Path to a gsplat checkpoint (.pt) with a "splats" state dict of
    # SH-color Gaussians (as saved by simple_trainer.py / simple_trainer_2dgs.py
    # without --app_opt).
    ckpt: str = ""
    # Dataset root directory (used to build camera poses to render/bake from).
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset.
    data_factor: int = 4
    # Every N images there is a test image; use a large value (e.g. 10_000) to
    # put ~all images in the "train" split used for mesh extraction/baking.
    test_every: int = 8
    # Reconstruction method.
    method: Literal["tsdf", "poisson"] = "tsdf"
    # Renderer used to produce depth maps for TSDF fusion.
    renderer: Literal["2dgs", "3dgs"] = "2dgs"
    # Path to a dense MVS point cloud (see examples/dense_mvs.py). Required
    # for --method poisson; optional fallback source for TSDF is unused.
    dense_points: Optional[str] = None
    # TSDF voxel size, in scene units.
    voxel_size: float = 0.01
    # TSDF truncation distance, in scene units.
    sdf_trunc: float = 0.04
    # Poisson reconstruction octree depth.
    poisson_depth: int = 9
    # Directory to write mesh.ply to.
    result_dir: str = "results/garden"
    # Whether to bake per-vertex texture from the training images.
    bake_texture_: bool = True
    # Torch device.
    device: str = "cuda"


def main(cfg: Config) -> None:
    assert cfg.ckpt, "--ckpt is required."
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=True,
        test_every=cfg.test_every,
    )
    dataset = Dataset(parser, split="train")

    if cfg.method == "tsdf":
        ckpt = torch.load(cfg.ckpt, map_location=cfg.device)
        splats = {k: v.to(cfg.device) for k, v in ckpt["splats"].items()}
        mesh = extract_mesh_tsdf(
            splats,
            dataset,
            renderer=cfg.renderer,
            voxel_size=cfg.voxel_size,
            sdf_trunc=cfg.sdf_trunc,
            device=cfg.device,
        )
    elif cfg.method == "poisson":
        assert cfg.dense_points, "--dense_points is required for --method poisson."
        pcd = o3d.io.read_point_cloud(cfg.dense_points)
        points_xyz = np.asarray(pcd.points)
        points_rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
        mesh = extract_mesh_poisson(points_xyz, points_rgb, depth=cfg.poisson_depth)
    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")

    print(
        f"[extract_mesh] extracted mesh with {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )

    if cfg.bake_texture_:
        mesh = bake_texture(mesh, dataset)
        print("[extract_mesh] baked per-vertex texture from training images")

    os.makedirs(cfg.result_dir, exist_ok=True)
    out_path = os.path.join(cfg.result_dir, "mesh.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"[extract_mesh] wrote {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
