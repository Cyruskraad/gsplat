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

Pass `--texture_mode atlas` to UV-unwrap the mesh and bake a texture atlas
instead of per-vertex colors, writing `mesh.obj` + `mesh.mtl` + `mesh_0.png`
(loadable with its texture in standard DCC tools and game engines).
"""

import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
import tyro
from datasets.colmap import Dataset, Parser

from gsplat.photogrammetry.mesh_extraction import (
    bake_mesh_texture,
    bake_normal_map,
    extract_mesh_poisson,
    extract_mesh_tsdf,
    simplify_mesh,
)
from gsplat.photogrammetry.metrics import mesh_quality_stats, point_to_mesh_distance


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
    # Directory of precomputed per-image transient/dynamic-object masks (see
    # docs/photogrammetry.md), one `<image_stem>.png` per training image:
    # nonzero = keep (static content), 0 = exclude. Excluded pixels are
    # dropped from TSDF fusion (--method tsdf only).
    mask_dir: Optional[str] = None
    # Directory to write mesh.ply to.
    result_dir: str = "results/garden"
    # Whether to bake texture from the training images.
    bake_texture_: bool = True
    # How to represent the baked texture. "vertex" writes per-vertex colors
    # into a .ply; "atlas" UV-unwraps the mesh and bakes a texture image,
    # writing a .obj + .mtl + .png that standard DCC tools and game engines
    # load with the texture attached. "atlas" resolves detail finer than the
    # mesh's vertex spacing; "vertex" is cheaper and works on any mesh.
    texture_mode: Literal["vertex", "atlas"] = "vertex"
    # Atlas width/height in texels (--texture_mode atlas only).
    texture_size: int = 2048
    # Decimate the extracted mesh to roughly this many triangles before
    # texturing (quadric error metrics). TSDF/Poisson output is tessellated to
    # the voxel grid rather than to the scene's complexity, so this is usually
    # a large reduction. Combine with --normal_map to keep the detail.
    target_triangles: Optional[int] = None
    # Bake the pre-decimation mesh's normals into a normal map on the textured
    # mesh's UV atlas, so the decimated mesh still shades like the dense one.
    # Requires --texture_mode atlas. Writes mesh_normal.png next to mesh.obj.
    normal_map: bool = False
    # Normal-map space. "tangent" is what engines expect; "object" is simpler
    # and immune to UV-seam tangent artifacts, fine for a static scanned asset.
    normal_map_space: Literal["tangent", "object"] = "tangent"
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
    dataset = Dataset(parser, split="train", mask_dir=cfg.mask_dir)

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
        # No dense MVS cloud on this path -- fall back to the sparse SfM
        # cloud as the cloud-to-mesh fit reference.
        reference_points = parser.points
    elif cfg.method == "poisson":
        assert cfg.dense_points, "--dense_points is required for --method poisson."
        pcd = o3d.io.read_point_cloud(cfg.dense_points)
        points_xyz = np.asarray(pcd.points)
        points_rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
        mesh = extract_mesh_poisson(points_xyz, points_rgb, depth=cfg.poisson_depth)
        reference_points = points_xyz
    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")

    print(
        f"[extract_mesh] extracted mesh with {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )

    if cfg.normal_map and cfg.texture_mode != "atlas":
        raise ValueError(
            "--normal_map needs UV coordinates to bake into, so it requires "
            "--texture_mode atlas (which also switches the output to .obj)."
        )

    # Decimate before texturing, so the atlas is built on the mesh that ships.
    # Keep the dense mesh: it is what --normal_map bakes detail from.
    dense_mesh = mesh
    decimation_stats = None
    if cfg.target_triangles is not None:
        before = len(mesh.triangles)
        mesh = simplify_mesh(mesh, target_triangles=cfg.target_triangles)
        after = len(mesh.triangles)
        decimation_stats = {
            "triangles_before": before,
            "triangles_after": after,
            "reduction": 1.0 - (after / before) if before else 0.0,
            "target_triangles": cfg.target_triangles,
        }
        print(
            f"[extract_mesh] decimated {before} -> {after} triangles "
            f"({decimation_stats['reduction']:.1%} fewer)"
        )

    texture = None
    if cfg.bake_texture_:
        mesh, texture = bake_mesh_texture(
            mesh,
            dataset,
            mode=cfg.texture_mode,
            texture_size=cfg.texture_size,
        )
        if texture is not None:
            print(
                f"[extract_mesh] baked a {texture.shape[1]}x{texture.shape[0]} "
                "UV texture atlas from training images"
            )
        else:
            print("[extract_mesh] baked per-vertex texture from training images")

    normal_map_stats = None
    if cfg.normal_map:
        # Runs after the color atlas so it reuses those UVs -- open3d's
        # unwrapper is non-deterministic, so a second unwrap would give the
        # normal map a different layout from the albedo.
        mesh, normal_map, normal_map_stats = bake_normal_map(
            dense_mesh,
            mesh,
            texture_size=cfg.texture_size,
            space=cfg.normal_map_space,
        )
        print(
            f"[extract_mesh] baked a {cfg.normal_map_space}-space normal map "
            f"({normal_map_stats['hit_fraction']:.1%} of texels hit the dense "
            "mesh)"
        )
        if normal_map_stats["hit_fraction"] < 0.5:
            print(
                "[extract_mesh] WARNING: most texels missed the dense mesh, so "
                "the normal map is mostly flat. The ray cage "
                f"({normal_map_stats['cage']:.4g} scene units) is probably too "
                "small to span the gap between the two meshes."
            )

    os.makedirs(cfg.result_dir, exist_ok=True)
    # A UV atlas needs a format that can carry UVs and a texture image; .ply
    # cannot, so an atlas mesh is written as .obj (+ .mtl + .png alongside).
    out_path = os.path.join(
        cfg.result_dir, "mesh.obj" if texture is not None else "mesh.ply"
    )
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"[extract_mesh] wrote {out_path}")

    if normal_map_stats is not None:
        normal_path = os.path.join(cfg.result_dir, "mesh_normal.png")
        imageio.imwrite(normal_path, normal_map)
        # open3d's OBJ writer emits map_Kd but nothing for a normal map, so
        # reference it here -- otherwise the .png ships beside an .mtl that
        # never mentions it and every importer ignores it.
        mtl_path = os.path.splitext(out_path)[0] + ".mtl"
        if os.path.exists(mtl_path):
            with open(mtl_path, "a") as f:
                f.write(f"norm {os.path.basename(normal_path)}\n")
                f.write(f"map_Bump {os.path.basename(normal_path)}\n")
        print(f"[extract_mesh] wrote {normal_path} (referenced from {mtl_path})")

    stats = mesh_quality_stats(mesh)
    if decimation_stats is not None:
        stats["decimation"] = decimation_stats
    if normal_map_stats is not None:
        stats["normal_map"] = normal_map_stats
    if len(mesh.triangles) == 0:
        # Extraction produced nothing usable. Say so plainly instead of
        # letting the cloud-to-mesh measurement fail against an empty surface.
        stats["point_to_mesh"] = None
        print(
            "[extract_mesh] WARNING: the extracted mesh has no triangles, so "
            "there is no cloud-to-mesh fit to measure. Check --voxel_size / "
            "--sdf_trunc (TSDF) or --poisson_depth for this scene."
        )
    else:
        stats["point_to_mesh"] = point_to_mesh_distance(reference_points, mesh)
        print(
            f"[extract_mesh] watertight={stats['is_watertight']} "
            f"components={stats['num_connected_components']} "
            f"cloud-to-mesh mean={stats['point_to_mesh']['mean']:.4f}"
        )
    stats_path = os.path.join(cfg.result_dir, "mesh_metrics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[extract_mesh] wrote stats to {stats_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
