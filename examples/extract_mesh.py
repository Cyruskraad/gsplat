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
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
import tyro
from datasets.colmap import Dataset, Parser

from gsplat.photogrammetry.mesh_extraction import (
    bake_ambient_occlusion,
    bake_mesh_texture,
    bake_normal_map,
    extract_mesh_poisson,
    extract_mesh_tsdf,
    simplify_mesh,
    simplify_mesh_to_error,
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
    # Robust multi-view fusion: discard observations more than this many
    # standard deviations from a point's own mean colour before averaging, so
    # a specular highlight or a slightly misregistered camera doesn't get
    # blended into the texture as ghosting. 0 disables it (plain weighted
    # mean). Only helps where the bad views are a per-point minority -- use
    # --mask_dir for content that occludes a surface in most views.
    texture_outlier_sigma: float = 0.0
    # Texture each face from a single chosen view instead of blending every
    # view that sees it (Waechter et al., "Let There Be Color!", ECCV 2014).
    # Blending is a low-pass filter -- views are never registered to sub-pixel
    # accuracy after real SfM -- so this keeps detail that would otherwise be
    # averaged away. It is a *tradeoff*, not a strict win: the result is
    # sharper but pointwise less accurate, so it is off by default. Requires
    # --texture_mode atlas. Complements --texture_outlier_sigma, which still
    # governs the blended fallback regions.
    texture_view_selection: bool = False
    # Seam penalty for --texture_view_selection: how much worse a view the
    # labelling will accept to avoid a colour discontinuity between two
    # neighbouring faces. Higher means fewer, larger single-view regions.
    texture_mrf_lambda: float = 1.0
    # Seam levelling for --texture_view_selection: how smoothly the per-view
    # colour correction is spread over each single-view region. Too small and
    # the correction is a sharp patch around each seam; too large and it cannot
    # close the seam at all. 0 disables levelling, which is only useful for
    # measuring what it was doing.
    texture_seam_smoothness: float = 0.1
    # Decimate the extracted mesh to roughly this many triangles before
    # texturing (quadric error metrics). TSDF/Poisson output is tessellated to
    # the voxel grid rather than to the scene's complexity, so this is usually
    # a large reduction. Combine with --normal_map to keep the detail.
    target_triangles: Optional[int] = None
    # Decimate to a *fit target* instead of a triangle count: the cloud-to-mesh
    # distance you are willing to accept, in units of the reference cloud's own
    # k-NN spacing (the same scale-free reading as the pipeline report's
    # `mesh_fit_over_point_spacing`). 1.0 means "stay within the cloud's own
    # sampling noise"; 2-4 gives a much lighter mesh for a viewer. Usually the
    # better question to answer than a triangle budget, since how many
    # triangles a scene needs depends on the scene. Mutually exclusive with
    # --target_triangles.
    target_fit_ratio: Optional[float] = None
    # Bake the pre-decimation mesh's normals into a normal map on the textured
    # mesh's UV atlas, so the decimated mesh still shades like the dense one.
    # Requires --texture_mode atlas. Writes mesh_normal.png next to mesh.obj.
    normal_map: bool = False
    # Normal-map space. "tangent" is what engines expect; "object" is simpler
    # and immune to UV-seam tangent artifacts, fine for a static scanned asset.
    normal_map_space: Literal["tangent", "object"] = "tangent"
    # Bake an ambient-occlusion map onto the same UV atlas: how much of the sky
    # each point can see, so creases and contact points darken. Requires
    # --texture_mode atlas. Writes mesh_ao.png next to mesh.obj.
    ao_map: bool = False
    # Rays per texel for --ao_map. Noise falls as 1/sqrt(n); 64 previews, a few
    # hundred is smooth (and proportionally slower).
    ao_samples: int = 64
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

    if (cfg.normal_map or cfg.ao_map) and cfg.texture_mode != "atlas":
        raise ValueError(
            "--normal_map/--ao_map need UV coordinates to bake into, so they "
            "require --texture_mode atlas (which also switches the output to "
            ".obj)."
        )
    if cfg.texture_view_selection and cfg.texture_mode != "atlas":
        raise ValueError(
            "--texture_view_selection chooses a view per *face* and bakes it "
            "into an atlas, so it requires --texture_mode atlas. Per-vertex "
            "colors have nothing to select a view for."
        )

    if cfg.target_triangles is not None and cfg.target_fit_ratio is not None:
        raise ValueError(
            "--target_triangles and --target_fit_ratio are two ways of asking "
            "the same question (how small a mesh?) and disagree about the "
            "answer. Pass one: a triangle budget, or the cloud-to-mesh fit you "
            "will accept."
        )

    # Decimate before texturing, so the atlas is built on the mesh that ships.
    # Keep the dense mesh: it is what --normal_map bakes detail from.
    dense_mesh = mesh
    decimation_stats = None
    if cfg.target_fit_ratio is not None:
        if len(mesh.triangles) == 0:
            print(
                "[extract_mesh] WARNING: nothing to decimate -- the extracted "
                "mesh has no triangles."
            )
        else:
            mesh, decimation_stats = simplify_mesh_to_error(
                mesh, reference_points, error_over_spacing=cfg.target_fit_ratio
            )
            print(
                f"[extract_mesh] decimated {decimation_stats['triangles_before']}"
                f" -> {decimation_stats['triangles_after']} triangles "
                f"({decimation_stats['reduction']:.1%} fewer) in "
                f"{decimation_stats['num_probes']} probes; cloud-to-mesh "
                f"{decimation_stats['error_before']:.5g} -> "
                f"{decimation_stats['error_after']:.5g} against a budget of "
                f"{decimation_stats['max_error']:.5g} "
                f"({cfg.target_fit_ratio} x the cloud's "
                f"{decimation_stats['point_spacing']:.5g} spacing)"
            )
            if not decimation_stats["target_met"]:
                warnings.warn(
                    "The mesh already misses --target_fit_ratio before any "
                    f"decimation (cloud-to-mesh "
                    f"{decimation_stats['error_before']:.5g} > "
                    f"{decimation_stats['max_error']:.5g}), so it was left "
                    "alone -- decimating can only move it further from the "
                    "cloud. Either the extraction is a poor fit (check "
                    "--voxel_size / --poisson_depth) or the target is tighter "
                    "than this reconstruction can be.",
                    RuntimeWarning,
                )
    elif cfg.target_triangles is not None:
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
    view_selection_stats: dict = {}
    if cfg.bake_texture_:
        mesh, texture = bake_mesh_texture(
            mesh,
            dataset,
            mode=cfg.texture_mode,
            texture_size=cfg.texture_size,
            outlier_sigma=(
                cfg.texture_outlier_sigma if cfg.texture_outlier_sigma > 0 else None
            ),
            view_selection=cfg.texture_view_selection,
            mrf_smoothness=cfg.texture_mrf_lambda,
            seam_smoothness=(
                cfg.texture_seam_smoothness if cfg.texture_seam_smoothness > 0 else None
            ),
            stats_out=view_selection_stats,
        )
        if texture is not None:
            print(
                f"[extract_mesh] baked a {texture.shape[1]}x{texture.shape[0]} "
                "UV texture atlas from training images"
            )
        else:
            print("[extract_mesh] baked per-vertex texture from training images")
        if view_selection_stats:
            mrf = view_selection_stats["mrf"]
            sharp = view_selection_stats["atlas_sharpness"]["mean_gradient"]
            blended_sharp = view_selection_stats["blended_atlas_sharpness"][
                "mean_gradient"
            ]
            print(
                f"[extract_mesh] view selection: {mrf['num_views_used']} views "
                f"over {mrf['num_faces']} faces, {mrf['num_seams']} seams, "
                f"{view_selection_stats['fallback_fraction']:.1%} of texels fell "
                "back to blending"
            )
            print(
                f"[extract_mesh] atlas sharpness {sharp:.4f} vs {blended_sharp:.4f} "
                f"blended ({sharp / blended_sharp - 1.0:+.1%})"
                if blended_sharp > 0
                else f"[extract_mesh] atlas sharpness {sharp:.4f}"
            )
            if blended_sharp > 0 and sharp <= blended_sharp:
                warnings.warn(
                    "View selection produced a *less* sharp atlas than blending "
                    f"({sharp:.4f} vs {blended_sharp:.4f}). That usually means "
                    "the views are already well registered, in which case "
                    "blending is the better choice here -- it is pointwise more "
                    "accurate. Consider dropping --texture_view_selection.",
                    RuntimeWarning,
                )
            if mrf["num_unlabelled"]:
                print(
                    f"[extract_mesh] {mrf['num_unlabelled']} faces could not be "
                    "textured from any single view and kept the blended color"
                )
            if "seam_discontinuity" in view_selection_stats:
                before = view_selection_stats["seam_discontinuity_before"]["mean"]
                after = view_selection_stats["seam_discontinuity"]["mean"]
                levelling = view_selection_stats["seam_levelling"]
                print(
                    f"[extract_mesh] seam levelling: discontinuity {before:.4f} "
                    f"-> {after:.4f} over {levelling['num_seam_edges']} seam "
                    f"edges ({levelling['iterations']} CG iterations)"
                )
                if after >= before:
                    warnings.warn(
                        f"Seam levelling did not reduce the seam discontinuity "
                        f"({before:.4f} -> {after:.4f}). "
                        "--texture_seam_smoothness is probably wrong for this "
                        "scene: too large and the correction cannot bend enough "
                        "to close a seam, too small and it fits sampling noise "
                        "instead of the exposure difference.",
                        RuntimeWarning,
                    )
                if not levelling["converged"]:
                    warnings.warn(
                        "Seam levelling's conjugate-gradient solve hit its "
                        f"iteration cap with relative residual "
                        f"{levelling['residual']:.2e}; the correction applied is "
                        "the partial solution. A very small "
                        "--texture_seam_smoothness conditions this badly.",
                        RuntimeWarning,
                    )

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

    ao_stats = None
    if cfg.ao_map:
        # Self-occlusion on the mesh that ships, unlike the normal map's
        # dense-vs-decimated bake. Casting against the dense mesh would need a
        # cage large enough to clear the decimation gap (most of a simplified
        # surface sits *inside* the mesh it came from), and that cage erases
        # occlusion detail finer than itself -- so it costs the fine cues it
        # was meant to add. AO's real signal is large cavities and creases,
        # which survive decimation. Pass `occluder_mesh` via the Python API if
        # you want the dense bake anyway. Shares the albedo's UVs regardless.
        mesh, ao_image, ao_stats = bake_ambient_occlusion(
            mesh,
            texture_size=cfg.texture_size,
            num_samples=cfg.ao_samples,
        )
        print(
            f"[extract_mesh] baked an ambient-occlusion map "
            f"(mean={ao_stats['mean_ao']:.3f}, min={ao_stats['min_ao']:.3f}, "
            f"{ao_stats['num_samples']} rays/texel)"
        )
        if ao_stats["mean_ao"] > 0.999:
            print(
                "[extract_mesh] NOTE: nothing occluded anything, so the AO map "
                "is uniformly white. Expected for a convex shape; otherwise the "
                f"occlusion distance ({ao_stats['max_distance']:.4g} scene "
                "units) may be too small."
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

    if ao_stats is not None:
        ao_path = os.path.join(cfg.result_dir, "mesh_ao.png")
        imageio.imwrite(ao_path, ao_image)
        # There is no standard MTL key for an AO map (it is an engine-side
        # input, not a Wavefront material property), so reference it as a
        # comment rather than inventing a key importers would choke on.
        mtl_path = os.path.splitext(out_path)[0] + ".mtl"
        if os.path.exists(mtl_path):
            with open(mtl_path, "a") as f:
                f.write(f"# ambient occlusion map: {os.path.basename(ao_path)}\n")
        print(f"[extract_mesh] wrote {ao_path}")

    stats = mesh_quality_stats(mesh)
    if decimation_stats is not None:
        stats["decimation"] = decimation_stats
    if normal_map_stats is not None:
        stats["normal_map"] = normal_map_stats
    if ao_stats is not None:
        stats["ambient_occlusion"] = ao_stats
    if view_selection_stats:
        stats["view_selection"] = view_selection_stats
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
