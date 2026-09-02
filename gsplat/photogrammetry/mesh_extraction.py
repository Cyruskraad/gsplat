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
"""Mesh / surface reconstruction from a trained Gaussian-splat scene.

gsplat has no built-in way to turn a trained 2DGS/3DGS scene into an actual
surface mesh -- this module closes that gap with two complementary paths:

- :func:`extract_mesh_tsdf`: render color + median depth (+ depth-derived
  normals) from the trained splats at each training camera pose, and fuse
  them into a TSDF volume (as in the original 2D Gaussian Splatting paper's
  mesh-extraction recipe), then extract a cleaned triangle mesh via marching
  cubes.
- :func:`extract_mesh_poisson`: Poisson surface reconstruction over a point
  cloud (typically the dense MVS cloud from
  :mod:`gsplat.photogrammetry.dense_mvs`, or the sparse SfM cloud as a
  fallback).

:func:`bake_texture` then colors a mesh's vertices from the training images,
with occlusion-aware, view-angle-weighted blending across views.

Requires the optional ``open3d`` dependency: ``pip install gsplat[mesh]``.
Only SH-color checkpoints (containing ``"sh0"``/``"shN"``) are supported --
appearance-embedding checkpoints (``"features"``/``"colors"``) are out of
scope, since per-image appearance variation doesn't map onto a single
canonical mesh texture.
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from ..rendering import rasterization, rasterization_2dgs


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "gsplat.photogrammetry.mesh_extraction requires open3d. Install "
            "it with `pip install gsplat[mesh]` (or `pip install open3d`)."
        ) from e
    return o3d


def _splat_colors(splats: Dict[str, Tensor]) -> Tensor:
    if "sh0" not in splats or "shN" not in splats:
        raise ValueError(
            "mesh_extraction only supports SH-color checkpoints (splats "
            "containing 'sh0'/'shN'), not appearance-embedding checkpoints "
            "('features'/'colors'). Train with the default `feature_dim=None` "
            "(i.e. without --app_opt) to produce a compatible checkpoint."
        )
    return torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]


def _clean_mesh(mesh, min_cluster_fraction: float = 0.02):
    """Remove degenerate geometry and small floating components."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangle_clusters = np.asarray(triangle_clusters)
    if len(cluster_n_triangles) > 0:
        min_size = max(int(min_cluster_fraction * cluster_n_triangles.max()), 1)
        remove_mask = cluster_n_triangles[triangle_clusters] < min_size
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    return mesh


@torch.no_grad()
def extract_mesh_tsdf(
    splats: Dict[str, Tensor],
    dataset,
    renderer: str = "2dgs",
    sh_degree: int = 3,
    voxel_size: float = 0.01,
    sdf_trunc: float = 0.04,
    depth_trunc: float = 10.0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    device: str = "cuda",
):
    """Extract a triangle mesh via TSDF fusion of rendered depth maps.

    Args:
        splats: Trained Gaussian parameters, as saved in a gsplat checkpoint's
            ``"splats"`` entry (``means``, ``quats``, unactivated ``scales``
            and ``opacities``, and SH color ``sh0``/``shN``).
        dataset: An ``examples.datasets.colmap.Dataset`` (or any indexable
            object yielding dicts with ``"camtoworld"`` (4, 4) and ``"K"``
            (3, 3) tensors and an ``"image"`` tensor whose shape gives the
            render resolution) providing the camera poses to render+fuse
            from. Using a dataset built with a larger ``test_every`` (i.e.
            more train views) improves mesh coverage.
        renderer: ``"2dgs"`` (recommended -- surfels give much better depth
            for TSDF fusion) or ``"3dgs"``.
        sh_degree: SH degree to evaluate ``splats``' colors at.
        voxel_size: TSDF voxel size, in scene units.
        sdf_trunc: TSDF truncation distance, in scene units.
        depth_trunc: Maximum depth (scene units) to integrate; farther pixels
            are ignored (background/sky).
        device: Torch device the trained splats live on / rendering runs on.

    Returns:
        An ``open3d.geometry.TriangleMesh``, cleaned of degenerate geometry
        and small floating components.
    """
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = _splat_colors(splats).to(device)

    views = []
    for i in range(len(dataset)):
        data = dataset[i]
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        height, width = data["image"].shape[:2]
        viewmat = torch.linalg.inv(camtoworld)

        if renderer == "2dgs":
            (
                render_colors,
                _render_alphas,
                _render_normals,
                _render_normals_from_depth,
                _render_distort,
                render_median,
                _meta,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
                near_plane=near_plane,
                far_plane=far_plane,
                sh_degree=sh_degree,
                render_mode="RGB",
                depth_mode="median",
            )
            color = render_colors[0, ..., :3]
            depth = render_median[0, ..., 0]
        elif renderer == "3dgs":
            render_colors, _render_alphas, _meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
                near_plane=near_plane,
                far_plane=far_plane,
                sh_degree=sh_degree,
                render_mode="RGB+ED",
            )
            color = render_colors[0, ..., :3]
            depth = render_colors[0, ..., 3]
        else:
            raise ValueError(f"Unknown renderer: {renderer!r}. Use '2dgs' or '3dgs'.")

        views.append(
            {
                "color": (color.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                "depth": depth.cpu().numpy().astype(np.float32),
                "K": K.cpu().numpy(),
                # `extrinsic` is the world-to-camera transform.
                "extrinsic": viewmat.cpu().numpy().astype(np.float64),
            }
        )

    return _tsdf_fuse(
        views, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
    )


def _tsdf_fuse(
    views,
    voxel_size: float = 0.01,
    sdf_trunc: float = 0.04,
    depth_trunc: float = 10.0,
):
    """Fuse a list of posed (color, depth) views into a triangle mesh via TSDF.

    Pure Open3D/numpy -- no gsplat rendering involved, so it can be exercised
    directly with synthetic depth maps in tests without a GPU or trained
    splats.

    Args:
        views: A list of dicts, one per view, each with:
            ``"color"``: (H, W, 3) uint8 RGB image.
            ``"depth"``: (H, W) float32 z-depth map, in scene units (0 = no
                depth at that pixel).
            ``"K"``: (3, 3) camera intrinsics.
            ``"extrinsic"``: (4, 4) world-to-camera transform.
        voxel_size: TSDF voxel size, in scene units.
        sdf_trunc: TSDF truncation distance, in scene units.
        depth_trunc: Maximum depth (scene units) to integrate.

    Returns:
        An ``open3d.geometry.TriangleMesh``, cleaned of degenerate geometry
        and small floating components.
    """
    o3d = _require_open3d()

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for view in views:
        height, width = view["depth"].shape[:2]
        o3d_color = o3d.geometry.Image(np.ascontiguousarray(view["color"]))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(view["depth"]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        K = view["K"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        volume.integrate(rgbd, intrinsic, view["extrinsic"])

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return _clean_mesh(mesh)


def extract_mesh_poisson(
    points_xyz: np.ndarray,
    points_rgb: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    depth: int = 9,
    density_quantile_threshold: float = 0.01,
):
    """Poisson surface reconstruction from a (typically dense MVS) point cloud.

    Args:
        points_xyz: (P, 3) point positions.
        points_rgb: Optional (P, 3) point colors, either in [0, 1] or [0, 255].
        normals: Optional (P, 3) point normals. If not given, normals are
            estimated from the point cloud and consistently oriented.
        depth: Octree depth passed to Open3D's Poisson reconstruction --
            higher values capture more detail but are slower / more likely to
            overfit noisy points.
        density_quantile_threshold: Vertices whose Poisson "density" (roughly,
            how much point support they had) falls below this quantile are
            trimmed -- the standard Poisson cleanup step to remove
            hallucinated geometry in unobserved regions.

    Returns:
        An ``open3d.geometry.TriangleMesh``.
    """
    o3d = _require_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    if points_rgb is not None:
        rgb = np.asarray(points_rgb, dtype=np.float64)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile_threshold)
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh.compute_vertex_normals()
    return _clean_mesh(mesh)


def bake_texture(mesh, dataset, max_views: Optional[int] = None):
    """Bake per-vertex colors onto ``mesh`` from ``dataset``'s training images.

    For each mesh vertex, projects into every (or up to ``max_views``)
    training camera, discards occluded/out-of-frame projections via
    ray-casting against ``mesh`` itself, and blends the remaining observed
    pixel colors weighted by view-direction/vertex-normal alignment and
    inverse distance.

    This produces per-vertex colors, not a UV-unwrapped texture atlas (which
    would need a UV unwrapper such as ``xatlas``) -- a deliberate scope
    boundary: vertex colors are correct and sufficient for inspection/most
    downstream uses, and adding UV-atlas baking is left as future work.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh`` (e.g. from
            :func:`extract_mesh_tsdf` or :func:`extract_mesh_poisson`).
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).

    Returns:
        ``mesh``, with ``vertex_colors`` set in place (and returned for
        chaining).
    """
    o3d = _require_open3d()

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    num_vertices = vertices.shape[0]
    color_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    weight_accum = np.zeros((num_vertices,), dtype=np.float64)

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    for i in range(num_views):
        data = dataset[i]
        camtoworld = data["camtoworld"].numpy()
        K = data["K"].numpy()
        image = data["image"].numpy() / 255.0  # (H, W, 3) in [0, 1]
        height, width = image.shape[:2]
        cam_pos = camtoworld[:3, 3]
        viewmat = np.linalg.inv(camtoworld)

        Xc = (viewmat[:3, :3] @ vertices.T + viewmat[:3, 3:4]).T  # (V, 3)
        in_front = Xc[:, 2] > 1e-4
        uvw = (K @ Xc.T).T
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        candidates = np.nonzero(in_front & in_bounds)[0]
        if candidates.size == 0:
            continue

        dirs = vertices[candidates] - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=1)
        dirs_n = dirs / np.clip(dists, 1e-8, None)[:, None]
        rays = np.concatenate(
            [np.repeat(cam_pos[None, :], len(candidates), axis=0), dirs_n], axis=1
        ).astype(np.float32)
        hit_t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        # Keep only vertices whose nearest ray hit is (approximately)
        # themselves, i.e. not occluded by other geometry in this view.
        visible = np.abs(hit_t - dists) < (1e-2 * dists + 1e-3)
        candidates = candidates[visible]
        if candidates.size == 0:
            continue
        dirs_n = dirs_n[visible]
        dists = dists[visible]

        px = np.clip(uv[candidates, 0].astype(np.int64), 0, width - 1)
        py = np.clip(uv[candidates, 1].astype(np.int64), 0, height - 1)
        sampled = image[py, px]  # (K, 3)

        cos_weight = np.clip((vertex_normals[candidates] * -dirs_n).sum(-1), 0.0, 1.0)
        dist_weight = 1.0 / np.clip(dists, 1e-3, None)
        weight = cos_weight * dist_weight + 1e-6

        color_accum[candidates] += sampled * weight[:, None]
        weight_accum[candidates] += weight

    has_color = weight_accum > 0
    vertex_colors = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.zeros((num_vertices, 3))
    )
    vertex_colors[has_color] = (
        color_accum[has_color] / weight_accum[has_color, None]
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors, 0.0, 1.0))
    return mesh
