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
with occlusion-aware, view-angle-weighted blending across views;
:func:`bake_texture_atlas` bakes the same multi-view color signal into a
UV-unwrapped texture atlas instead, so the result carries detail beyond the
mesh's vertex density and loads with its texture in standard DCC tools and
game engines.

Requires the optional ``open3d`` dependency: ``pip install gsplat[mesh]``.
Only SH-color checkpoints (containing ``"sh0"``/``"shN"``) are supported --
appearance-embedding checkpoints (``"features"``/``"colors"``) are out of
scope, since per-image appearance variation doesn't map onto a single
canonical mesh texture.
"""

import warnings
from dataclasses import dataclass
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
            more train views) improves mesh coverage. If items include a
            ``"mask"`` (e.g. ``Dataset(..., mask_dir=...)``), pixels where
            it's ``0`` are excluded from TSDF fusion.
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

        depth_np = depth.cpu().numpy().astype(np.float32)
        if "mask" in data:
            # Zero out depth at excluded (e.g. transient-object) pixels --
            # Open3D's RGBDImage treats depth == 0 as "no data", so those
            # pixels contribute nothing to the fused mesh.
            depth_np = depth_np * data["mask"].cpu().numpy().astype(np.float32)

        views.append(
            {
                "color": (color.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                "depth": depth_np,
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
            width,
            height,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
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


def _bake_points_from_views(
    mesh,
    dataset,
    points: np.ndarray,
    normals: np.ndarray,
    max_views: Optional[int] = None,
    chunk_size: int = 1 << 20,
):
    """Accumulate occlusion-aware, view-weighted colors for surface points.

    Shared by :func:`bake_texture` (which bakes at mesh vertices) and
    :func:`bake_texture_atlas` (which bakes at texel positions), so both
    produce the same color signal and only differ in where they sample it.

    For each point, projects into every (or up to ``max_views``) training
    camera, discards out-of-frame projections and -- via ray casting against
    ``mesh`` itself -- occluded ones, and accumulates the remaining observed
    pixel colors weighted by view-direction/surface-normal alignment and
    inverse distance.

    Args:
        mesh: The ``open3d.geometry.TriangleMesh`` to ray-cast against for
            occlusion. ``points`` are expected to lie on its surface.
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        points: (P, 3) surface points to bake.
        normals: (P, 3) unit-length outward normals, one per point.
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).
        chunk_size: Max points ray-cast at once, bounding peak memory when
            baking the millions of texels a large atlas can contain.

    Returns:
        ``(color_accum, weight_accum)``: a (P, 3) sum of weighted colors and
        the (P,) sum of weights. Points with ``weight_accum == 0`` were never
        observed.
    """
    o3d = _require_open3d()

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    num_points = points.shape[0]
    color_accum = np.zeros((num_points, 3), dtype=np.float64)
    weight_accum = np.zeros((num_points,), dtype=np.float64)
    if num_points == 0:
        return color_accum, weight_accum

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    for i in range(num_views):
        data = dataset[i]
        camtoworld = data["camtoworld"].numpy()
        K = data["K"].numpy()
        image = data["image"].numpy() / 255.0  # (H, W, 3) in [0, 1]
        height, width = image.shape[:2]
        cam_pos = camtoworld[:3, 3]
        viewmat = np.linalg.inv(camtoworld)

        Xc = (viewmat[:3, :3] @ points.T + viewmat[:3, 3:4]).T  # (P, 3)
        in_front = Xc[:, 2] > 1e-4
        uvw = (K @ Xc.T).T
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        candidates = np.nonzero(in_front & in_bounds)[0]
        if candidates.size == 0:
            continue

        for start in range(0, candidates.size, chunk_size):
            chunk = candidates[start : start + chunk_size]
            dirs = points[chunk] - cam_pos[None, :]
            dists = np.linalg.norm(dirs, axis=1)
            dirs_n = dirs / np.clip(dists, 1e-8, None)[:, None]
            rays = np.concatenate(
                [np.repeat(cam_pos[None, :], len(chunk), axis=0), dirs_n], axis=1
            ).astype(np.float32)
            hit_t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            # Keep only points whose nearest ray hit is (approximately)
            # themselves, i.e. not occluded by other geometry in this view.
            visible = np.abs(hit_t - dists) < (1e-2 * dists + 1e-3)
            chunk = chunk[visible]
            if chunk.size == 0:
                continue
            dirs_n = dirs_n[visible]
            dists = dists[visible]

            px = np.clip(uv[chunk, 0].astype(np.int64), 0, width - 1)
            py = np.clip(uv[chunk, 1].astype(np.int64), 0, height - 1)
            sampled = image[py, px]  # (K, 3)

            cos_weight = np.clip((normals[chunk] * -dirs_n).sum(-1), 0.0, 1.0)
            dist_weight = 1.0 / np.clip(dists, 1e-3, None)
            weight = cos_weight * dist_weight + 1e-6

            # `chunk` indexes each point at most once per view, so a plain
            # in-place add is correct (and much faster than np.add.at).
            color_accum[chunk] += sampled * weight[:, None]
            weight_accum[chunk] += weight

    return color_accum, weight_accum


def bake_texture(mesh, dataset, max_views: Optional[int] = None):
    """Bake per-vertex colors onto ``mesh`` from ``dataset``'s training images.

    Colors each mesh vertex by the occlusion-aware, view-weighted blend
    described in :func:`_bake_points_from_views`.

    This produces per-vertex colors, whose effective resolution is the mesh's
    own vertex density. For image-resolution detail independent of tessellation
    -- and for a mesh that loads with its texture in standard DCC tools and
    game engines -- use :func:`bake_texture_atlas` instead.

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

    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    color_accum, weight_accum = _bake_points_from_views(
        mesh, dataset, vertices, vertex_normals, max_views=max_views
    )

    has_color = weight_accum > 0
    vertex_colors = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.zeros((vertices.shape[0], 3))
    )
    vertex_colors[has_color] = color_accum[has_color] / weight_accum[has_color, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors, 0.0, 1.0))
    return mesh


def _fill_texture_holes(texture: np.ndarray, filled: np.ndarray, iterations: int):
    """Pad baked texels outward into unfilled ones by nearest-neighbor growth.

    Two kinds of texel end up unfilled: those outside every UV island, and
    those inside an island that no camera ever observed. Renderers sample an
    atlas bilinearly (and build mipmaps from it), so both kinds bleed the fill
    color across every UV seam and into every unobserved patch. Growing the
    baked colors a few texels outward removes that artifact.

    Args:
        texture: (S, S, 3) float texture, unfilled texels at 0.
        filled: (S, S) bool mask of texels that carry a baked color.
        iterations: How many texels to grow outward.

    Returns:
        A new (S, S, 3) float texture with the padding applied.
    """
    texture = texture.copy()
    filled = filled.copy()
    for _ in range(max(iterations, 0)):
        if filled.all():
            break
        neighbor_sum = np.zeros_like(texture)
        neighbor_count = np.zeros(filled.shape, dtype=np.float64)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted_tex = np.roll(texture, shift, axis=axis)
            shifted_filled = np.roll(filled, shift, axis=axis)
            # np.roll wraps around; blank the wrapped-in edge so colors can't
            # leak from one side of the atlas to the other.
            edge = 0 if shift > 0 else -1
            if axis == 0:
                shifted_tex[edge, :] = 0.0
                shifted_filled[edge, :] = False
            else:
                shifted_tex[:, edge] = 0.0
                shifted_filled[:, edge] = False
            neighbor_sum += shifted_tex * shifted_filled[..., None]
            neighbor_count += shifted_filled
        grow = (~filled) & (neighbor_count > 0)
        if not grow.any():
            break
        texture[grow] = neighbor_sum[grow] / neighbor_count[grow][:, None]
        filled |= grow
    return texture


@dataclass
class _AtlasTexels:
    """The per-texel surface frame of a UV-unwrapped mesh.

    ``rows``/``cols`` index the texels the rasterizer covered; ``positions``,
    ``normals`` and (optionally) ``tangents`` are the interpolated surface
    frame at each of those texels, in the same order.
    """

    triangle_uvs: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    tangents: Optional[np.ndarray] = None


def _vertex_tangents(vertices, vertex_normals, triangles, corner_uvs):
    """Per-vertex tangents from the UV parameterization (Lengyel's method).

    Accumulates each triangle's UV-aligned tangent onto its vertices, then
    orthogonalizes against the vertex normal. Vertices on a UV seam belong to
    charts with different orientations, so they get a blended tangent -- the
    standard artifact of not splitting vertices per chart, confined to the
    seam texels the dilation pass already covers.
    """
    tangents = np.zeros_like(vertices)
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    duv1 = corner_uvs[:, 1] - corner_uvs[:, 0]
    duv2 = corner_uvs[:, 2] - corner_uvs[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    # A degenerate UV triangle carries no direction; skip it rather than
    # dividing by ~0 and poisoning its vertices' accumulated tangent.
    usable = np.abs(det) > 1e-12
    per_face = np.zeros_like(edge1)
    per_face[usable] = (
        edge1[usable] * duv2[usable, 1:2] - edge2[usable] * duv1[usable, 1:2]
    ) / det[usable, None]
    for corner in range(3):
        np.add.at(tangents, triangles[:, corner], per_face)

    tangents -= vertex_normals * (vertex_normals * tangents).sum(-1, keepdims=True)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    # A vertex whose tangents cancelled out gets an arbitrary perpendicular
    # rather than a zero vector, so the tangent frame stays well-defined.
    degenerate = lengths[:, 0] < 1e-12
    if degenerate.any():
        fallback = np.tile(np.array([1.0, 0.0, 0.0]), (int(degenerate.sum()), 1))
        parallel = np.abs(vertex_normals[degenerate][:, 0]) > 0.9
        fallback[parallel] = np.array([0.0, 1.0, 0.0])
        normals_d = vertex_normals[degenerate]
        fallback -= normals_d * (normals_d * fallback).sum(-1, keepdims=True)
        tangents[degenerate] = fallback
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.clip(lengths, 1e-12, None)


def _unwrap_and_rasterize(
    mesh,
    texture_size: int,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    with_tangents: bool = False,
    reuse_uvs: bool = True,
) -> "_AtlasTexels":
    """UV-unwrap ``mesh`` and recover each texel's surface frame.

    Shared by :func:`bake_texture_atlas` and :func:`bake_normal_map`, which
    differ only in what they bake into the atlas, never in how the atlas is
    built or which texels it covers.

    **open3d's ``compute_uvatlas`` is not deterministic** -- unwrapping the
    same mesh twice yields different UV layouts. So when ``mesh`` already
    carries ``triangle_uvs``, they are reused rather than recomputed
    (``reuse_uvs=False`` forces a fresh unwrap). Without that, baking an
    albedo atlas and then a normal map for one mesh would produce two maps
    with *different* UV layouts, and the second would be applied through the
    first's coordinates -- a silently broken asset.

    It also requires a manifold mesh and **segfaults** rather than raising on
    non-manifold input, so that is checked up front: a crash in the middle of
    a long pipeline run would otherwise take the whole run down.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``texture_size`` is not
            positive, or ``mesh`` is non-manifold.
    """
    o3d = _require_open3d()

    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}.")
    if len(mesh.triangles) == 0:
        raise ValueError("Cannot UV-unwrap a mesh with no triangles.")
    # Boundary edges are fine (and normal for TSDF output) -- only edges shared
    # by more than two triangles, or vertices joining disconnected fans, break
    # the unwrapper.
    if not mesh.is_edge_manifold(allow_boundary_edges=True):
        raise ValueError(
            "Cannot UV-unwrap a mesh with non-manifold edges (open3d's "
            "compute_uvatlas requires a manifold mesh and crashes on this "
            "input). Run `mesh.remove_non_manifold_edges()` first, or use "
            "bake_texture() for per-vertex colors instead."
        )
    if not mesh.is_vertex_manifold():
        raise ValueError(
            "Cannot UV-unwrap a mesh with non-manifold vertices (open3d's "
            "compute_uvatlas requires a manifold mesh and crashes on this "
            "input). Use bake_texture() for per-vertex colors instead."
        )

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    num_triangles = len(mesh.triangles)
    existing_uvs = np.asarray(mesh.triangle_uvs)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    if reuse_uvs and existing_uvs.shape[0] == 3 * num_triangles:
        t_mesh.triangle["texture_uvs"] = o3d.core.Tensor(
            existing_uvs.reshape(num_triangles, 3, 2).astype(np.float32)
        )
    else:
        t_mesh.compute_uvatlas(
            size=texture_size if unwrap_size is None else unwrap_size,
            gutter=gutter,
            max_stretch=max_stretch,
        )

    # UV coordinates are per triangle corner, so unwrapping leaves the mesh's
    # vertices and triangles untouched -- the atlas can be attached straight
    # back onto the input mesh by the caller.
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    corner_uvs = t_mesh.triangle["texture_uvs"].numpy()  # (F, 3, 2)

    attrs = {"_bake_xyz", "_bake_normal", "_bake_coverage"}
    t_mesh.vertex["_bake_xyz"] = o3d.core.Tensor(vertices.astype(np.float32))
    t_mesh.vertex["_bake_normal"] = o3d.core.Tensor(vertex_normals.astype(np.float32))
    # A constant-1 attribute bakes to exactly the texels the rasterizer covered,
    # which is what marks a texel as being on the surface. Testing the baked
    # position against 0 instead would misclassify a surface passing through
    # the world origin.
    t_mesh.vertex["_bake_coverage"] = o3d.core.Tensor(
        np.ones((vertices.shape[0], 1), dtype=np.float32)
    )
    if with_tangents:
        tangents = _vertex_tangents(
            vertices, vertex_normals, np.asarray(mesh.triangles), corner_uvs
        )
        t_mesh.vertex["_bake_tangent"] = o3d.core.Tensor(tangents.astype(np.float32))
        attrs.add("_bake_tangent")

    baked = t_mesh.bake_vertex_attr_textures(
        texture_size, attrs, margin=margin, fill=0.0, update_material=False
    )
    covered = baked["_bake_coverage"].numpy()[..., 0] > 0.5
    rows, cols = np.nonzero(covered)

    positions = baked["_bake_xyz"].numpy()[rows, cols].astype(np.float64)
    normals = baked["_bake_normal"].numpy()[rows, cols].astype(np.float64)
    # Barycentric interpolation of unit vertex normals doesn't preserve length;
    # renormalize so downstream dot products stay comparable across texels.
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)

    texel_tangents = None
    if with_tangents:
        texel_tangents = baked["_bake_tangent"].numpy()[rows, cols].astype(np.float64)
        # Re-orthogonalize after interpolation, for the same reason.
        texel_tangents -= normals * (normals * texel_tangents).sum(-1, keepdims=True)
        texel_tangents /= np.clip(
            np.linalg.norm(texel_tangents, axis=1, keepdims=True), 1e-8, None
        )

    return _AtlasTexels(
        triangle_uvs=corner_uvs.reshape(-1, 2).astype(np.float64),
        rows=rows,
        cols=cols,
        positions=positions,
        normals=normals,
        tangents=texel_tangents,
    )


def bake_texture_atlas(
    mesh,
    dataset,
    texture_size: int = 2048,
    max_views: Optional[int] = None,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
):
    """Bake a UV-unwrapped texture atlas onto ``mesh`` from the training images.

    Unlike :func:`bake_texture`, whose per-vertex colors can only carry as
    much detail as the mesh is tessellated for, this UV-unwraps ``mesh`` and
    bakes one color per *texel*, so texture detail is independent of vertex
    density -- and the result is what standard DCC tools and game engines
    expect a textured mesh to look like.

    The pipeline is: UV-unwrap via open3d's ``compute_uvatlas``; rasterize the
    mesh's per-vertex positions and normals into the atlas via
    ``bake_vertex_attr_textures`` to recover each texel's 3D surface point;
    color those points with the same occlusion-aware, view-weighted blend
    :func:`bake_texture` uses (see :func:`_bake_points_from_views`); then pad
    the result outward across UV seams (see :func:`_fill_texture_holes`).

    The returned mesh carries ``triangle_uvs`` and ``textures``, so writing it
    with ``open3d.io.write_triangle_mesh("mesh.obj", mesh)`` emits the ``.obj``,
    its ``.mtl``, and the texture ``.png`` together. Note that ``.ply`` cannot
    carry a UV atlas -- write ``.obj`` on this path.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``. Must be manifold (see
            below).
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        texture_size: Width/height of the (square) texture, in texels.
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).
        unwrap_size: Texture size assumed while unwrapping, which sets the
            scale ``gutter`` is measured against. Defaults to ``texture_size``.
        gutter: Space around each UV island, in texels, passed to
            ``compute_uvatlas``.
        margin: Extra texels rasterized around each UV island by
            ``bake_vertex_attr_textures``.
        max_stretch: Per-chart stretch tolerance for ``compute_uvatlas``.
            Lower values cut the mesh into more, less distorted charts.
        dilation: How many texels to grow baked colors outward across seams
            and unobserved patches.

    Returns:
        ``(mesh, texture)``: ``mesh`` with ``triangle_uvs``/``textures`` set in
        place, and the (``texture_size``, ``texture_size``, 3) ``uint8`` texture
        as a numpy array.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``texture_size`` is not
            positive, or ``mesh`` is non-manifold. open3d's ``compute_uvatlas``
            requires a manifold mesh and *segfaults* rather than raising on
            non-manifold input, so this is checked up front -- a crash in the
            middle of a long pipeline run would otherwise take the whole run
            down. :func:`extract_mesh_tsdf`/:func:`extract_mesh_poisson` output
            has already been through ``remove_non_manifold_edges``; fall back
            to :func:`bake_texture` for a mesh that still fails the check.
    """
    o3d = _require_open3d()

    atlas = _unwrap_and_rasterize(
        mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
    )
    rows, cols = atlas.rows, atlas.cols

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if rows.size > 0:
        color_accum, weight_accum = _bake_points_from_views(
            mesh, dataset, atlas.positions, atlas.normals, max_views=max_views
        )
        observed = weight_accum > 0
        texture[rows[observed], cols[observed]] = (
            color_accum[observed] / weight_accum[observed, None]
        )
        filled[rows[observed], cols[observed]] = True

    texture = _fill_texture_holes(texture, filled, dilation)
    texture = (np.clip(texture, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    mesh.textures = [o3d.geometry.Image(texture)]
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(mesh.triangles), dtype=np.int32)
    )
    return mesh, texture


def bake_mesh_texture(
    mesh,
    dataset,
    mode: str = "vertex",
    texture_size: int = 2048,
    max_views: Optional[int] = None,
    allow_atlas_fallback: bool = True,
):
    """Bake texture onto ``mesh`` in either supported form.

    The one entry point the CLIs use, so the choice between per-vertex colors
    and a UV atlas -- and what happens when a mesh can't be unwrapped -- is
    decided in one place rather than in each script.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object (see
            :func:`bake_texture`).
        mode: ``"vertex"`` for :func:`bake_texture`, ``"atlas"`` for
            :func:`bake_texture_atlas`.
        texture_size: Atlas width/height in texels (``"atlas"`` mode only).
        max_views: If given, only the first ``max_views`` dataset images are
            used.
        allow_atlas_fallback: In ``"atlas"`` mode, whether a mesh that can't be
            UV-unwrapped falls back to per-vertex colors (with a warning)
            instead of raising. Defaults to ``True``: this runs at the end of a
            long training run or pipeline, where losing the mesh entirely is
            the worse outcome.

    Returns:
        ``(mesh, texture)``. ``texture`` is the ``uint8`` atlas in ``"atlas"``
        mode, and ``None`` for per-vertex colors -- which is also what a
        fallback returns, so callers can use it to decide whether to write
        ``.obj`` (a UV atlas needs one) or ``.ply``.
    """
    if mode == "vertex":
        return bake_texture(mesh, dataset, max_views=max_views), None
    if mode != "atlas":
        raise ValueError(
            f"Unknown texture mode {mode!r}, expected 'vertex' or 'atlas'."
        )

    try:
        return bake_texture_atlas(
            mesh, dataset, texture_size=texture_size, max_views=max_views
        )
    except ValueError as e:
        if not allow_atlas_fallback:
            raise
        warnings.warn(
            f"UV-atlas texture baking failed ({e}); falling back to per-vertex "
            "colors. The mesh is still written, but without a texture atlas.",
            RuntimeWarning,
            stacklevel=2,
        )
        return bake_texture(mesh, dataset, max_views=max_views), None


def simplify_mesh(
    mesh,
    target_triangles: int,
    maximum_error: float = float("inf"),
    boundary_weight: float = 1.0,
):
    """Decimate ``mesh`` to roughly ``target_triangles`` via quadric error metrics.

    TSDF and Poisson extraction produce meshes tessellated to the voxel grid,
    not to the scene's actual geometric complexity -- routinely millions of
    triangles for a scene a few hundred thousand would describe. Garland &
    Heckbert quadric-error decimation collapses edges cheapest-first, so flat
    regions lose triangles and detailed ones keep them.

    The detail this removes is not lost if you follow it with
    :func:`bake_normal_map`, which records the *original* mesh's normals onto
    the decimated one's UV atlas -- the standard photogrammetry delivery path
    (dense scan -> low-poly mesh + normal map).

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        target_triangles: Triangle budget. Decimation stops here, or earlier if
            ``maximum_error`` is reached first, so the result can have more
            triangles than requested only when the input already had fewer.
        maximum_error: Stop collapsing once the cheapest remaining edge exceeds
            this quadric error, whatever the triangle count.
        boundary_weight: How strongly to penalize collapsing boundary edges.
            Higher keeps open borders (a scanned scene's outer edge) sharper.

    Returns:
        A new decimated mesh, cleaned of degenerate/duplicate geometry and
        small floating components. The input is not modified.

    Raises:
        ValueError: If ``target_triangles`` is not positive.
    """
    _require_open3d()

    if target_triangles <= 0:
        raise ValueError(f"target_triangles must be positive, got {target_triangles}.")

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_triangles),
        maximum_error=maximum_error,
        boundary_weight=boundary_weight,
    )
    simplified = _clean_mesh(simplified)
    simplified.compute_vertex_normals()
    return simplified


def bake_normal_map(
    high_mesh,
    low_mesh,
    texture_size: int = 2048,
    space: str = "tangent",
    cage: Optional[float] = None,
    max_distance: Optional[float] = None,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
):
    """Bake ``high_mesh``'s surface normals onto ``low_mesh``'s UV atlas.

    The other half of the standard photogrammetry delivery path: after
    :func:`simplify_mesh` cuts a dense extraction down to a workable triangle
    budget, this records the detail that was removed as a normal map, so the
    low-poly mesh *shades* like the dense one. It is what makes decimation
    nearly free visually, and it is why production scans ship as a light mesh
    plus texture maps rather than as raw multi-million-triangle geometry.

    For each texel of ``low_mesh``'s atlas, a ray is cast from just outside the
    low surface, inward along its own normal, onto ``high_mesh``; the high
    mesh's normal at the hit point (barycentrically interpolated) is what gets
    stored. Texels whose ray misses -- and texels in the atlas margin, which
    lie off the surface by construction -- fall back to the low mesh's own
    normal, i.e. "no change from the base geometry".

    Args:
        high_mesh: The dense mesh to take detail from.
        low_mesh: The decimated mesh to bake onto. Must be manifold; it gains
            ``triangle_uvs`` in place, matching the returned map.
        texture_size: Width/height of the (square) normal map, in texels.
        space: ``"tangent"`` stores normals relative to each texel's surface
            frame -- the portable choice, and what engines expect, since it
            stays valid if the mesh is transformed. ``"object"`` stores them
            in mesh space: simpler and immune to UV-seam tangent artifacts,
            and a common choice for static scanned assets.
        cage: How far outside the low surface each ray starts, in scene units.
            It must clear the gap between the two meshes; defaults to 2% of
            ``low_mesh``'s bounding-box diagonal.
        max_distance: Reject hits farther than this from the ray origin.
            Defaults to twice ``cage``, so a ray that punches through a gap
            and hits unrelated geometry behind it is discarded.
        unwrap_size, gutter, margin, max_stretch, dilation: As in
            :func:`bake_texture_atlas`.

    Returns:
        ``(low_mesh, normal_map, stats)``:

        - ``low_mesh`` with ``triangle_uvs`` set, matching the returned map.
        - ``normal_map``, a (``texture_size``, ``texture_size``, 3) ``uint8``
          array encoded the usual way (``0.5 + 0.5 * n``, so a flat texel is
          ~(128, 128, 255) in tangent space).
        - ``stats``: ``num_texels``, ``num_hits``, ``hit_fraction`` and the
          ``cage``/``max_distance`` actually used. A low ``hit_fraction`` means
          most texels fell back to the base normal and the map is doing
          nothing -- almost always a ``cage`` too small to span the gap
          between the two meshes. Worth checking before shipping the asset,
          which is why it is returned rather than left to be eyeballed.

    Raises:
        ValueError: If ``space`` is not ``"tangent"``/``"object"``, or
            ``low_mesh`` cannot be unwrapped (see
            :func:`bake_texture_atlas`), or ``high_mesh`` has no triangles.
    """
    o3d = _require_open3d()

    if space not in ("tangent", "object"):
        raise ValueError(
            f"Unknown normal-map space {space!r}, expected 'tangent' or 'object'."
        )
    if len(high_mesh.triangles) == 0:
        raise ValueError("Cannot bake normals from a high mesh with no triangles.")

    if not high_mesh.has_vertex_normals():
        high_mesh.compute_vertex_normals()

    atlas = _unwrap_and_rasterize(
        low_mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
        with_tangents=(space == "tangent"),
    )

    if cage is None:
        extent = np.asarray(low_mesh.get_max_bound()) - np.asarray(
            low_mesh.get_min_bound()
        )
        cage = 0.02 * float(np.linalg.norm(extent))
        cage = max(cage, 1e-6)
    if max_distance is None:
        max_distance = 2.0 * cage

    positions, normals = atlas.positions, atlas.normals
    # Default to the low mesh's own normal: for a texel with no hit that is
    # exactly right -- "this surface is already as detailed as the base mesh".
    detailed = normals.copy()
    num_hits = 0

    if positions.shape[0] > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(high_mesh))
        origins = positions + normals * cage
        rays = np.concatenate([origins, -normals], axis=1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays))
        hit_t = result["t_hit"].numpy()
        primitives = result["primitive_ids"].numpy()
        bary = result["primitive_uvs"].numpy()
        hit = np.isfinite(hit_t) & (hit_t <= max_distance)

        num_hits = int(hit.sum())
        if hit.any():
            high_triangles = np.asarray(high_mesh.triangles)[primitives[hit]]
            corner_normals = np.asarray(high_mesh.vertex_normals)[high_triangles]
            u, v = bary[hit, 0], bary[hit, 1]
            # open3d's primitive_uvs weight the corners (1-u-v, u, v) --
            # verified by reconstructing hit positions from them.
            weights = np.stack([1.0 - u - v, u, v], axis=-1)[..., None]
            detailed[hit] = (corner_normals * weights).sum(axis=1)
            detailed /= np.clip(
                np.linalg.norm(detailed, axis=1, keepdims=True), 1e-12, None
            )

    if space == "tangent":
        tangent = atlas.tangents
        bitangent = np.cross(normals, tangent)
        encoded = np.stack(
            [
                (detailed * tangent).sum(-1),
                (detailed * bitangent).sum(-1),
                (detailed * normals).sum(-1),
            ],
            axis=-1,
        )
    else:
        encoded = detailed

    normal_map = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if atlas.rows.size > 0:
        normal_map[atlas.rows, atlas.cols] = 0.5 + 0.5 * encoded
        filled[atlas.rows, atlas.cols] = True
    # Unfilled texels would otherwise decode to a zero-length normal; the
    # dilation pass spreads real values across seams the same way the color
    # atlas does.
    normal_map = _fill_texture_holes(normal_map, filled, dilation)
    if not filled.all():
        # Anything still untouched gets "flat": +Z in tangent space, and the
        # neutral 0.5 grey in object space.
        neutral = np.array([0.5, 0.5, 1.0]) if space == "tangent" else np.full(3, 0.5)
        normal_map[~_grown_mask(filled, dilation)] = neutral
    normal_map = (np.clip(normal_map, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    low_mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    num_texels = int(positions.shape[0])
    stats = {
        "num_texels": num_texels,
        "num_hits": num_hits,
        "hit_fraction": float(num_hits / num_texels) if num_texels else 0.0,
        "cage": float(cage),
        "max_distance": float(max_distance),
        "space": space,
    }
    return low_mesh, normal_map, stats


def _grown_mask(filled: np.ndarray, iterations: int) -> np.ndarray:
    """Which texels :func:`_fill_texture_holes` reaches in ``iterations`` steps."""
    grown = filled.copy()
    for _ in range(max(iterations, 0)):
        if grown.all():
            break
        neighbors = np.zeros_like(grown)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted = np.roll(grown, shift, axis=axis)
            edge = 0 if shift > 0 else -1
            if axis == 0:
                shifted[edge, :] = False
            else:
                shifted[:, edge] = False
            neighbors |= shifted
        grown |= neighbors
    return grown


def bake_ambient_occlusion(
    mesh,
    occluder_mesh=None,
    texture_size: int = 2048,
    num_samples: int = 64,
    max_distance: Optional[float] = None,
    cage: Optional[float] = None,
    seed: int = 0,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
    max_rays_per_batch: int = 1 << 22,
):
    """Bake an ambient-occlusion map onto ``mesh``'s UV atlas.

    The third map of the standard photogrammetry set, after albedo
    (:func:`bake_texture_atlas`) and normals (:func:`bake_normal_map`). AO
    records how much of the sky each point of the surface can actually see, so
    creases, contact points and cavities darken -- the cue that stops a scanned
    asset reading as flat under ambient light, and the one thing neither of the
    other two maps carries.

    Each texel casts ``num_samples`` rays over the cosine-weighted hemisphere
    about its normal (Malley's method: a uniform disk projected up, which is
    exactly the distribution the occlusion integral is weighted by, so a plain
    mean of the results is an unbiased estimate). The value stored is the
    fraction that escaped -- 1.0 fully open, 0.0 fully enclosed.

    Args:
        mesh: The ``open3d.geometry.TriangleMesh`` to bake onto. Must be
            manifold; it gains ``triangle_uvs`` in place, reusing any it
            already carries so this map lines up with the others.
        occluder_mesh: What the rays are tested against. Defaults to ``mesh``
            itself (self-occlusion). Pass the pre-decimation mesh to capture
            detail the decimated one no longer has, exactly as
            :func:`bake_normal_map` does.
        texture_size: Width/height of the (square) map, in texels.
        num_samples: Rays per texel. The estimate is a Monte-Carlo average, so
            its noise falls as ``1/sqrt(num_samples)``; 64 is a reasonable
            preview and a few hundred is smooth.
        max_distance: Occluders farther than this are ignored, which is what
            keeps AO a *local* contact cue rather than a global one. Defaults
            to half the mesh's bounding-box diagonal.
        cage: How far along the normal each ray starts, in scene units. When
            ``occluder_mesh`` is a *different* mesh this must clear the gap
            between the two surfaces, and defaults to 2% of the bounding-box
            diagonal to do so. Decimation cuts corners, so most of a
            simplified mesh's surface lies *inside* the dense one it came from
            -- measured at 80% of texels on a decimated bumpy sphere -- and a
            ray starting under the occluder hits it immediately, baking a
            uniformly dark map that looks plausible and is entirely wrong. For
            self-occlusion the default is a hair off the surface, just enough
            to avoid re-hitting the originating triangle.
        seed: Seeds the hemisphere sampling. Re-baking the *same* mesh gives
            the same map, since it then carries the UVs from the first bake;
            two separately unwrapped copies of one mesh will differ, because
            open3d's unwrapper is not deterministic (see
            :func:`_unwrap_and_rasterize`).
        unwrap_size, gutter, margin, max_stretch, dilation: As in
            :func:`bake_texture_atlas`.
        max_rays_per_batch: Cap on rays cast at once. A large atlas times a
            high sample count is easily hundreds of millions of rays, so they
            are cast in batches to bound peak memory.

    Returns:
        ``(mesh, ao_map, stats)``: the mesh with ``triangle_uvs`` set; the
        (``texture_size``, ``texture_size``, 3) ``uint8`` grayscale map; and
        ``stats`` with ``num_texels``, ``num_samples``, ``mean_ao``,
        ``min_ao`` and the ``max_distance`` used. A ``mean_ao`` of essentially
        1.0 means nothing occluded anything -- expected for a convex shape,
        and a sign the distance is too small for anything else.

    Raises:
        ValueError: If ``num_samples`` is not positive, or ``mesh`` cannot be
            unwrapped (see :func:`bake_texture_atlas`).
    """
    o3d = _require_open3d()

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    occluder = mesh if occluder_mesh is None else occluder_mesh
    if len(occluder.triangles) == 0:
        raise ValueError(
            "Cannot bake ambient occlusion against a mesh with no triangles."
        )

    atlas = _unwrap_and_rasterize(
        mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
        with_tangents=True,
    )

    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    diagonal = float(np.linalg.norm(extent))
    if max_distance is None:
        max_distance = 0.5 * diagonal
    if cage is None:
        # Against a different mesh the origin has to clear the gap between the
        # two surfaces; against itself it only has to clear its own triangle.
        cage = (0.02 if occluder_mesh is not None else 1e-4) * diagonal
    cage = max(cage, 1e-9)

    positions, normals = atlas.positions, atlas.normals
    tangents = atlas.tangents
    bitangents = np.cross(normals, tangents)
    num_texels = int(positions.shape[0])
    openness = np.ones(num_texels, dtype=np.float64)

    if num_texels > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occluder))
        rng = np.random.default_rng(seed)
        origins_all = positions + normals * cage
        batch = max(1, max_rays_per_batch // num_samples)
        for start in range(0, num_texels, batch):
            stop = min(start + batch, num_texels)
            count = stop - start
            # Malley's method: a uniformly sampled disk lifted onto the
            # hemisphere is cosine-distributed about the normal.
            r1 = rng.random((count, num_samples))
            r2 = rng.random((count, num_samples))
            radius = np.sqrt(r1)
            angle = 2.0 * np.pi * r2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.sqrt(np.clip(1.0 - r1, 0.0, None))
            directions = (
                x[..., None] * tangents[start:stop, None, :]
                + y[..., None] * bitangents[start:stop, None, :]
                + z[..., None] * normals[start:stop, None, :]
            )
            origins = np.repeat(origins_all[start:stop, None, :], num_samples, axis=1)
            rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6)
            hit_t = (
                scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"]
                .numpy()
                .reshape(count, num_samples)
            )
            occluded = np.isfinite(hit_t) & (hit_t < max_distance)
            openness[start:stop] = 1.0 - occluded.mean(axis=1)

    ao_map = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if num_texels > 0:
        ao_map[atlas.rows, atlas.cols] = openness[:, None]
        filled[atlas.rows, atlas.cols] = True
    ao_map = _fill_texture_holes(ao_map, filled, dilation)
    # Texels the dilation never reaches are outside every chart; leave them
    # fully open rather than black, so a stray sample can't darken the asset.
    ao_map[~_grown_mask(filled, dilation)] = 1.0
    ao_map = (np.clip(ao_map, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    stats = {
        "num_texels": num_texels,
        "num_samples": int(num_samples),
        "mean_ao": float(openness.mean()) if num_texels else 1.0,
        "min_ao": float(openness.min()) if num_texels else 1.0,
        "max_distance": float(max_distance),
        "cage": float(cage),
    }
    return mesh, ao_map, stats
