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
- :func:`simplify_mesh`: quadric decimation to a triangle budget, since either
  path tessellates to its own grid rather than to the scene's complexity.

Dressing the resulting surface -- texture atlases, normal maps, ambient
occlusion -- lives in :mod:`gsplat.photogrammetry.texturing`. Those names are
re-exported at the bottom of this file for backwards compatibility.

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
from ._open3d import _require_open3d


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


def _point_spacing(points: np.ndarray, k: int = 4, max_samples: int = 20000) -> float:
    """Mean distance from a point to its ``k`` nearest neighbours in the cloud.

    The cloud's own sampling scale, which is what makes a cloud-to-mesh
    distance readable: a fit of 0.01 scene units means nothing on its own, but
    measured against the spacing of the evidence it becomes a verdict. Same
    quantity :func:`gsplat.photogrammetry.metrics.point_cloud_stats` reports,
    computed here through open3d's own neighbour search so decimation does not
    pull ``scikit-learn`` into the ``mesh`` extra.

    Large clouds are subsampled to ``max_samples`` query points: this is a
    density estimate, and averaging over twenty thousand neighbourhoods is
    already far more than it needs.

    Raises:
        ValueError: If there are fewer than ``k + 1`` points, so no point has
            ``k`` neighbours to average over.
    """
    o3d = _require_open3d()

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")
    if points.shape[0] < k + 1:
        raise ValueError(
            f"Need at least {k + 1} points to measure {k}-nearest-neighbour "
            f"spacing, got {points.shape[0]}."
        )

    search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
    search.knn_index()
    queries = points
    if points.shape[0] > max_samples:
        stride = points.shape[0] // max_samples
        queries = points[::stride][:max_samples]
    # k + 1 because the nearest neighbour of a query drawn from the cloud is
    # itself, at distance zero.
    _indices, squared = search.knn_search(o3d.core.Tensor(queries), k + 1)
    distances = np.sqrt(np.maximum(squared.numpy(), 0.0))
    return float(distances[:, 1:].mean())


def simplify_mesh_to_error(
    mesh,
    points: np.ndarray,
    max_error: Optional[float] = None,
    error_over_spacing: Optional[float] = None,
    min_triangles: int = 4,
    max_probes: int = 16,
    boundary_weight: float = 1.0,
):
    """Decimate as far as a **fit target** allows, rather than to a triangle count.

    :func:`simplify_mesh` takes a triangle budget, which is the wrong question
    to have to answer: how many triangles a scene needs depends on the scene,
    and picking the number is guesswork that is only checked afterwards -- by
    measuring the cloud-to-mesh fit, which is the thing actually cared about.
    This inverts that. Give it the fit you are willing to accept and it finds
    the smallest mesh that still delivers it, by binary search over the
    triangle count, measuring
    :func:`gsplat.photogrammetry.metrics.point_to_mesh_distance` at each probe.

    The target is best given as ``error_over_spacing``: cloud-to-mesh distance
    measured **in units of the cloud's own k-NN spacing**, the same scale-free
    reading as the pipeline's ``mesh_fit_over_point_spacing``. At or below ~1
    the mesh tracks the cloud to within its sampling noise, and that means the
    same thing on a tabletop scan and on a city block.

    **The returned mesh is always one whose error was measured**, not one the
    search assumed was fine. Quadric decimation is only *roughly* monotone in
    the triangle count -- collapsing a different edge order can find a slightly
    better fit with fewer triangles -- so a binary search's final bracket is
    not itself a guarantee, and the decimator does not always land on the
    triangle count it was asked for. If no probe met the target, the input
    comes back unchanged with ``target_met`` False.

    Among the probes that *did* meet it, the one with the fewest triangles is
    returned. That part is about not handing back a needlessly large mesh
    rather than about correctness -- every candidate considered is feasible by
    measurement -- and on well-behaved input it picks the same mesh the last
    feasible probe would have.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        points: ``(P, 3)`` reference cloud -- the dense MVS or sparse SfM cloud
            the mesh was built from, the same one
            :func:`~gsplat.photogrammetry.metrics.point_to_mesh_distance` is
            measured against elsewhere.
        max_error: Fit target in scene units. Exactly one of this and
            ``error_over_spacing`` must be given.
        error_over_spacing: Fit target as a multiple of the cloud's own mean
            k-NN spacing (see :func:`_point_spacing`).
        min_triangles: Never decimate below this.
        max_probes: Cap on decimate-and-measure rounds. 16 resolves a
            million-triangle mesh to within ~15 triangles.
        boundary_weight: Passed to :func:`simplify_mesh`.

    Returns:
        ``(mesh, stats)``. ``stats`` reports ``triangles_before``/
        ``triangles_after``/``reduction``, ``error_before``/``error_after``,
        the resolved ``max_error`` (and ``point_spacing``/``error_over_spacing``
        when that route was used), ``num_probes``, ``target_met``, and
        ``probes`` -- every (triangles, error) pair measured, so a caller can
        see the curve rather than just the answer.

    Raises:
        ValueError: If neither or both targets are given, if the target is not
            positive, if ``mesh`` has no triangles, or if ``points`` is empty
            (there is nothing to measure the fit against).
    """
    from .metrics import point_to_mesh_distance

    _require_open3d()

    if (max_error is None) == (error_over_spacing is None):
        raise ValueError(
            "Give exactly one of max_error (scene units) or error_over_spacing "
            "(multiples of the cloud's own k-NN spacing)."
        )
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(
            "Cannot decimate to a fit target against an empty point cloud: "
            "there is nothing to measure the fit against. Pass the dense or "
            "sparse cloud the mesh was reconstructed from."
        )
    if len(mesh.triangles) == 0:
        raise ValueError("Cannot decimate a mesh with no triangles.")

    spacing = None
    if error_over_spacing is not None:
        if error_over_spacing <= 0:
            raise ValueError(
                f"error_over_spacing must be positive, got {error_over_spacing}."
            )
        spacing = _point_spacing(points)
        max_error = error_over_spacing * spacing
    elif max_error <= 0:
        raise ValueError(f"max_error must be positive, got {max_error}.")

    triangles_before = len(mesh.triangles)
    error_before = point_to_mesh_distance(points, mesh)["mean"]

    stats = {
        "triangles_before": int(triangles_before),
        "triangles_after": int(triangles_before),
        "reduction": 0.0,
        "error_before": float(error_before),
        "error_after": float(error_before),
        "max_error": float(max_error),
        "point_spacing": spacing,
        "error_over_spacing": error_over_spacing,
        "num_probes": 0,
        "target_met": bool(error_before <= max_error),
        "probes": [],
    }
    if error_before > max_error:
        # Decimation can only move the surface further from the cloud, so
        # there is no mesh below this one that meets the target. Say so
        # instead of returning a mesh that silently misses it.
        return mesh, stats
    if triangles_before <= min_triangles:
        return mesh, stats

    best_mesh, best_triangles = None, None
    low, high = min_triangles, triangles_before
    for _ in range(max_probes):
        if low > high:
            break
        target = (low + high) // 2
        if target >= triangles_before:
            break
        candidate = simplify_mesh(
            mesh, target_triangles=max(target, 1), boundary_weight=boundary_weight
        )
        if len(candidate.triangles) == 0:
            low = target + 1
            continue
        error = point_to_mesh_distance(points, candidate)["mean"]
        actual = len(candidate.triangles)
        stats["probes"].append({"triangles": int(actual), "error": float(error)})
        if error <= max_error:
            # Feasible. Keep it if it is the smallest that has worked, and
            # keep looking further down.
            if best_triangles is None or actual < best_triangles:
                best_mesh, best_triangles = candidate, actual
            high = min(target, actual) - 1
        else:
            low = target + 1
    stats["num_probes"] = len(stats["probes"])

    if best_mesh is None:
        return mesh, stats

    stats["triangles_after"] = int(best_triangles)
    stats["reduction"] = 1.0 - (best_triangles / triangles_before)
    stats["error_after"] = float(
        next(p["error"] for p in stats["probes"] if p["triangles"] == best_triangles)
    )
    stats["target_met"] = True
    return best_mesh, stats


# Texturing moved to gsplat.photogrammetry.texturing when this module outgrew
# holding both. Re-exported here so existing imports keep resolving -- the
# example CLIs and the test suite both import bakers from this path.
from .texturing import (  # noqa: E402,F401  (import placement is deliberate)
    _AtlasTexels,
    _bake_points_from_views,
    _fill_texture_holes,
    _grown_mask,
    _unwrap_and_rasterize,
    _vertex_tangents,
    _view_samples,
    bake_ambient_occlusion,
    bake_mesh_texture,
    bake_normal_map,
    bake_texture,
    bake_texture_atlas,
    bake_texture_atlas_view_selected,
    face_view_quality,
    level_seams,
    select_views_mrf,
)
