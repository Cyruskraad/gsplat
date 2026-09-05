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
    voxel_size: Optional[float] = None,
    sdf_trunc: Optional[float] = None,
    depth_trunc: Optional[float] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    device: str = "cuda",
    stats_out: Optional[dict] = None,
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
        voxel_size: TSDF voxel size, in scene units. ``None`` (the default)
            derives it from the depth actually being fused, rather than
            assuming a scene scale -- see
            :func:`derive_reconstruction_parameters`. The derivation happens in
            :func:`_tsdf_fuse`, on the pure-open3d side of the seam, so it is
            testable without a GPU.
        sdf_trunc: TSDF truncation distance, in scene units. ``None`` derives
            it as four voxels.
        depth_trunc: Maximum depth (scene units) to integrate; farther pixels
            are ignored (background/sky). ``None`` derives it from the scene's
            own extent.
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
        views,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        stats_out=stats_out,
    )


def _backprojected_cloud(views, max_points: int = 20000) -> np.ndarray:
    """The depth being fused, as a world-space point cloud.

    This is the evidence the reconstruction actually has, so it is what its
    resolution should be derived from -- and unlike the splats or the MVS
    cloud it is available *here*, on the pure-open3d side of the seam, which is
    what makes the derivation testable without a GPU.

    Pixels are strided rather than all back-projected: this feeds a k-nearest
    neighbour density estimate and a bounding box, and twenty thousand points
    settle both.
    """
    per_view = max(1, max_points // max(len(views), 1))
    clouds = []
    for view in views:
        depth = np.asarray(view["depth"], dtype=np.float64)
        K = np.asarray(view["K"], dtype=np.float64)
        extrinsic = np.asarray(view["extrinsic"], dtype=np.float64)
        rows, cols = np.nonzero(depth > 0)
        if rows.size == 0:
            continue
        if rows.size > per_view:
            stride = rows.size // per_view
            rows, cols = rows[::stride][:per_view], cols[::stride][:per_view]
        z = depth[rows, cols]
        x = (cols + 0.5 - K[0, 2]) * z / K[0, 0]
        y = (rows + 0.5 - K[1, 2]) * z / K[1, 1]
        camera = np.stack([x, y, z], axis=1)
        camtoworld = np.linalg.inv(extrinsic)
        clouds.append(camera @ camtoworld[:3, :3].T + camtoworld[:3, 3])
    if not clouds:
        return np.zeros((0, 3))
    return np.concatenate(clouds, axis=0)


def _tsdf_fuse(
    views,
    voxel_size: Optional[float] = None,
    sdf_trunc: Optional[float] = None,
    depth_trunc: Optional[float] = None,
    stats_out: Optional[dict] = None,
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
        voxel_size: TSDF voxel size, in scene units. ``None`` derives it from
            the depth being fused -- see
            :func:`derive_reconstruction_parameters`.
        sdf_trunc: TSDF truncation distance, in scene units. ``None`` derives
            it as four voxels.
        depth_trunc: Maximum depth (scene units) to integrate. ``None`` derives
            it from the scene's extent.
        stats_out: If given, a dict updated in place with whatever was derived,
            so a caller can report the numbers it did not have to choose.

    Returns:
        An ``open3d.geometry.TriangleMesh``, cleaned of degenerate geometry
        and small floating components.
    """
    o3d = _require_open3d()

    if voxel_size is None or sdf_trunc is None or depth_trunc is None:
        cloud = _backprojected_cloud(views)
        if len(cloud) < 5:
            raise ValueError(
                "Cannot derive TSDF parameters: the views carry fewer than "
                "five depth samples between them, so there is no scale to "
                "measure. Pass voxel_size/sdf_trunc/depth_trunc explicitly, or "
                "check that the depth maps are not empty."
            )
        derived = derive_reconstruction_parameters(cloud)
        if stats_out is not None:
            stats_out.update(derived)
            stats_out["derived"] = [
                name
                for name, value in (
                    ("voxel_size", voxel_size),
                    ("sdf_trunc", sdf_trunc),
                    ("depth_trunc", depth_trunc),
                )
                if value is None
            ]
        voxel_size = derived["voxel_size"] if voxel_size is None else voxel_size
        sdf_trunc = derived["sdf_trunc"] if sdf_trunc is None else sdf_trunc
        depth_trunc = derived["depth_trunc"] if depth_trunc is None else depth_trunc

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
    normal_radius: Optional[float] = None,
    normal_max_nn: int = 30,
    stats_out: Optional[dict] = None,
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
        normal_radius: Neighbourhood radius for normal estimation, in scene
            units. ``None`` (the default) derives it as three point spacings,
            which is the radius at which a disc holds about the
            ``normal_max_nn`` neighbours a stable plane fit wants -- see
            :func:`derive_reconstruction_parameters`. The old fixed ``0.1``
            was a scene-unit constant on a cloud of unknown scale: on a dense
            capture it swept in thousands of points and over-smoothed every
            normal, and on a sparse one it found none at all.
        normal_max_nn: Neighbour cap for normal estimation.
        stats_out: If given, a dict updated in place with what was derived.

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
        derived_names = []
        if normal_radius is None:
            derived = derive_reconstruction_parameters(
                points_xyz, normal_max_nn=normal_max_nn
            )
            normal_radius = derived["normal_radius"]
            derived_names = ["normal_radius"]
            if stats_out is not None:
                stats_out.update(derived)
        if stats_out is not None:
            # Reported from the value about to be *used*, not from the
            # derivation that suggested it. Writing the derived number instead
            # lets a call site drift from what it reports -- a mutation that
            # hardcoded 0.1 here kept every stats assertion green.
            stats_out["normal_radius"] = float(normal_radius)
            stats_out["derived"] = derived_names
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
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


def _geometry_only_copy(mesh):
    """A geometry-only copy, without vertex colours or UVs.

    ``remove_triangles_by_mask`` mutates in place, so the caller's mesh has to
    be left alone; and re-indexing vertices would silently invalidate any
    ``triangle_uvs`` the mesh carried, so those are dropped rather than
    quietly corrupted.
    """
    o3d = _require_open3d()

    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy()),
        o3d.utility.Vector3iVector(np.asarray(mesh.triangles).copy()),
    )


def cull_unobserved_faces(
    mesh,
    dataset,
    max_views: Optional[int] = None,
    min_views: int = 1,
    clean: bool = True,
):
    """Remove faces no camera in ``dataset`` ever saw.

    TSDF fusion returns a *closed* surface. That is what makes it watertight
    and easy to work with, and it also means it invents geometry: the
    underside of anything resting on the ground, the back of an object the
    capture only circled halfway, the inner shell of a volume sealed off from
    every camera. None of it was observed, so none of it can be textured --
    those faces end up carrying the seam-dilation fill colour -- and all of it
    is paid for in triangles, atlas area and file size.

    Visibility comes from
    :func:`gsplat.photogrammetry.texturing.face_visibility`, the same
    projection-plus-occlusion test every bake uses. **Not** from
    :func:`~gsplat.photogrammetry.texturing.face_view_quality` being zero:
    quality is gradient energy, so a perfectly visible face on a flat
    untextured surface also scores ~0, and culling on that would delete
    observed geometry.

    Run this **before** decimating and texturing: decimation should spend its
    triangle budget on surface that will actually be seen, and the UV atlas
    should not reserve area for faces that will never carry a colour. It is
    also why this drops any existing vertex colours or UVs rather than trying
    to carry them across the re-indexing -- at that point in the pipeline
    there are none yet.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        max_views: If given, only the first ``max_views`` images are consulted.
            Note that a face seen only by a view outside that window is culled,
            so this trades runtime for over-culling.
        min_views: Keep a face only if at least this many views see it. Raising
            it culls grazing, single-view geometry that is technically observed
            but poorly constrained; 1 removes only the genuinely unseen.
        clean: Also drop degenerate geometry and small floating components
            afterwards (see :func:`_clean_mesh`). Culling routinely leaves
            specks behind where a mostly-unseen region kept a few faces.

    Returns:
        ``(mesh, stats)`` -- a new mesh, and a dict with ``num_faces_before``/
        ``num_faces_after``/``num_culled``/``fraction_culled``,
        ``num_views_used``, ``min_views``, and ``observation_histogram``: how
        many faces were seen by exactly 0, 1, 2, ... views, truncated at 8+.
        The histogram is the diagnostic worth reading -- a big spike at 0 on a
        capture that circled the subject means the poses or the scale are
        wrong, not that the subject has a large unseen back.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``min_views`` is not
            positive, or *every* face would be culled. The last is not a
            legitimate outcome to return quietly: it means the dataset and the
            mesh do not describe the same scene (wrong poses, wrong scale, a
            mesh in a different coordinate frame), and handing back an empty
            mesh at the end of a long run would hide that.
    """
    from .texturing import face_visibility

    _require_open3d()

    if min_views <= 0:
        raise ValueError(f"min_views must be positive, got {min_views}.")
    if len(mesh.triangles) == 0:
        raise ValueError("Cannot cull faces from a mesh with no triangles.")

    visible = face_visibility(mesh, dataset, max_views=max_views)
    counts = visible.sum(axis=1)
    keep = counts >= min_views

    histogram = np.bincount(np.minimum(counts, 8), minlength=9).tolist()
    stats = {
        "num_faces_before": int(len(mesh.triangles)),
        "num_faces_after": int(len(mesh.triangles)),
        "num_culled": 0,
        "fraction_culled": 0.0,
        "num_views_used": int(visible.shape[1]),
        "min_views": int(min_views),
        "observation_histogram": histogram,
    }

    if not keep.any():
        raise ValueError(
            f"Every one of the {len(mesh.triangles)} faces would be culled: no "
            f"view among the {visible.shape[1]} consulted sees any of them. "
            "That is a mismatch between the mesh and the dataset -- wrong "
            "poses, wrong scale, or a mesh in a different coordinate frame -- "
            "not a mesh with a large unseen back."
        )

    culled = mesh
    if not keep.all():
        culled = _geometry_only_copy(mesh)
        culled.remove_triangles_by_mask(~keep)
        culled.remove_unreferenced_vertices()
        if clean:
            culled = _clean_mesh(culled)
        culled.compute_vertex_normals()

    stats["num_faces_after"] = int(len(culled.triangles))
    stats["num_culled"] = stats["num_faces_before"] - stats["num_faces_after"]
    stats["fraction_culled"] = (
        stats["num_culled"] / stats["num_faces_before"]
        if stats["num_faces_before"]
        else 0.0
    )
    return culled, stats


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


def _vertex_neighbours(triangles: np.ndarray, num_vertices: int):
    """Flattened one-ring adjacency: ``(starts, counts, neighbours)``.

    A CSR-style layout rather than a list of sets, so the smoothing step is a
    couple of vectorised gathers instead of a Python loop over every vertex.
    """
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
            triangles[:, [1, 0]],
            triangles[:, [2, 1]],
            triangles[:, [0, 2]],
        ]
    )
    edges = np.unique(edges, axis=0)
    order = np.argsort(edges[:, 0], kind="stable")
    edges = edges[order]
    counts = np.bincount(edges[:, 0], minlength=num_vertices)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return starts, counts, edges[:, 1]


def _tangential_smoothing(vertices, normals, starts, counts, neighbours, strength):
    """Laplacian smoothing projected onto each vertex's tangent plane.

    **The projection is the whole point.** A plain Laplacian moves every vertex
    toward the average of its neighbours, and on any convex surface that
    average lies *inside* it -- so the regulariser shrinks a correct sphere
    a little every iteration, and a refinement that reported "converged" would
    be reporting a slow collapse. Restricting the smoothing to the tangent
    plane lets it redistribute vertices *over* the surface without moving the
    surface, which is exactly what a regulariser here should do: the
    photometric term already owns the normal direction, and the two then never
    fight.
    """
    sums = np.zeros_like(vertices)
    valid = counts > 0
    for axis in range(3):
        component = vertices[neighbours, axis]
        sums[:, axis] = (
            np.add.reduceat(component, starts, axis=0) if len(neighbours) else 0.0
        )
    means = np.zeros_like(vertices)
    means[valid] = sums[valid] / counts[valid, None]
    delta = np.zeros_like(vertices)
    delta[valid] = means[valid] - vertices[valid]
    # Drop the normal component; keep only motion along the surface.
    delta -= (delta * normals).sum(axis=1, keepdims=True) * normals
    return vertices + strength * delta


def refine_mesh_photometric(
    mesh,
    dataset,
    num_levels: int = 1,
    iterations: int = 6,
    num_offsets: int = 4,
    step_spacings: float = 0.5,
    smoothing: float = 0.3,
    max_views: Optional[int] = None,
    reference_points: Optional[np.ndarray] = None,
    min_views: int = 2,
):
    """Move each vertex along its normal to agree with the photographs.

    Nothing else in this package moves a vertex to fit the images. Every
    geometry lever is either upstream (bundle adjustment, dense MVS, 2DGS
    depth) or subtractive (:func:`cull_unobserved_faces`,
    :func:`simplify_mesh`). This is the variational photometric refinement
    stage OpenMVS ships as ``RefineMesh``, after Vu, Labatut, Pons & Keriven,
    *High Accuracy and Visibility-Consistent Dense Multiview Stereo*, TPAMI
    2012.

    It composes with
    :func:`gsplat.photogrammetry.photometric_alignment.refine_camera_poses_photometric`
    in the obvious order: refine the cameras against the surface, then the
    surface against the cameras.

    **The search is discrete, not a gradient step.** Each vertex is offered a
    fixed set of offsets along its own normal and keeps the one whose views
    agree best. A photoconsistency objective is not smooth -- occlusion changes
    discontinuously as a vertex crosses a silhouette -- so a line search over a
    handful of candidates is both cheaper to reason about and better behaved
    than differentiating through the visibility test. All candidates for all
    vertices go through :func:`~.texturing._view_samples` in **one** pass, so
    the cost is one sampling pass per iteration, not one per offset.

    Photoconsistency is the weighted variance of the colours the visible views
    report at a point: zero when every camera agrees, which is what a correct
    surface produces. A vertex seen by fewer than ``min_views`` views is left
    alone -- one view agrees with itself at every depth, so the objective is
    flat and any motion it induces is noise.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``. Not modified; a refined copy
            is returned.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        num_levels: Image-pyramid levels, coarsest first. **Defaults to 1 --
            no pyramid -- because it was measured and does not help here.**
            Coarse-to-fine is the standard prescription for a photometric
            objective, and on this one it makes both axes worse: recovering a
            sphere perturbed by 0.03, single-scale improves the radial error
            1.95x where three levels manage 1.21x, and on an *already correct*
            sphere three levels drift the mean radius to 0.983 where
            single-scale holds 0.9986. Halving the image blurs away the very
            detail the photoconsistency is measured from, so the coarse levels
            optimise noise. (The same prescription was tested and falsified for
            camera refinement in
            :mod:`gsplat.photogrammetry.photometric_alignment`, for the same
            reason: the displacement being corrected is already well inside the
            detail's wavelength.) Kept as an option for captures whose error is
            large relative to their texture.
        iterations: Search/smooth rounds per level.
        num_offsets: Candidate offsets each side of the current position, so
            each vertex chooses among ``2 * num_offsets + 1`` positions.
        step_spacings: The largest offset, in units of the mesh's own vertex
            spacing (:func:`_point_spacing`). Scale-free by construction: the
            step means the same thing on a tabletop scan and a city block, and
            it halves at each finer pyramid level.
        smoothing: Strength of the tangential Laplacian regulariser, per round.
        max_views: If given, only the first ``max_views`` images are used.
        reference_points: Optional cloud to measure the cloud-to-mesh fit
            against, before and after.
        min_views: Minimum views that must see a vertex for it to move.

    Returns:
        ``(mesh, stats)``. ``stats`` reports ``mean_photoconsistency`` before
        and after, ``mean_vertex_displacement``, ``num_vertices_moved``, the
        per-level history, and -- when ``reference_points`` is given --
        ``point_to_mesh`` before and after.

    Raises:
        ValueError: If ``mesh`` has no triangles or no vertices.
    """
    from .photometric_alignment import _PosedPyramidDataset
    from .texturing import _view_samples

    o3d = _require_open3d()

    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        raise ValueError(
            "Cannot photometrically refine a mesh with no geometry: there are "
            "no vertices to move."
        )
    if num_offsets < 1:
        raise ValueError(
            f"num_offsets must be at least 1, got {num_offsets}: the search "
            "needs at least one candidate either side of the current position "
            "to compare against."
        )

    working = _geometry_only_copy(mesh)
    working.compute_vertex_normals()
    triangles = np.asarray(working.triangles)
    vertices = np.asarray(working.vertices, dtype=np.float64).copy()
    initial = vertices.copy()
    starts, counts, neighbours = _vertex_neighbours(triangles, len(vertices))

    camtoworlds = np.stack(
        [
            np.asarray(dataset[i]["camtoworld"].numpy(), dtype=np.float64)
            for i in range(len(dataset))
        ]
    )
    spacing = _point_spacing(vertices) if len(vertices) > 5 else 0.0

    def visible_pairs(points, normals, occluder):
        """Which views can see each surface point, and with what weight.

        Visibility is decided **once, at the surface**, and then reused for
        every candidate offset. That is not an optimisation, it is the only
        formulation that works: :func:`~.texturing._view_samples` ray-casts
        each sample against the mesh, so a point displaced off the surface is
        occluded *by the surface it came from* and reports as invisible.
        Measured on a correct sphere, that rejected **every one of 482
        candidates at every nonzero offset** -- the search had nothing to
        choose between and the mesh drifted on noise alone.

        The question being asked is "if the surface were here instead, would
        the cameras agree?", and the cameras that can see a vertex do not
        change because it moved a fraction of a vertex spacing. So the hub
        still owns visibility, occlusion and weighting; the offsets are pure
        projection.
        """
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occluder))
        pairs = []
        for chunk, _sampled, weight, view in _view_samples(
            scene, o3d, dataset, points, normals, max_views, 1 << 20
        ):
            pairs.append((view, chunk, weight))
        return pairs

    def photoconsistency(points, pairs):
        """Weighted colour variance of ``points`` through pre-computed views."""
        from .texturing import _bilinear

        weight = np.zeros(len(points))
        first = np.zeros((len(points), 3))
        second = np.zeros((len(points), 3))
        seen = np.zeros(len(points), dtype=np.int64)
        for view, chunk, view_weight in pairs:
            data = dataset[view]
            camtoworld = np.asarray(data["camtoworld"].numpy(), dtype=np.float64)
            K = np.asarray(data["K"].numpy(), dtype=np.float64)
            image = np.asarray(data["image"].numpy(), dtype=np.float64) / 255.0
            height, width = image.shape[:2]
            viewmat = np.linalg.inv(camtoworld)
            Xc = (viewmat[:3, :3] @ points[chunk].T + viewmat[:3, 3:4]).T
            uvw = (K @ Xc.T).T
            uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
            inside = (
                (Xc[:, 2] > 1e-4)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            if not inside.any():
                continue
            keep = chunk[inside]
            sampled = _bilinear(image, uv[inside])
            kept_weight = view_weight[inside]
            weight[keep] += kept_weight
            first[keep] += sampled * kept_weight[:, None]
            second[keep] += (sampled**2) * kept_weight[:, None]
            seen[keep] += 1
        ok = weight > 0
        mean = np.zeros_like(first)
        mean[ok] = first[ok] / weight[ok, None]
        variance = np.zeros_like(first)
        variance[ok] = np.clip(second[ok] / weight[ok, None] - mean[ok] ** 2, 0.0, None)
        return variance.sum(axis=1), seen

    def measure(current):
        surface = _geometry_only_copy(working)
        surface.vertices = o3d.utility.Vector3dVector(current)
        surface.compute_vertex_normals()
        normals = np.asarray(surface.vertex_normals, dtype=np.float64)
        pairs = visible_pairs(current, normals, surface)
        cost, seen = photoconsistency(current, pairs)
        usable = seen >= min_views
        if not usable.any():
            return float("nan"), normals, surface
        return float(cost[usable].mean()), normals, surface

    view0 = _PosedPyramidDataset(dataset, camtoworlds, levels=0)
    dataset_full = dataset
    dataset = view0
    before, _normals, _surface = measure(vertices)

    history = []
    for level in range(num_levels - 1, -1, -1):
        dataset = _PosedPyramidDataset(dataset_full, camtoworlds, levels=level)
        # The step does *not* shrink with the level. Coarse-to-fine buys a
        # wider basin of convergence, which is bought with a larger step at the
        # coarse level, not a smaller one -- and on this objective it does not
        # pay for itself either way; see `num_levels`.
        step = step_spacings * spacing
        if step <= 0.0:
            continue
        offsets = np.linspace(-step, step, 2 * num_offsets + 1)
        zero_index = num_offsets
        for _ in range(iterations):
            surface = _geometry_only_copy(working)
            surface.vertices = o3d.utility.Vector3dVector(vertices)
            surface.compute_vertex_normals()
            normals = np.asarray(surface.vertex_normals, dtype=np.float64)

            # Visibility once, at the surface; then every candidate offset is
            # a projection through those same views.
            pairs = visible_pairs(vertices, normals, surface)
            costs, seens = [], []
            for offset in offsets:
                cost_d, seen_d = photoconsistency(vertices + offset * normals, pairs)
                costs.append(cost_d)
                seens.append(seen_d)
            cost = np.stack(costs)
            seen = np.stack(seens)

            # A candidate no view can see is not "perfectly consistent"; it is
            # unmeasured, and must never win. Same distinction
            # `point_to_mesh_distance` makes between None and 0.0.
            cost = np.where(seen >= min_views, cost, np.inf)
            choice = np.argmin(cost, axis=0)
            best = cost[choice, np.arange(cost.shape[1])]
            here = cost[zero_index]
            # **Staying put is the default, and a move has to earn it.**
            # `np.argmin` returns the *first* minimum, and the offsets run from
            # most-inward to most-outward, so every tie -- and every vertex
            # whose candidates are all unmeasurable -- silently resolves
            # inward. Measured, that alone walked a correct sphere from radius
            # 1.000 to 0.986. Requiring a strict improvement over the current
            # position removes the drift without weakening the search.
            improves = np.isfinite(here) & np.isfinite(best) & (best < here)
            choice = np.where(improves, choice, zero_index)
            vertices = vertices + offsets[choice][:, None] * normals

            vertices = _tangential_smoothing(
                vertices, normals, starts, counts, neighbours, smoothing
            )
        level_cost, _n, _s = measure(vertices)
        history.append({"level": int(level), "photoconsistency": level_cost})

    dataset = view0
    after, _normals, refined_surface = measure(vertices)

    displacement = np.linalg.norm(vertices - initial, axis=1)
    stats = {
        "mean_photoconsistency_before": before,
        "mean_photoconsistency_after": after,
        "mean_vertex_displacement": float(displacement.mean()),
        "max_vertex_displacement": float(displacement.max()),
        "num_vertices_moved": int((displacement > 1e-12).sum()),
        "num_vertices": int(len(vertices)),
        "vertex_spacing": float(spacing),
        "levels": history,
    }
    if reference_points is not None:
        from .metrics import point_to_mesh_distance

        stats["point_to_mesh_before"] = point_to_mesh_distance(reference_points, mesh)
        stats["point_to_mesh_after"] = point_to_mesh_distance(
            reference_points, refined_surface
        )
    return refined_surface, stats


def derive_reconstruction_parameters(
    points: np.ndarray,
    max_grid: int = 2048,
    sdf_trunc_voxels: float = 4.0,
    normal_radius_spacings: float = 3.0,
    normal_max_nn: int = 30,
) -> Dict[str, float]:
    """Choose the reconstruction's absolute constants from the evidence.

    ``voxel_size=0.01``, ``sdf_trunc=0.04``, ``depth_trunc=10.0`` and
    ``estimate_normals(radius=0.1)`` were the last magic numbers in this
    package, and they are all in **scene units** -- on a pipeline whose stated
    design goal is that a number in scene units means nothing on its own
    (``docs/handoff/SCOPE.md``). 0.01 is a fifth of a millimetre on a coin and
    a centimetre on a cathedral; one of those reconstructions is impossible and
    the other throws away most of what was captured.

    Each is derived here from the reference cloud's own k-nearest-neighbour
    spacing and extent, the same way :func:`simplify_mesh_to_error` already
    derives its error budget:

    - **voxel_size = the cloud's point spacing.** One voxel per sample. Finer
      invents detail no measurement supports and multiplies the grid; coarser
      discards detail that was paid for. Clamped so the grid stays under
      ``max_grid`` voxels along the bounding box's diagonal, because the
      alternative on a large scene is exhausting memory rather than producing
      a poor mesh.
    - **sdf_trunc = 4 x voxel_size**, the ratio open3d's own examples use and
      exactly the ratio the old defaults encoded (0.04 / 0.01). Keeping it as a
      *ratio* is the point: it now follows the voxel size instead of having to
      be remembered alongside it.
    - **depth_trunc = 1.5 x the bounding-box diagonal.** Its job is to reject
      background and sky, so the scene's own size is the only defensible scale.
      The 1.5 leaves room for cameras standing well back from the subject.
    - **normal_radius = 3 x point spacing.** A plane fit needs enough
      neighbours to be stable, and on a surface sampled at spacing ``s`` a disc
      of radius ``3s`` contains about ``pi * 9 = 28`` points -- which is where
      the companion ``max_nn=30`` comes from. The two agreed by coincidence
      before; now they agree by construction.

    Args:
        points: ``(P, 3)`` reference cloud -- the dense MVS cloud, the sparse
            SfM points, or the back-projected depth being fused.
        max_grid: Cap on voxels along the bounding-box diagonal.
        sdf_trunc_voxels: Truncation distance, in voxels.
        normal_radius_spacings: Normal-estimation radius, in point spacings.
        normal_max_nn: Neighbour cap passed to open3d's normal estimation.

    Returns:
        A dict with the derived ``voxel_size``, ``sdf_trunc``, ``depth_trunc``,
        ``normal_radius`` and ``normal_max_nn``, plus the ``point_spacing`` and
        ``diagonal`` they came from and whether the voxel size was
        ``clamped``.

    Raises:
        ValueError: If there are too few points to measure a spacing, or the
            cloud is degenerate (every point in one place).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")

    spacing = _point_spacing(points)
    extent = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0.0 or spacing <= 0.0:
        raise ValueError(
            "Cannot derive reconstruction parameters from a degenerate cloud "
            f"(extent {extent}, point spacing {spacing}). Every point is in "
            "the same place, or the cloud contains non-finite coordinates."
        )

    voxel_size = spacing
    smallest_allowed = diagonal / float(max_grid)
    clamped = voxel_size < smallest_allowed
    if clamped:
        voxel_size = smallest_allowed

    return {
        "point_spacing": float(spacing),
        "diagonal": diagonal,
        "voxel_size": float(voxel_size),
        "sdf_trunc": float(sdf_trunc_voxels * voxel_size),
        "depth_trunc": float(1.5 * diagonal),
        "normal_radius": float(normal_radius_spacings * spacing),
        "normal_max_nn": int(normal_max_nn),
        "clamped": bool(clamped),
    }


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
    bake_texture_atlas_pages,
    bake_texture_atlas_view_selected,
    face_projected_areas,
    face_view_quality,
    partition_faces,
    face_visibility,
    level_seams,
    recommended_texture_size,
    select_views_mrf,
)


# ---------------------------------------------------------------------------
# Level-set extraction from a scalar field
#
# TSDF fusion of rendered depth (`extract_mesh_tsdf`) is the 2020-era answer to
# "turn a radiance field into a surface". Current work extracts the surface
# from the Gaussian field itself -- Gaussian Opacity Fields (Yu et al., 2024)
# takes a level set of the opacity field via tetrahedral marching; SuGaR
# (Guedon & Lepetit, 2024) and PGSR (Chen et al., 2024) take related routes.
#
# Evaluating a Gaussian field needs a GPU and gsplat's compiled CUDA
# extension, neither of which exists in the environment this was written in.
# So the split below is deliberate and follows the precedent `_tsdf_fuse`
# already set: everything except the field evaluation is pure NumPy/open3d and
# is verified against an **analytic** field, and the one function that needs a
# GPU is thin, isolated, and marked as never having run.
# ---------------------------------------------------------------------------

# Kuhn decomposition: the six tetrahedra that tile a unit cube, as indices into
# the cube's eight corners ordered by (i, j, k) bits. Every tetrahedron shares
# the cube's main diagonal 0-7, which is what makes the tiling consistent
# between neighbouring cubes -- and therefore the surface watertight.
_CUBE_TETRAHEDRA = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int64,
)

_TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _marching_tetrahedra_table():
    """Which tetrahedron edges each inside/outside pattern cuts, as triangles.

    Generated rather than typed out. A hand-written 16-case table is the
    classic place for a transposition to hide: it would produce a surface that
    looks plausible and is quietly non-manifold in a few cells, which is
    exactly the kind of defect `compute_uvatlas` then *segfaults* on
    (``docs/handoff/ISSUES.md``) rather than reporting.

    The rule is short enough to state directly. A tetrahedron edge is cut when
    its two ends fall on opposite sides of the level. With one corner inside,
    the three edges leaving it form one triangle. With three inside, the same
    holds for the one corner outside. With two inside (``i``, ``j``) and two out
    (``k``, ``l``), the four cut edges form a quad, and taking them in the
    cyclic order (i,k), (i,l), (j,l), (j,k) is what makes the two triangles
    split it along a diagonal instead of crossing it.
    """
    index = {edge: i for i, edge in enumerate(_TET_EDGES)}

    def edge_id(a, b):
        return index[(a, b) if a < b else (b, a)]

    table = []
    for case in range(16):
        inside = [c for c in range(4) if case & (1 << c)]
        outside = [c for c in range(4) if not case & (1 << c)]
        if len(inside) in (0, 4):
            table.append([])
        elif len(inside) == 1:
            i = inside[0]
            table.append([[edge_id(i, c) for c in outside]])
        elif len(inside) == 3:
            k = outside[0]
            table.append([[edge_id(c, k) for c in inside]])
        else:
            i, j = inside
            k, l = outside
            quad = [edge_id(i, k), edge_id(i, l), edge_id(j, l), edge_id(j, k)]
            table.append([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
    return table


_MARCHING_TET_TABLE = _marching_tetrahedra_table()


def extract_level_set(
    field,
    bounds,
    resolution: int = 64,
    level: float = 0.0,
    batch_size: int = 1 << 20,
    clean: bool = True,
):
    """Extract the ``level`` iso-surface of a scalar field by marching tetrahedra.

    Pure NumPy and open3d: ``field`` is any callable taking ``(P, 3)`` query
    points and returning ``(P,)`` values, so this is exercised here against a
    closed-form sphere rather than against a Gaussian field it cannot run.
    Pair it with :func:`gaussian_density_field` for the real thing.

    Tetrahedra rather than cubes, following Gaussian Opacity Fields: marching
    cubes' ambiguous face cases need a disambiguation rule to avoid holes,
    while a tetrahedron's four corners admit no ambiguity at all -- one corner
    apart from the other three, or two apart from two, and both have a single
    triangulation. Each grid cube is split by the Kuhn decomposition, whose six
    tetrahedra all share the cube's main diagonal, so neighbouring cubes agree
    on their shared faces and the surface closes.

    Vertices are identified by the **grid edge** they sit on, so the two
    tetrahedra either side of a face produce the *same* vertex rather than two
    coincident ones. That is what makes the result watertight instead of a
    triangle soup that merely looks like a surface.

    Args:
        field: Callable ``(P, 3) -> (P,)``. Negative inside the surface by the
            usual convention, though only the sign change matters.
        bounds: ``((min_x, min_y, min_z), (max_x, max_y, max_z))``.
        resolution: Cells along the longest axis of ``bounds``. Cells are cubic,
            so the other axes get however many fit.
        level: The iso-value to extract.
        batch_size: Query points per call into ``field``, bounding peak memory
            (and, for a GPU field, VRAM).
        clean: Drop degenerate triangles and small floating components via
            :func:`_clean_mesh`.

    Returns:
        ``(mesh, stats)``. ``stats`` reports the ``cell_size``, the
        ``grid_shape``, how many points were evaluated, and the triangle and
        vertex counts.

    Raises:
        ValueError: If ``resolution`` is below 1 or ``bounds`` is degenerate.
    """
    o3d = _require_open3d()

    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    extent = upper - lower
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}.")
    if not np.all(np.isfinite(extent)) or np.max(extent) <= 0.0:
        raise ValueError(
            f"Degenerate bounds {bounds!r}: the box has no positive extent, so "
            "there is no volume to march through."
        )

    cell = float(np.max(extent)) / resolution
    counts = np.maximum(np.ceil(extent / cell).astype(np.int64), 1)
    shape = counts + 1  # grid *points* per axis

    axes = [lower[a] + cell * np.arange(shape[a]) for a in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    values = np.empty(len(grid), dtype=np.float64)
    for start in range(0, len(grid), batch_size):
        chunk = grid[start : start + batch_size]
        values[start : start + batch_size] = np.asarray(
            field(chunk), dtype=np.float64
        ).reshape(-1)

    # Cube corner offsets, ordered so bit 0 is +i, bit 1 is +j, bit 2 is +k --
    # the order `_CUBE_TETRAHEDRA` indexes into.
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    corner_offsets = np.array(
        [[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], dtype=np.int64
    )
    base = np.stack(
        np.meshgrid(*[np.arange(counts[a]) for a in range(3)], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cube_corners = (base[:, None, :] + corner_offsets[None, :, :]) @ strides  # (C, 8)

    tets = cube_corners[:, _CUBE_TETRAHEDRA].reshape(-1, 4)  # (6C, 4)
    tet_values = values[tets]
    inside = tet_values < level
    case = (
        inside[:, 0].astype(np.int64)
        | (inside[:, 1].astype(np.int64) << 1)
        | (inside[:, 2].astype(np.int64) << 2)
        | (inside[:, 3].astype(np.int64) << 3)
    )

    edge_pairs = []
    triangles = []
    for case_id, tri_list in enumerate(_MARCHING_TET_TABLE):
        if not tri_list:
            continue
        selected = np.nonzero(case == case_id)[0]
        if selected.size == 0:
            continue
        corners = tets[selected]
        for tri in tri_list:
            columns = []
            for edge in tri:
                a, b = _TET_EDGES[edge]
                pair = np.stack([corners[:, a], corners[:, b]], axis=1)
                # Canonical (low, high) order, so the two tetrahedra sharing
                # this edge name the same vertex.
                pair = np.sort(pair, axis=1)
                columns.append(pair)
            edge_pairs.append(np.concatenate(columns, axis=0))
            triangles.append(selected.size)

    mesh = o3d.geometry.TriangleMesh()
    stats = {
        "cell_size": cell,
        "grid_shape": [int(v) for v in shape],
        "num_points_evaluated": int(len(grid)),
        "level": float(level),
        "num_vertices": 0,
        "num_triangles": 0,
    }
    if not edge_pairs:
        return mesh, stats

    all_pairs = np.concatenate(edge_pairs, axis=0)
    unique_pairs, inverse = np.unique(all_pairs, axis=0, return_inverse=True)

    low = values[unique_pairs[:, 0]]
    high = values[unique_pairs[:, 1]]
    denominator = high - low
    # A vertex whose two ends carry the same value sits at the midpoint: the
    # crossing is real (the signs differ) but the linear model has no slope to
    # place it with.
    t = np.where(
        np.abs(denominator) > 1e-12,
        (level - low) / np.where(denominator == 0, 1, denominator),
        0.5,
    )
    t = np.clip(t, 0.0, 1.0)
    p0 = grid[unique_pairs[:, 0]]
    p1 = grid[unique_pairs[:, 1]]
    vertices = p0 + t[:, None] * (p1 - p0)

    # `edge_pairs` was built one triangle-corner column at a time, so the
    # inverse indices come back grouped by corner; regroup into triangles.
    faces = []
    cursor = 0
    for count in triangles:
        corner_block = inverse[cursor : cursor + 3 * count].reshape(3, count).T
        faces.append(corner_block)
        cursor += 3 * count
    faces = np.concatenate(faces, axis=0)

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if clean:
        mesh = _clean_mesh(mesh)
    mesh.compute_vertex_normals()

    stats["num_vertices"] = int(len(mesh.vertices))
    stats["num_triangles"] = int(len(mesh.triangles))
    return mesh, stats


def gaussian_density_field(
    splats: Dict[str, Tensor],
    device: str = "cuda",
    batch_size: int = 1 << 18,
):
    """The Gaussian field's density at arbitrary points, as a callable.

    **This half has never been run.** It needs a GPU and gsplat's compiled CUDA
    extension, and the environment this was written in has neither
    (``torch.cuda.is_available()`` is False, "No CUDA toolkit found"). It is
    deliberately thin, and everything downstream of it --
    :func:`extract_level_set` and the mesh cleanup -- is pure NumPy/open3d and
    *is* verified, against an analytic field. See ``docs/handoff/PROGRESS.md``
    for what that split means.

    The density at a point is the opacity-weighted sum of each Gaussian's
    value there, which is the field Gaussian Opacity Fields takes a level set
    of. Returned negated (``level - density``) so the convention matches
    :func:`extract_level_set`'s: negative inside the surface.

    Args:
        splats: Trained Gaussian parameters, as in a gsplat checkpoint's
            ``"splats"`` entry.
        device: Torch device to evaluate on.
        batch_size: Query points per batch.

    Returns:
        A callable ``(P, 3) -> (P,)`` suitable for :func:`extract_level_set`.

    Raises:
        ValueError: If ``splats`` carries per-image appearance embeddings --
            the same refusal :func:`_splat_colors` makes, and for the same
            reason: per-image appearance does not map onto one canonical
            surface.
    """
    # Reuses the appearance-embedding refusal rather than re-deriving it.
    _splat_colors(splats)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)

    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quats.unbind(-1)
    rotation = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    def field(points: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(np.asarray(points), dtype=means.dtype, device=device)
        out = torch.empty(len(query), dtype=means.dtype, device=device)
        for start in range(0, len(query), batch_size):
            chunk = query[start : start + batch_size]
            # (B, N, 3) in each Gaussian's own frame, then the Mahalanobis
            # exponent of an axis-aligned Gaussian there.
            delta = chunk[:, None, :] - means[None, :, :]
            local = torch.einsum("nij,bnj->bni", rotation.transpose(1, 2), delta)
            exponent = -0.5 * ((local / scales[None, :, :]) ** 2).sum(-1)
            out[start : start + batch_size] = (
                opacities[None, :] * torch.exp(exponent)
            ).sum(-1)
        return out.detach().cpu().numpy()

    return field
