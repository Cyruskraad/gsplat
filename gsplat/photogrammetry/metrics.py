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
"""Automatic quality metrics for the photogrammetry pipeline.

:mod:`gsplat.photogrammetry.bundle_adjustment` already reports quantitative
stats (reprojection error before/after), but the rest of the pipeline --
dense MVS, mesh extraction, neural-SfM import -- had no automatic way to
tell whether its output was any good. This module closes that gap with
three pure, dependency-light functions that the CLIs in ``examples/`` and
``Runner.extract_mesh()`` in ``examples/simple_trainer_2dgs.py`` write to
``stats/*.json`` files, following the same convention already used by the
trainers' render-quality evaluation (PSNR/SSIM/LPIPS).

- :func:`point_to_mesh_distance`: "cloud-to-mesh" fit -- does an extracted
  mesh actually pass through the point cloud (sparse SfM or dense MVS) it
  was built from?
- :func:`mesh_quality_stats`: watertightness, connected-component count,
  surface area/volume, edge-length statistics.
- :func:`point_cloud_stats`: point count, bounding-box extent, and k-NN
  spacing (a density proxy) for a (sparse or dense) point cloud.

``point_to_mesh_distance`` and ``mesh_quality_stats`` require the optional
``open3d`` dependency (``pip install gsplat[mesh]``, same as
:mod:`gsplat.photogrammetry.mesh_extraction`); ``point_cloud_stats`` requires
``scikit-learn`` (already required by :mod:`gsplat.photogrammetry.neural_sfm`
and ``examples/utils.py``).
"""

from typing import Dict, Sequence

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "gsplat.photogrammetry.metrics requires open3d for mesh-based "
            "metrics. Install it with `pip install gsplat[mesh]` (or `pip "
            "install open3d`)."
        ) from e
    return o3d


def point_to_mesh_distance(
    points: np.ndarray,
    mesh,
    percentiles: Sequence[float] = (50.0, 95.0),
) -> Dict[str, float]:
    """Unsigned distance from each point in a cloud to the nearest mesh surface.

    The classic "cloud-to-mesh" QA check: gsplat has no ground-truth mesh to
    compare an extracted mesh to, so the point cloud it was reconstructed
    from (the sparse SfM cloud, or a dense MVS cloud) is the practical,
    always-available reference -- a mesh that doesn't pass close to the
    points it was built from is a bad reconstruction regardless of how
    plausible it looks.

    Args:
        points: (P, 3) point positions.
        mesh: An ``open3d.geometry.TriangleMesh`` (e.g. from
            :func:`gsplat.photogrammetry.mesh_extraction.extract_mesh_tsdf`
            or :func:`.extract_mesh_poisson`).
        percentiles: Additional distance percentiles to report (besides
            mean/rms/max).

    Returns:
        A dict with ``num_points``, ``mean``, ``rms``, ``max``, and one
        ``p{percentile}`` entry per value in ``percentiles``, all in the
        same scene units as ``points``/``mesh``.
    """
    o3d = _require_open3d()

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    distances = scene.compute_distance(o3d.core.Tensor(points)).numpy()

    stats: Dict[str, float] = {
        "num_points": int(points.shape[0]),
        "mean": float(distances.mean()),
        "rms": float(np.sqrt(np.mean(distances.astype(np.float64) ** 2))),
        "max": float(distances.max()),
    }
    for p in percentiles:
        stats[f"p{p:g}"] = float(np.percentile(distances, p))
    return stats


def mesh_quality_stats(mesh) -> Dict[str, object]:
    """Intrinsic quality stats for a triangle mesh.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.

    Returns:
        A dict with ``num_vertices``, ``num_triangles``, ``is_watertight``,
        ``num_connected_components``, ``surface_area``, ``volume`` (``None``
        if the mesh is not watertight, since Open3D's volume computation
        requires a watertight, orientable mesh), and
        ``mean``/``min``/``max_edge_length``.
    """
    o3d = _require_open3d()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    num_vertices = int(vertices.shape[0])
    num_triangles = int(triangles.shape[0])

    if num_triangles > 0:
        edge_lengths = np.concatenate(
            [
                np.linalg.norm(
                    vertices[triangles[:, a]] - vertices[triangles[:, b]], axis=1
                )
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
        )
        mean_edge_length = float(edge_lengths.mean())
        min_edge_length = float(edge_lengths.min())
        max_edge_length = float(edge_lengths.max())
        _, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        num_connected_components = int(len(np.asarray(cluster_n_triangles)))
        is_watertight = bool(mesh.is_watertight())
        surface_area = float(mesh.get_surface_area())
        volume = float(mesh.get_volume()) if is_watertight else None
    else:
        mean_edge_length = min_edge_length = max_edge_length = 0.0
        num_connected_components = 0
        is_watertight = False
        surface_area = 0.0
        volume = None

    return {
        "num_vertices": num_vertices,
        "num_triangles": num_triangles,
        "is_watertight": is_watertight,
        "num_connected_components": num_connected_components,
        "surface_area": surface_area,
        "volume": volume,
        "mean_edge_length": mean_edge_length,
        "min_edge_length": min_edge_length,
        "max_edge_length": max_edge_length,
    }


def point_cloud_stats(points: np.ndarray, k: int = 4) -> Dict[str, object]:
    """Point count, bounding-box extent, and k-NN spacing for a point cloud.

    The mean/median k-NN spacing is a density proxy -- e.g. useful to check
    that a dense MVS cloud is actually denser than the sparse SfM cloud it
    was fused from, following the same
    ``sklearn.neighbors.NearestNeighbors`` pattern as ``examples/utils.py``'s
    ``knn()``.

    Args:
        points: (P, 3) point positions.
        k: Number of nearest neighbors (excluding the point itself) to
            average spacing over.

    Returns:
        A dict with ``num_points``, ``bbox_min``, ``bbox_max``,
        ``bbox_extent`` (each a 3-list), and ``mean_knn_distance``/
        ``median_knn_distance``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")
    num_points = points.shape[0]

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    stats: Dict[str, object] = {
        "num_points": int(num_points),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": (bbox_max - bbox_min).tolist(),
    }

    if num_points > 1:
        from sklearn.neighbors import NearestNeighbors

        # +1 neighbor since a point is always its own (zero-distance)
        # nearest neighbor; that self-match is dropped below.
        k_eff = min(k + 1, num_points)
        model = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(points)
        distances, _ = model.kneighbors(points)
        neighbor_distances = distances[:, 1:]
        stats["mean_knn_distance"] = float(neighbor_distances.mean())
        stats["median_knn_distance"] = float(np.median(neighbor_distances))
    else:
        stats["mean_knn_distance"] = 0.0
        stats["median_knn_distance"] = 0.0

    return stats
