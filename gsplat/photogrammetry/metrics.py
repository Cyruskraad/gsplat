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

Every stage of the pipeline reports quantitative stats through this module,
so a full run is measurable end to end rather than only at the render-quality
level. The CLIs in ``examples/`` (and ``Runner.extract_mesh()`` in
``examples/simple_trainer_2dgs.py``) write these to ``stats/*.json`` files,
following the same convention already used by the trainers' render-quality
evaluation (PSNR/SSIM/LPIPS), and
:mod:`gsplat.photogrammetry.pipeline` collects them into one report.

Geometry quality:

- :func:`point_to_mesh_distance`: "cloud-to-mesh" fit -- does an extracted
  mesh actually pass through the point cloud (sparse SfM or dense MVS) it
  was built from?
- :func:`mesh_quality_stats`: watertightness, connected-component count,
  surface area/volume, edge-length statistics.
- :func:`point_cloud_stats`: point count, bounding-box extent, and k-NN
  spacing (a density proxy) for a (sparse or dense) point cloud.

Reconstruction / input quality:

- :func:`reconstruction_stats`: image/point/observation counts, mean track
  length and mean reprojection error of a COLMAP model -- the baseline
  bundle adjustment improves on, and the way neural-SfM output is measured.
- :func:`track_stats`: track-length distribution for feature tracks (used by
  :func:`gsplat.photogrammetry.neural_sfm.merge_point_maps_to_tracks`).
- :func:`mask_coverage_stats` / :func:`depth_prior_stats`: how much of each
  training image the AI-assisted transient masks exclude, and how usable the
  monocular depth priors are.

Texture quality:

- :func:`atlas_sharpness`: how much high-frequency detail a baked UV atlas
  carries -- the number that says whether per-face view selection bought
  anything over blending every view, which pointwise error does not.

``point_to_mesh_distance`` and ``mesh_quality_stats`` require the optional
``open3d`` dependency (``pip install gsplat[mesh]``, same as
:mod:`gsplat.photogrammetry.mesh_extraction`); ``point_cloud_stats`` requires
``scikit-learn`` and ``reconstruction_stats`` requires ``pycolmap`` (both
already required elsewhere in this package).
"""

import os
from typing import Dict, Optional, Sequence

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
        same scene units as ``points``/``mesh``. With no points to measure,
        the distance entries are ``None`` rather than ``0.0`` -- "nothing was
        measured" and "the fit is perfect" must not look alike, here or in the
        derived cross-stage metrics that divide by them.

    Raises:
        ValueError: If ``mesh`` has no triangles. There is no surface to
            measure against, and open3d's raycaster fails on an empty scene
            with an opaque ``IndexError: _Map_base::at`` -- which extraction
            producing a degenerate mesh would otherwise surface at the very
            end of a long run.
    """
    o3d = _require_open3d()

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")
    if len(mesh.triangles) == 0:
        raise ValueError(
            "Cannot measure cloud-to-mesh distance against a mesh with no "
            "triangles. Mesh extraction produced nothing usable -- check the "
            "TSDF voxel_size/sdf_trunc or the Poisson depth for this scene."
        )

    empty_stats: Dict[str, float] = {
        "num_points": 0,
        "mean": None,
        "rms": None,
        "max": None,
    }
    if points.shape[0] == 0:
        for p in percentiles:
            empty_stats[f"p{p:g}"] = None
        return empty_stats

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

    if num_points == 0:
        # An empty cloud is a real outcome (dense MVS fusing nothing), and the
        # rest of this module reports empty input as zeros rather than
        # raising; numpy's bare "zero-size array to reduction operation
        # minimum" would say nothing about which stage produced nothing.
        return {
            "num_points": 0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "bbox_extent": [0.0, 0.0, 0.0],
            "mean_knn_distance": 0.0,
            "median_knn_distance": 0.0,
        }

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


def reconstruction_stats(reconstruction) -> Dict[str, object]:
    """Aggregate quality stats for a COLMAP sparse reconstruction.

    Measures the SfM stage itself -- the baseline bundle adjustment improves
    on, and the way an imported neural-SfM model
    (:func:`gsplat.photogrammetry.neural_sfm.write_colmap_reconstruction`)
    is judged before it's fed to the rest of the pipeline.

    Args:
        reconstruction: A path to a COLMAP sparse model directory, or an
            already-loaded ``pycolmap.Reconstruction``.

    Returns:
        A dict with ``num_images``, ``num_cameras``, ``num_points3D``,
        ``num_observations``, ``mean_track_length``,
        ``mean_observations_per_image`` and ``mean_reprojection_error``
        (pixels).
    """
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "gsplat.photogrammetry.metrics.reconstruction_stats requires "
            "pycolmap. Install it with `pip install pycolmap`."
        ) from e

    if isinstance(reconstruction, (str, os.PathLike)):
        reconstruction = pycolmap.Reconstruction(str(reconstruction))

    return {
        "num_images": int(reconstruction.num_images()),
        "num_cameras": int(reconstruction.num_cameras()),
        "num_points3D": int(reconstruction.num_points3D()),
        "num_observations": int(reconstruction.compute_num_observations()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error": float(
            reconstruction.compute_mean_reprojection_error()
        ),
    }


def track_stats(tracks: Sequence[Sequence]) -> Dict[str, object]:
    """Track-length distribution for a set of feature tracks.

    A track seen by only one image gives bundle adjustment no cross-view
    constraint, so the share of multi-view tracks is the headline number for
    :func:`gsplat.photogrammetry.neural_sfm.merge_point_maps_to_tracks`.

    Args:
        tracks: A sequence of tracks, each a sequence of per-image
            observations (as returned in ``merge_point_maps_to_tracks``'s
            ``"tracks"``).

    Returns:
        A dict with ``num_tracks``, ``num_observations``,
        ``mean``/``median``/``max_track_length`` and
        ``multi_view_track_fraction`` (share of tracks seen by >= 2 images).
    """
    lengths = np.array([len(t) for t in tracks], dtype=np.int64)
    if lengths.size == 0:
        return {
            "num_tracks": 0,
            "num_observations": 0,
            "mean_track_length": 0.0,
            "median_track_length": 0.0,
            "max_track_length": 0,
            "multi_view_track_fraction": 0.0,
        }
    return {
        "num_tracks": int(lengths.size),
        "num_observations": int(lengths.sum()),
        "mean_track_length": float(lengths.mean()),
        "median_track_length": float(np.median(lengths)),
        "max_track_length": int(lengths.max()),
        "multi_view_track_fraction": float((lengths >= 2).mean()),
    }


def mask_coverage_stats(mask_dir: str, max_masks: Optional[int] = None) -> Dict:
    """How much of each training image a transient-object mask keeps.

    Sanity-checks the AI-assisted masking stage (see ``--mask_dir`` in
    ``docs/photogrammetry.md``) before a long training run: a mean kept
    fraction near 1.0 means the segmenter found almost nothing to exclude,
    while a very low one means it is masking away most of the scene.

    Args:
        mask_dir: Directory of ``<image_stem>.png`` masks (nonzero = keep).
        max_masks: If given, only inspect this many masks (for speed).

    Returns:
        A dict with ``num_masks`` and ``mean``/``min``/``max_kept_fraction``
        (0-1, fraction of pixels kept), plus ``mean_excluded_fraction``.
    """
    import imageio.v2 as imageio

    names = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".png"))
    if max_masks is not None:
        names = names[:max_masks]

    kept_fractions = []
    for name in names:
        mask = imageio.imread(os.path.join(mask_dir, name))
        if mask.ndim == 3:
            mask = mask[..., 0]
        kept_fractions.append(float((mask != 0).mean()))

    if not kept_fractions:
        return {
            "num_masks": 0,
            "mean_kept_fraction": 0.0,
            "min_kept_fraction": 0.0,
            "max_kept_fraction": 0.0,
            "mean_excluded_fraction": 0.0,
        }

    kept = np.array(kept_fractions, dtype=np.float64)
    return {
        "num_masks": int(kept.size),
        "mean_kept_fraction": float(kept.mean()),
        "min_kept_fraction": float(kept.min()),
        "max_kept_fraction": float(kept.max()),
        "mean_excluded_fraction": float(1.0 - kept.mean()),
    }


def depth_prior_stats(
    mono_depth_dir: str, max_maps: Optional[int] = None
) -> Dict[str, object]:
    """Usability stats for a directory of monocular depth priors.

    Catches the common failure modes of an externally-run depth model before
    training burns hours on it: maps that are empty, constant, or full of
    non-finite values carry no gradient for
    :func:`gsplat.losses.pearson_depth_loss`.

    Args:
        mono_depth_dir: Directory of ``<image_stem>.npy`` float depth maps.
        max_maps: If given, only inspect this many maps (for speed).

    Returns:
        A dict with ``num_maps``, ``mean_finite_fraction``,
        ``mean_value``/``min_value``/``max_value``,
        ``num_degenerate_maps`` (maps that are constant or entirely
        non-finite, and so useless as supervision), and ``num_not_2d_maps``
        (maps that aren't a single (H, W) array and so can't be loaded at
        all -- typically a model's raw ``(1, H, W)`` output saved unsqueezed).
    """
    names = sorted(f for f in os.listdir(mono_depth_dir) if f.endswith(".npy"))
    if max_maps is not None:
        names = names[:max_maps]

    finite_fractions, mins, maxs, means = [], [], [], []
    num_degenerate = 0
    num_not_2d = 0
    for name in names:
        depth = np.load(os.path.join(mono_depth_dir, name)).astype(np.float64)
        # A map that isn't a bare (H, W) array once singleton axes are dropped
        # can't be used as a depth prior -- the loader rejects it. Count it
        # here so a whole directory of them is caught before training starts.
        if np.squeeze(depth).ndim != 2:
            num_not_2d += 1
        finite = np.isfinite(depth)
        finite_fractions.append(float(finite.mean()))
        if not finite.any():
            num_degenerate += 1
            continue
        values = depth[finite]
        mins.append(float(values.min()))
        maxs.append(float(values.max()))
        means.append(float(values.mean()))
        if values.min() == values.max():
            num_degenerate += 1

    if not names:
        return {
            "num_maps": 0,
            "mean_finite_fraction": 0.0,
            "mean_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
            "num_degenerate_maps": 0,
            "num_not_2d_maps": 0,
        }

    return {
        "num_maps": int(len(names)),
        "mean_finite_fraction": float(np.mean(finite_fractions)),
        "mean_value": float(np.mean(means)) if means else 0.0,
        "min_value": float(np.min(mins)) if mins else 0.0,
        "max_value": float(np.max(maxs)) if maxs else 0.0,
        "num_degenerate_maps": int(num_degenerate),
        "num_not_2d_maps": int(num_not_2d),
    }


def atlas_sharpness(
    texture: np.ndarray, covered_mask: Optional[np.ndarray] = None
) -> Dict[str, object]:
    """How much high-frequency detail a baked texture atlas actually carries.

    This is the number that says whether per-face view selection
    (:func:`gsplat.photogrammetry.texturing.bake_texture_atlas_view_selected`)
    bought anything over blending every view. **Pointwise error against ground
    truth does not**, and measuring it instead is the trap here: blending
    *attenuates* detail while single-view sampling *displaces* it, and on a
    synthetic sphere with simulated residual pose error a displaced-but-sharp
    atlas scores worse pointwise (L1 0.185 vs 0.152 at 45' of camera rotation)
    while retaining 105% of the ground truth's contrast where blending retains
    53%. Sharpness and contrast are what separate those two, so they are what
    is reported.

    Args:
        texture: ``(H, W, 3)`` atlas, ``uint8`` or float in [0, 1].
        covered_mask: ``(H, W)`` bool, True on texels the bake actually covered.
            Uncovered texels are excluded, and so are gradients that straddle
            the boundary between covered and uncovered -- a chart's edge
            against the fill color is a step of arbitrary size, and counting it
            would report the atlas's *layout* as detail. Defaults to all texels.

    Returns:
        A dict with ``num_covered_texels``, ``mean_gradient`` (mean absolute
        forward difference of luminance across covered texel pairs; higher =
        sharper), ``color_std`` (contrast of the covered texels), and
        ``mean_value``. Zero-filled when nothing is covered, matching the rest
        of this module's handling of empty input.
    """
    texture = np.asarray(texture)
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError(f"texture must be (H, W, 3), got shape {texture.shape}.")
    values = texture.astype(np.float64)
    if texture.dtype == np.uint8:
        values /= 255.0

    if covered_mask is None:
        covered = np.ones(values.shape[:2], dtype=bool)
    else:
        covered = np.asarray(covered_mask, dtype=bool)
        if covered.shape != values.shape[:2]:
            raise ValueError(
                f"covered_mask shape {covered.shape} does not match the "
                f"texture's {values.shape[:2]}."
            )

    num_covered = int(covered.sum())
    if num_covered == 0:
        return {
            "num_covered_texels": 0,
            "mean_gradient": 0.0,
            "color_std": 0.0,
            "mean_value": 0.0,
        }

    gray = values.mean(axis=2)
    # Only pairs where *both* texels are covered, so chart borders don't
    # masquerade as detail.
    pair_x = covered[:, :-1] & covered[:, 1:]
    pair_y = covered[:-1, :] & covered[1:, :]
    diff_total = float(
        np.abs(np.diff(gray, axis=1))[pair_x].sum()
        + np.abs(np.diff(gray, axis=0))[pair_y].sum()
    )
    num_pairs = int(pair_x.sum() + pair_y.sum())

    return {
        "num_covered_texels": num_covered,
        "mean_gradient": (diff_total / num_pairs) if num_pairs else 0.0,
        "color_std": float(values[covered].std()),
        "mean_value": float(values[covered].mean()),
    }
