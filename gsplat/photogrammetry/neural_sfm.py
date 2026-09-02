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
"""Import feed-forward neural-SfM output (DUSt3R/MASt3R/VGGT-style) as a
COLMAP model.

gsplat does not run any neural SfM model itself -- consistent with how
:mod:`gsplat.photogrammetry.dense_mvs` shells out to ``colmap`` rather than
reimplementing multi-view stereo, this module is a tool-agnostic *adapter*:
it consumes plain arrays (poses, per-image 3D point maps) that a user (or a
small tool-specific script they write) has already extracted from a neural
SfM tool's output, and turns them into a real COLMAP model that
``examples.datasets.colmap.Parser`` and, most usefully,
:func:`gsplat.photogrammetry.bundle_adjustment.refine_reconstruction` can
consume directly -- since feed-forward neural-SfM poses are typically less
precise than classical incremental SfM, running bundle adjustment on the
imported model is the expected next step.

Two functions, used in sequence:

- :func:`merge_point_maps_to_tracks` -- feed-forward tools like DUSt3R emit
  one dense 3D point *per pixel per image*, with no COLMAP-style track
  structure linking the same 3D point as seen by multiple images. A track of
  length 1 gives bundle adjustment no cross-view constraint at all, so this
  merges near-duplicate points across images (radius-based nearest-neighbor
  clustering) into shared tracks.
- :func:`write_colmap_reconstruction` -- builds a ``pycolmap.Reconstruction``
  from scratch (poses + merged tracks) and writes it to disk as a normal
  COLMAP sparse model.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def merge_point_maps_to_tracks(
    points_per_image: Sequence[np.ndarray],
    pixel_xy_per_image: Sequence[np.ndarray],
    colors_per_image: Optional[Sequence[np.ndarray]] = None,
    confidence_per_image: Optional[Sequence[np.ndarray]] = None,
    confidence_threshold: float = 0.0,
    merge_radius: float = 0.01,
    min_track_length: int = 1,
    max_points_per_image: Optional[int] = None,
) -> Dict[str, object]:
    """Merge per-image 3D point maps into cross-view COLMAP-style tracks.

    Args:
        points_per_image: One ``(N_i, 3)`` array per image of 3D point
            positions (already in a shared world frame, as produced by a
            neural SfM tool's own global alignment).
        pixel_xy_per_image: One ``(N_i, 2)`` array per image of the pixel
            coordinates each point came from.
        colors_per_image: Optional one ``(N_i, 3)`` array per image of point
            colors (any numeric dtype; averaged and cast to uint8 on merge).
        confidence_per_image: Optional one ``(N_i,)`` array per image of a
            per-point confidence score, as most feed-forward tools output.
        confidence_threshold: Points with confidence below this are dropped
            before merging (no-op if ``confidence_per_image`` is None).
        merge_radius: Points within this Euclidean distance (world units)
            across any images are merged into the same track.
        min_track_length: Drop merged tracks observed by fewer than this
            many images. Use >= 2 when the output will feed
            :func:`gsplat.photogrammetry.bundle_adjustment.refine_reconstruction`,
            since a track of length 1 contributes no cross-view constraint.
        max_points_per_image: If set, subsample each image down to this many
            points first (highest-confidence, if confidence is given; random
            otherwise) -- keeps the merge step tractable for dense point
            maps.

    Returns:
        A dict with ``"points_xyz"`` ``(T, 3)`` float64, ``"points_rgb"``
        ``(T, 3)`` uint8 (or None if no colors given), and ``"tracks"``: a
        list of length ``T``, each entry a list of ``(image_idx, (x, y))``
        observations for that merged point.
    """
    try:
        from scipy.sparse.csgraph import connected_components
        from sklearn.neighbors import radius_neighbors_graph
    except ImportError as e:
        raise ImportError(
            "merge_point_maps_to_tracks requires scipy and scikit-learn. "
            "Install them with `pip install scipy scikit-learn`."
        ) from e

    num_images = len(points_per_image)
    all_points, all_pixel_xy, all_image_idx, all_colors = [], [], [], []
    for i in range(num_images):
        pts = np.asarray(points_per_image[i], dtype=np.float64)
        pix = np.asarray(pixel_xy_per_image[i], dtype=np.float64)
        if pts.shape[0] != pix.shape[0]:
            raise ValueError(
                f"image {i}: points_per_image has {pts.shape[0]} points but "
                f"pixel_xy_per_image has {pix.shape[0]}."
            )
        idx = np.arange(pts.shape[0])
        if confidence_per_image is not None:
            conf = np.asarray(confidence_per_image[i])
            idx = idx[conf[idx] >= confidence_threshold]
        if max_points_per_image is not None and idx.size > max_points_per_image:
            if confidence_per_image is not None:
                conf = np.asarray(confidence_per_image[i])
                idx = idx[np.argsort(-conf[idx])[:max_points_per_image]]
            else:
                rng = np.random.default_rng(0)
                idx = rng.choice(idx, size=max_points_per_image, replace=False)

        all_points.append(pts[idx])
        all_pixel_xy.append(pix[idx])
        all_image_idx.append(np.full(idx.shape[0], i, dtype=np.int64))
        if colors_per_image is not None:
            all_colors.append(np.asarray(colors_per_image[i], dtype=np.float64)[idx])

    if sum(p.shape[0] for p in all_points) == 0:
        return {"points_xyz": np.zeros((0, 3)), "points_rgb": None, "tracks": []}

    points = np.concatenate(all_points, axis=0)
    pixel_xy = np.concatenate(all_pixel_xy, axis=0)
    image_idx = np.concatenate(all_image_idx, axis=0)
    colors = (
        np.concatenate(all_colors, axis=0) if colors_per_image is not None else None
    )

    graph = radius_neighbors_graph(
        points, radius=merge_radius, mode="connectivity", include_self=False
    )
    num_components, labels = connected_components(graph, directed=False)

    tracks: List[List[Tuple[int, Tuple[float, float]]]] = [
        [] for _ in range(num_components)
    ]
    for k in range(points.shape[0]):
        tracks[labels[k]].append(
            (int(image_idx[k]), (float(pixel_xy[k, 0]), float(pixel_xy[k, 1])))
        )

    merged_xyz = np.zeros((num_components, 3), dtype=np.float64)
    merged_rgb = (
        np.zeros((num_components, 3), dtype=np.uint8) if colors is not None else None
    )
    for c in range(num_components):
        mask = labels == c
        merged_xyz[c] = points[mask].mean(axis=0)
        if colors is not None:
            merged_rgb[c] = colors[mask].mean(axis=0).astype(np.uint8)

    if min_track_length > 1:
        keep = [c for c in range(num_components) if len(tracks[c]) >= min_track_length]
        merged_xyz = merged_xyz[keep]
        merged_rgb = merged_rgb[keep] if merged_rgb is not None else None
        tracks = [tracks[c] for c in keep]

    return {"points_xyz": merged_xyz, "points_rgb": merged_rgb, "tracks": tracks}


def write_colmap_reconstruction(
    image_names: Sequence[str],
    camtoworlds: np.ndarray,
    Ks: np.ndarray,
    image_sizes,
    points_xyz: np.ndarray,
    tracks: Sequence[Sequence[Tuple[int, Tuple[float, float]]]],
    output_dir: str,
    points_rgb: Optional[np.ndarray] = None,
) -> None:
    """Build and write a COLMAP sparse model from plain pose/point arrays.

    One pinhole camera is created per image (rather than assuming shared
    intrinsics), since neural SfM tools commonly predict per-image focal
    length. The result is a normal COLMAP model: point
    ``examples.datasets.colmap.Parser`` at ``output_dir`` via its
    ``colmap_dir`` argument to use it directly, or run
    :func:`gsplat.photogrammetry.bundle_adjustment.refine_reconstruction` on
    it first to clean up the (typically less precise) neural-SfM poses.

    Args:
        image_names: Image filenames, length ``N`` (must match the actual
            files under the dataset's ``images/`` directory).
        camtoworlds: ``(N, 4, 4)`` camera-to-world matrices.
        Ks: ``(N, 3, 3)`` per-image intrinsics, or a single ``(3, 3)`` shared
            by all images.
        image_sizes: A ``(width, height)`` tuple shared by all images, or a
            length-``N`` sequence of them.
        points_xyz: ``(P, 3)`` merged 3D point positions, e.g. from
            :func:`merge_point_maps_to_tracks`.
        tracks: Length-``P`` sequence, each entry a list of
            ``(image_idx, (x, y))`` observations for that point -- the same
            format :func:`merge_point_maps_to_tracks` returns.
        output_dir: Directory to write the COLMAP model to.
        points_rgb: Optional ``(P, 3)`` uint8 point colors.
    """
    import pycolmap

    num_images = len(image_names)
    camtoworlds = np.asarray(camtoworlds, dtype=np.float64)
    Ks = np.asarray(Ks, dtype=np.float64)
    if Ks.ndim == 2:
        Ks = np.stack([Ks] * num_images, axis=0)
    if isinstance(image_sizes, tuple):
        image_sizes = [image_sizes] * num_images

    recon = pycolmap.Reconstruction()
    for i in range(num_images):
        width, height = image_sizes[i]
        K = Ks[i]
        camera = pycolmap.Camera.create_from_model_id(
            i + 1,
            pycolmap.CameraModelId.PINHOLE,
            float(K[0, 0]),
            int(width),
            int(height),
        )
        camera.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        recon.add_camera_with_trivial_rig(camera)

    # Assign each track's per-image observations a sequential point2D index,
    # and collect them into each image's points2D list -- COLMAP requires
    # points2D to exist before a track can reference (image_id, point2D_idx).
    per_image_point2d: List[List[Tuple[float, float]]] = [[] for _ in range(num_images)]
    obs_point2d_idx: List[List[int]] = []
    for track in tracks:
        idxs = []
        for image_idx, xy in track:
            point2d_idx = len(per_image_point2d[image_idx])
            per_image_point2d[image_idx].append(xy)
            idxs.append(point2d_idx)
        obs_point2d_idx.append(idxs)

    image_ids = []
    for i in range(num_images):
        img = pycolmap.Image()
        img.image_id = i + 1
        img.camera_id = i + 1
        img.name = image_names[i]
        img.points2D = [pycolmap.Point2D(np.array(xy)) for xy in per_image_point2d[i]]

        w2c = np.linalg.inv(camtoworlds[i])
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3])
        recon.add_image_with_trivial_frame(img, cam_from_world)
        image_ids.append(img.image_id)

    for t_idx, track_obs in enumerate(tracks):
        pc_track = pycolmap.Track()
        for (image_idx, _xy), point2d_idx in zip(track_obs, obs_point2d_idx[t_idx]):
            pc_track.add_element(
                pycolmap.TrackElement(image_ids[image_idx], point2d_idx)
            )
        color = (
            np.asarray(points_rgb[t_idx], dtype=np.uint8)
            if points_rgb is not None
            else np.array([128, 128, 128], dtype=np.uint8)
        )
        recon.add_point3D(points_xyz[t_idx], pc_track, color)

    os.makedirs(output_dir, exist_ok=True)
    recon.write(output_dir)
