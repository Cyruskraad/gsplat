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

"""Tests for gsplat.photogrammetry.metrics against known analytic shapes (no
GPU/pycolmap needed, same style as tests/test_mesh_extraction.py):
`mesh_quality_stats` and `point_to_mesh_distance` against an Open3D unit
sphere, `point_cloud_stats` against a synthetic grid of known spacing.
"""

import numpy as np
import pytest

pytest.importorskip(
    "open3d", reason="open3d is not installed (pip install gsplat[mesh])"
)

import open3d as o3d

from gsplat.photogrammetry.metrics import (
    atlas_sharpness,
    mesh_quality_stats,
    point_cloud_stats,
    point_to_mesh_distance,
)


def _unit_sphere_mesh():
    return o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)


def test_mesh_quality_stats_on_sphere():
    mesh = _unit_sphere_mesh()
    stats = mesh_quality_stats(mesh)

    assert stats["num_vertices"] == len(mesh.vertices)
    assert stats["num_triangles"] == len(mesh.triangles)
    assert stats["is_watertight"] is True
    assert stats["num_connected_components"] == 1
    assert stats["surface_area"] == pytest.approx(4 * np.pi, rel=0.01)
    assert stats["volume"] == pytest.approx((4 / 3) * np.pi, rel=0.01)
    assert stats["mean_edge_length"] > 0
    assert (
        stats["min_edge_length"]
        <= stats["mean_edge_length"]
        <= stats["max_edge_length"]
    )


def test_mesh_quality_stats_empty_mesh_does_not_crash():
    mesh = o3d.geometry.TriangleMesh()
    stats = mesh_quality_stats(mesh)
    assert stats["num_vertices"] == 0
    assert stats["num_triangles"] == 0
    assert stats["is_watertight"] is False
    assert stats["volume"] is None


def test_point_to_mesh_distance_on_surface_points_are_near_zero():
    mesh = _unit_sphere_mesh()
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(300, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    on_surface = dirs * 1.0  # exactly on the unit sphere

    stats = point_to_mesh_distance(on_surface, mesh)
    assert stats["num_points"] == 300
    # The mesh is a polyhedral approximation of the sphere, not exact, but
    # should stay very close to the analytic surface at this resolution.
    assert stats["mean"] < 0.01
    assert stats["max"] < 0.02


def test_point_to_mesh_distance_matches_known_offset():
    mesh = _unit_sphere_mesh()
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(300, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    offset = 0.1
    points = dirs * (1.0 + offset)

    stats = point_to_mesh_distance(points, mesh)
    assert stats["mean"] == pytest.approx(offset, abs=0.01)
    assert stats["p50"] == pytest.approx(offset, abs=0.01)


def test_point_cloud_stats_grid_spacing():
    spacing = 0.5
    xs = np.arange(10) * spacing
    grid = np.stack(np.meshgrid(xs, xs, [0.0], indexing="ij"), axis=-1).reshape(-1, 3)

    stats = point_cloud_stats(grid, k=4)
    assert stats["num_points"] == 100
    assert stats["bbox_extent"] == pytest.approx([xs[-1], xs[-1], 0.0])
    # Each interior grid point's 4 nearest neighbors are its axis-aligned
    # neighbors, each exactly `spacing` away.
    assert stats["mean_knn_distance"] == pytest.approx(spacing, rel=0.1)
    assert stats["median_knn_distance"] == pytest.approx(spacing, abs=1e-6)


def test_point_cloud_stats_single_point():
    stats = point_cloud_stats(np.array([[1.0, 2.0, 3.0]]), k=4)
    assert stats["num_points"] == 1
    assert stats["mean_knn_distance"] == 0.0
    assert stats["median_knn_distance"] == 0.0


def test_point_cloud_stats_empty_cloud():
    """An empty cloud is a real outcome (dense MVS fusing nothing), and the
    rest of this module reports empty input as zeros rather than raising.

    Without this, numpy raised a bare "zero-size array to reduction operation
    minimum", which says nothing about which stage produced nothing.
    """
    stats = point_cloud_stats(np.zeros((0, 3)))
    assert stats["num_points"] == 0
    assert stats["bbox_extent"] == [0.0, 0.0, 0.0]
    assert stats["mean_knn_distance"] == 0.0
    assert stats["median_knn_distance"] == 0.0
    # Same keys as a populated cloud, so downstream readers don't special-case.
    assert set(stats) == set(point_cloud_stats(np.zeros((5, 3))))


def test_point_to_mesh_distance_rejects_an_empty_mesh():
    """A degenerate extraction must report why, not fail opaquely.

    open3d's raycaster fails on an empty scene with `IndexError:
    _Map_base::at`, which mesh extraction producing nothing would otherwise
    surface at the very end of a long run.
    """
    with pytest.raises(ValueError, match="no triangles"):
        point_to_mesh_distance(np.zeros((5, 3)), o3d.geometry.TriangleMesh())


def test_point_to_mesh_distance_with_no_points_reports_nothing_measured():
    """No points measured must not read as a perfect fit.

    `0.0` here would mean "every point lies exactly on the mesh" -- and would
    feed a fake `mesh_fit_over_point_spacing` into the cross-stage metrics.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    stats = point_to_mesh_distance(np.zeros((0, 3)), sphere)
    assert stats["num_points"] == 0
    assert stats["mean"] is None and stats["rms"] is None and stats["max"] is None
    # Percentile keys are present too, so the schema doesn't change shape.
    populated = point_to_mesh_distance(np.zeros((5, 3)), sphere)
    assert set(stats) == set(populated)


# ---------------------------------------------------------------------------
# atlas_sharpness
# ---------------------------------------------------------------------------


def test_atlas_sharpness_rises_with_detail():
    """A checkerboard is sharper than a ramp is sharper than a flat field."""
    size = 64
    flat = np.full((size, size, 3), 0.5)
    ramp = np.tile(np.linspace(0.0, 1.0, size)[None, :, None], (size, 1, 3))
    checker = np.indices((size, size)).sum(axis=0) % 2
    checker = np.repeat(checker[..., None].astype(float), 3, axis=2)

    gradients = [atlas_sharpness(t)["mean_gradient"] for t in (flat, ramp, checker)]
    assert gradients[0] == pytest.approx(0.0)
    assert gradients[0] < gradients[1] < gradients[2]
    # Contrast tracks the same ordering but is a different quantity: the ramp
    # and the checkerboard span the same range, so only the gradient separates
    # them, which is why both are reported.
    assert atlas_sharpness(flat)["color_std"] == pytest.approx(0.0)


def test_atlas_sharpness_ignores_the_edge_of_the_covered_region():
    """A chart's border against the fill color is layout, not detail.

    The step from a baked texel to an unbaked one is arbitrary in size, and
    counting it would report a *more fragmented* UV atlas as a *sharper* one.
    """
    size = 32
    texture = np.zeros((size, size, 3))
    covered = np.zeros((size, size), dtype=bool)
    # A flat, bright square on a black (unbaked) background: zero real detail,
    # but a large step all the way round its edge.
    texture[8:24, 8:24] = 0.9
    covered[8:24, 8:24] = True

    assert atlas_sharpness(texture, covered)["mean_gradient"] == pytest.approx(0.0)
    # Premise: without the mask that same step is reported as detail, so the
    # assertion above is not vacuous.
    assert atlas_sharpness(texture)["mean_gradient"] > 0.01


def test_atlas_sharpness_handles_an_uncovered_atlas():
    """Nothing baked is a real outcome, and returns zeros like the rest of the
    module rather than dividing by an empty count."""
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    stats = atlas_sharpness(texture, np.zeros((8, 8), dtype=bool))
    assert stats["num_covered_texels"] == 0
    assert stats["mean_gradient"] == 0.0 and stats["color_std"] == 0.0
    # Same keys as a populated result, so the report's schema doesn't change.
    assert set(stats) == set(atlas_sharpness(texture))


def test_atlas_sharpness_reads_uint8_and_float_alike():
    """uint8 atlases (what the bakers return) and floats must agree."""
    rng = np.random.default_rng(0)
    as_uint8 = (rng.random((32, 32, 3)) * 255).round().astype(np.uint8)
    as_float = as_uint8 / 255.0
    a, b = atlas_sharpness(as_uint8), atlas_sharpness(as_float)
    assert a["mean_gradient"] == pytest.approx(b["mean_gradient"])
    assert a["color_std"] == pytest.approx(b["color_std"])


def test_atlas_sharpness_rejects_a_mismatched_mask():
    with pytest.raises(ValueError, match="does not match"):
        atlas_sharpness(np.zeros((8, 8, 3)), np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
        atlas_sharpness(np.zeros((8, 8)))
