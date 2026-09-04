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

"""Tests for gsplat.photogrammetry.texturing's per-face view selection.

Everything here runs on synthetic data whose right answer is known
independently -- hand-built adjacency, hand-built quality matrices, and the
analytic sphere dataset from tests/test_mesh_extraction.py.
"""

import numpy as np
import pytest

pytest.importorskip(
    "open3d", reason="open3d is not installed (pip install gsplat[mesh])"
)

import open3d as o3d

from gsplat.photogrammetry.texturing import (
    NO_VIEW,
    _face_adjacency,
    _gradient_summed_area,
    face_view_quality,
    select_views_mrf,
)


# ---------------------------------------------------------------------------
# Face adjacency
# ---------------------------------------------------------------------------


def test_face_adjacency_finds_shared_edges():
    """Two triangles sharing an edge are adjacent; disjoint ones are not."""
    # 0-1-2 and 1-2-3 share edge (1, 2).
    shared = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    pairs = _face_adjacency(shared)
    assert pairs.shape == (1, 2)
    assert set(pairs[0]) == {0, 1}

    # Disjoint triangles share nothing.
    disjoint = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    assert len(_face_adjacency(disjoint)) == 0


def test_face_adjacency_on_a_closed_mesh_matches_eulers_formula():
    """A closed manifold mesh has exactly 3F/2 interior edges.

    Checks the adjacency against a known identity rather than against itself:
    every triangle has 3 edges and every edge is shared by exactly 2 faces.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
    triangles = np.asarray(sphere.triangles)
    pairs = _face_adjacency(triangles)
    assert len(pairs) == 3 * len(triangles) // 2
    # No face is ever paired with itself, and pairs are unique.
    assert (pairs[:, 0] != pairs[:, 1]).all()
    canonical = {tuple(sorted(p)) for p in pairs}
    assert len(canonical) == len(pairs)


# ---------------------------------------------------------------------------
# MRF labelling
# ---------------------------------------------------------------------------


def test_mrf_picks_the_best_view_when_there_are_no_seams_to_pay_for():
    """With every face preferring the same view, that view must win outright."""
    quality = np.array([[0.1, 9.0], [0.2, 9.0], [0.3, 9.0]])
    adjacency = np.array([[0, 1], [1, 2]])
    labels, stats = select_views_mrf(quality, adjacency, smoothness=1.0)
    assert labels.tolist() == [1, 1, 1]
    assert stats["num_seams"] == 0
    assert stats["num_unlabelled"] == 0
    assert stats["num_views_used"] == 1


def test_mrf_trades_quality_against_seams_as_smoothness_rises():
    """Raising the seam penalty must not increase the seam count.

    One face mildly prefers a different view from its neighbours. At low
    smoothness the MRF should indulge it (a seam); at high smoothness it should
    conform. Monotonicity is the property worth pinning -- it is what says the
    energy's two terms have the right signs relative to each other.
    """
    # Faces 0 and 2 clearly prefer view 0; face 1 mildly prefers view 1.
    quality = np.array([[9.0, 0.1], [1.0, 1.6], [9.0, 0.1]])
    adjacency = np.array([[0, 1], [1, 2]])

    seams = [
        select_views_mrf(quality, adjacency, smoothness=s)[1]["num_seams"]
        for s in (0.0, 0.1, 0.5, 2.0, 10.0)
    ]
    assert seams == sorted(seams, reverse=True), seams
    # Premise: the low-smoothness end must actually produce a seam, or the
    # monotonicity above is vacuously satisfied by an all-zero list.
    assert seams[0] > 0, seams
    assert seams[-1] == 0, seams

    # And at high smoothness the dissenting face has conformed.
    labels, _ = select_views_mrf(quality, adjacency, smoothness=10.0)
    assert labels.tolist() == [0, 0, 0]


def test_mrf_never_picks_a_view_that_cannot_see_the_face():
    """Zero quality means "cannot texture this", not merely "expensive".

    A finite penalty could be outweighed by enough smoothness pressure, which
    would texture a face from a camera that never saw it. Both faces here have
    exactly one usable view, and those differ, so smoothness pressure is
    maximal -- yet neither may switch.
    """
    quality = np.array([[5.0, 0.0], [0.0, 5.0]])
    adjacency = np.array([[0, 1]])
    labels, stats = select_views_mrf(quality, adjacency, smoothness=1e6)
    assert labels.tolist() == [0, 1]
    assert stats["num_seams"] == 1


def test_mrf_marks_faces_no_view_can_texture():
    quality = np.array([[5.0, 1.0], [0.0, 0.0], [1.0, 5.0]])
    adjacency = np.array([[0, 1], [1, 2]])
    labels, stats = select_views_mrf(quality, adjacency, smoothness=0.5)
    assert labels[1] == NO_VIEW
    assert stats["num_unlabelled"] == 1
    assert (labels[[0, 2]] != NO_VIEW).all()


def test_mrf_is_deterministic_and_terminates():
    """Same input, same labels -- a labelling nobody can reproduce is not a
    reviewable artifact, and ICM must converge rather than cycle."""
    rng = np.random.default_rng(0)
    quality = rng.random((60, 5))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=4)
    adjacency = _face_adjacency(np.asarray(sphere.triangles))[:60]

    first, stats_a = select_views_mrf(quality, adjacency, smoothness=0.3)
    second, stats_b = select_views_mrf(quality, adjacency, smoothness=0.3)
    np.testing.assert_array_equal(first, second)
    assert stats_a == stats_b
    # Converged before the cap, i.e. a sweep changed nothing.
    assert stats_a["iterations"] < 20


def test_mrf_handles_a_mesh_with_no_adjacency():
    """Disconnected faces have no seams to pay for, so each takes its best."""
    quality = np.array([[1.0, 4.0], [4.0, 1.0]])
    labels, stats = select_views_mrf(quality, np.zeros((0, 2), dtype=np.int64))
    assert labels.tolist() == [1, 0]
    assert stats["num_seams"] == 0


# ---------------------------------------------------------------------------
# Quality term
# ---------------------------------------------------------------------------


def test_gradient_summed_area_matches_a_direct_sum():
    """The summed-area table must agree with the obvious slow computation."""
    rng = np.random.default_rng(1)
    image = rng.random((16, 20, 3))
    table = _gradient_summed_area(image)

    gray = image.mean(axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, :-1] = np.abs(np.diff(gray, axis=1))
    grad_y[:-1, :] = np.abs(np.diff(gray, axis=0))
    magnitude = grad_x + grad_y

    for y0, y1, x0, x1 in [(0, 16, 0, 20), (2, 9, 3, 11), (5, 6, 7, 8)]:
        expected = magnitude[y0:y1, x0:x1].sum()
        got = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
        assert got == pytest.approx(expected)


def test_face_view_quality_prefers_the_closer_view():
    """A camera nearer the surface projects it larger, so must score higher.

    Ground truth here is geometric and independent of the implementation: the
    same face, the same image content, two distances.
    """
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mesh_extraction import _SphereDataset, _unit_sphere_mesh

    mesh = _unit_sphere_mesh(resolution=6)
    near = _SphereDataset(num_views=6, cam_dist=2.5)
    far = _SphereDataset(num_views=6, cam_dist=6.0)

    near_quality = face_view_quality(mesh, near)
    far_quality = face_view_quality(mesh, far)

    assert near_quality.shape == (len(mesh.triangles), 6)
    # Compare only faces both distances can actually see.
    both = (near_quality.max(axis=1) > 0) & (far_quality.max(axis=1) > 0)
    assert both.sum() > 10, "test needs faces visible at both distances"
    assert near_quality[both].max(axis=1).mean() > far_quality[both].max(axis=1).mean()


def test_face_view_quality_is_zero_where_a_view_cannot_see_the_face():
    """Occluded and back-facing faces must score exactly zero, not merely low.

    Every face of a closed sphere is invisible from most cameras, so a
    quality matrix with no zeros would mean the visibility test was not
    consulted at all.
    """
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mesh_extraction import _SphereDataset, _unit_sphere_mesh

    mesh = _unit_sphere_mesh(resolution=8)
    quality = face_view_quality(mesh, _SphereDataset(num_views=8))

    assert (quality >= 0).all()
    assert (quality == 0).any(), "no face was ever occluded -- visibility ignored?"
    # A closed sphere hides roughly half of itself from any one camera.
    assert 0.2 < (quality == 0).mean() < 0.9, (quality == 0).mean()
    # But every face is seen by something, from 8 views around the sphere.
    assert (quality.max(axis=1) > 0).all()


def test_face_view_quality_rejects_an_empty_mesh():
    with pytest.raises(ValueError, match="no triangles"):
        face_view_quality(o3d.geometry.TriangleMesh(), [])


def test_mrf_multi_seed_escapes_a_bad_greedy_minimum():
    """Multi-seed ICM must beat single-seed on the case that motivated it.

    Swept only from the per-face best view, a strong smoothness term lets the
    first face to move cascade every other face onto its neighbour's label --
    landing on a labelling far worse than one that was available. Seeding also
    from "every face takes view alpha" reaches it. Pinning the energy gap here
    stops the extra seeds being removed as redundant.
    """
    quality = np.array([[9.0, 0.1], [1.0, 1.6], [9.0, 0.1]])
    adjacency = np.array([[0, 1], [1, 2]])

    single, single_stats = select_views_mrf(
        quality, adjacency, smoothness=10.0, max_seeds=0
    )
    multi, multi_stats = select_views_mrf(quality, adjacency, smoothness=10.0)

    # Premise: the single-seed sweep really does go wrong here.
    assert single.tolist() == [1, 1, 1], single
    assert multi.tolist() == [0, 0, 0], multi
    # And the win is in the energy the optimiser is actually minimising.
    assert multi_stats["energy"] < single_stats["energy"]
    assert multi_stats["energy"] == pytest.approx(-4.394, abs=1e-2)
    assert multi_stats["num_seeds"] > 1

    # Extra seeds may never make the result worse, on any smoothness.
    for smoothness in (0.0, 0.25, 1.0, 4.0):
        _, few = select_views_mrf(
            quality, adjacency, smoothness=smoothness, max_seeds=0
        )
        _, many = select_views_mrf(quality, adjacency, smoothness=smoothness)
        assert many["energy"] <= few["energy"] + 1e-9, (smoothness, few, many)
