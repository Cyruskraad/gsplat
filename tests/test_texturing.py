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

import os

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


# ---------------------------------------------------------------------------
# View-selected atlas bake
# ---------------------------------------------------------------------------


def _high_frequency_pattern(points):
    """A surface color whose detail sits near the misregistration scale.

    ``tests/test_mesh_extraction.py``'s ``_surface_pattern`` has a wavelength of
    roughly half the sphere -- far coarser than the few pixels a residual pose
    error displaces a projection by -- so blending several views of it barely
    blurs it at all, and a detail-retention test written against it measures
    nothing. This one has a wavelength of ~0.14 world units, about three times
    the displacement a 45' pose error produces at this camera distance, which
    is where averaging misaligned views actually costs something.
    """
    points = np.asarray(points, dtype=np.float64)
    return np.stack(
        [
            0.5 + 0.45 * np.sin(45.0 * points[..., 0]),
            0.5 + 0.45 * np.sin(43.0 * points[..., 1]),
            0.5 + 0.45 * np.sin(47.0 * points[..., 2]),
        ],
        axis=-1,
    )


def _sphere_fixtures():
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mesh_extraction import _SphereDataset, _unit_sphere_mesh

    return _SphereDataset, _unit_sphere_mesh


def _default_pattern():
    """tests/test_mesh_extraction.py's analytic sphere colour."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mesh_extraction import _surface_pattern

    return _surface_pattern


def _ground_truth_atlas(mesh, texture_size, pattern):
    """The pattern evaluated at each texel's own surface point.

    Reuses the mesh's existing UV layout (``_unwrap_and_rasterize`` reuses
    ``triangle_uvs`` when present), so this is directly comparable to an atlas
    baked on that same mesh. Unwrapping a second mesh instead would produce a
    *different* layout -- ``compute_uvatlas`` is non-deterministic -- and every
    comparison against it would be meaningless.
    """
    from gsplat.photogrammetry.texturing import _unwrap_and_rasterize

    atlas = _unwrap_and_rasterize(mesh, texture_size)
    image = np.zeros((texture_size, texture_size, 3))
    covered = np.zeros((texture_size, texture_size), dtype=bool)
    unit = atlas.positions / np.linalg.norm(atlas.positions, axis=1, keepdims=True)
    image[atlas.rows, atlas.cols] = pattern(unit)
    covered[atlas.rows, atlas.cols] = True
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8), covered


def test_view_selection_retains_detail_that_blending_destroys():
    """The whole point of the feature, measured the only way that shows it.

    **Do not "fix" this test into an error comparison.** Pointwise error goes
    the *other* way and is asserted below as such: blending attenuates detail
    while single-view sampling displaces it, so a blurred atlas scores better
    pointwise than a sharp but slightly-shifted one, even though it looks far
    worse. Gradient retention is what separates them.
    """
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas,
        bake_texture_atlas_view_selected,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    size = 256
    dataset = _SphereDataset(
        num_views=16, pattern=_high_frequency_pattern, pose_error_arcmin=45.0
    )
    # One mesh, so both bakes and the ground truth share a single UV layout.
    mesh = _unit_sphere_mesh(resolution=10)
    _, blended = bake_texture_atlas(mesh, dataset, texture_size=size)
    _, selected, stats = bake_texture_atlas_view_selected(
        mesh, dataset, texture_size=size
    )
    truth, covered = _ground_truth_atlas(mesh, size, _high_frequency_pattern)

    truth_grad = atlas_sharpness(truth, covered)["mean_gradient"]
    blended_grad = atlas_sharpness(blended, covered)["mean_gradient"]
    selected_grad = atlas_sharpness(selected, covered)["mean_gradient"]

    # Premise: blending must actually be losing detail here, or everything
    # below is vacuous. Measured ~59% at this perturbation.
    assert blended_grad / truth_grad < 0.70, blended_grad / truth_grad
    # And view selection keeps essentially all of it. Measured ~106% -- above
    # 100% because sampling one photograph also carries that photograph's own
    # pixel noise, which blending averages away along with the detail.
    assert selected_grad / truth_grad > 0.95, selected_grad / truth_grad
    assert selected_grad > blended_grad

    # The bake reports the same verdict from its own stats, without a caller
    # having to build a ground-truth atlas.
    assert (
        stats["atlas_sharpness"]["mean_gradient"]
        > stats["blended_atlas_sharpness"]["mean_gradient"]
    )

    # The tradeoff, pinned deliberately: pointwise it is *worse*, and that is
    # expected. If this ever fails because view selection became pointwise
    # better, that is good news -- update the docs' tradeoff claim rather than
    # deleting the assertion.
    truth_f = truth[covered] / 255.0
    l1_blended = np.abs(blended[covered] / 255.0 - truth_f).mean()
    l1_selected = np.abs(selected[covered] / 255.0 - truth_f).mean()
    assert l1_selected > l1_blended, (l1_selected, l1_blended)


def test_view_selected_bake_falls_back_to_the_blend_it_cannot_improve():
    """Texels no single chosen view can see keep the blended color.

    Not a hole and not black: a handful of averaged views is the right answer
    where one view is unavailable, and the blend is computed anyway.
    """
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas,
        bake_texture_atlas_view_selected,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    size = 192
    dataset = _SphereDataset(num_views=16)
    mesh = _unit_sphere_mesh(resolution=10)
    _, blended = bake_texture_atlas(mesh, dataset, texture_size=size, max_views=3)
    _, selected, stats = bake_texture_atlas_view_selected(
        mesh, dataset, texture_size=size, max_views=3
    )

    # Premise: with only three views this must exercise both paths -- some
    # faces no view can texture at all, and some texels whose face's chosen
    # view cannot see them.
    assert stats["mrf"]["num_unlabelled"] > 0, stats["mrf"]
    assert stats["num_texels_blended"] > 0, stats
    assert stats["num_texels_view_selected"] > 0, stats
    assert 0.0 < stats["fallback_fraction"] < 0.2, stats["fallback_fraction"]

    # Coverage is *exactly* the blend's: view selection never drops a texel the
    # blended bake had. Checked against an independent run of the blended
    # sampler rather than against the atlas's pixel values -- a texel is
    # legitimately black wherever a view saw past the sphere's silhouette, so
    # "is it black" is not a coverage test.
    import open3d as o3d_

    from gsplat.photogrammetry.texturing import (
        _bake_points_from_views,
        _face_adjacency,
        _texel_face_ids,
        _unwrap_and_rasterize,
        face_view_quality,
        select_views_mrf,
    )

    atlas = _unwrap_and_rasterize(mesh, size)
    _, weights = _bake_points_from_views(
        mesh, dataset, atlas.positions, atlas.normals, max_views=3
    )
    covered = weights > 0
    assert int(covered.sum()) == (
        stats["num_texels_view_selected"] + stats["num_texels_blended"]
    )

    # Now isolate the fallback texels and check what they actually contain.
    # Their inputs are recomputed here from the same public pieces the bake
    # composes -- the labelling and the per-texel face id -- rather than
    # re-deriving the bake's output, so this pins the *composition*: a face no
    # view can texture must keep the blend, byte for byte.
    face_ids, has_face, _ = _texel_face_ids(o3d_, mesh, atlas.positions, atlas.normals)
    labels, _ = select_views_mrf(
        face_view_quality(mesh, dataset, max_views=3),
        _face_adjacency(np.asarray(mesh.triangles)),
    )
    unlabelled = np.zeros(len(face_ids), dtype=bool)
    unlabelled[has_face] = labels[face_ids[has_face]] == NO_VIEW
    fell_back = covered & unlabelled

    # Premise: with three views there really are covered texels on faces the
    # MRF could not label, or the assertion below is vacuous.
    assert fell_back.sum() > 20, int(fell_back.sum())
    rows, cols = atlas.rows[fell_back], atlas.cols[fell_back]
    np.testing.assert_array_equal(selected[rows, cols], blended[rows, cols])
    # And they are not merely equal because both are black.
    assert selected[rows, cols].max() > 0


def test_view_selected_albedo_normal_and_ao_share_one_uv_layout():
    """All three maps must be addressable by the same UVs.

    ``compute_uvatlas`` is non-deterministic, so a second unwrap would silently
    give the normal map a different layout from the albedo and the asset would
    be wrong in a way nothing raises about. The blended path is already pinned
    this way in tests/test_mesh_extraction.py; the view-selected path takes a
    different route to the atlas and needs its own guard.
    """
    from gsplat.photogrammetry.texturing import (
        bake_ambient_occlusion,
        bake_normal_map,
        bake_texture_atlas_view_selected,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    size = 128
    mesh = _unit_sphere_mesh(resolution=8)
    dataset = _SphereDataset(num_views=8)

    mesh, albedo, _ = bake_texture_atlas_view_selected(mesh, dataset, texture_size=size)
    albedo_uvs = np.asarray(mesh.triangle_uvs).copy()

    mesh, normal_map, _ = bake_normal_map(mesh, mesh, texture_size=size)
    np.testing.assert_array_equal(np.asarray(mesh.triangle_uvs), albedo_uvs)

    mesh, ao_map, _ = bake_ambient_occlusion(mesh, texture_size=size, num_samples=16)
    np.testing.assert_array_equal(np.asarray(mesh.triangle_uvs), albedo_uvs)

    assert albedo.shape == (size, size, 3)
    assert normal_map.shape[:2] == (size, size)
    assert ao_map.shape[:2] == (size, size)


def test_view_selected_bake_is_deterministic():
    """Same mesh, same dataset, same atlas -- twice."""
    from gsplat.photogrammetry.texturing import bake_texture_atlas_view_selected

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    dataset = _SphereDataset(num_views=8)
    mesh = _unit_sphere_mesh(resolution=8)

    _, first, stats_a = bake_texture_atlas_view_selected(
        mesh, dataset, texture_size=128
    )
    _, second, stats_b = bake_texture_atlas_view_selected(
        mesh, dataset, texture_size=128
    )
    np.testing.assert_array_equal(first, second)
    assert stats_a["mrf"] == stats_b["mrf"]


def test_box_means_never_reports_a_negative_mean():
    """A mean of gradient magnitudes cannot be negative -- and must not be.

    The four-corner readout of a cumulative sum loses precision on a
    nearly-empty box in a large table and lands below zero. That value reaches
    ``-log()`` in :func:`select_views_mrf`, becomes ``NaN``, and ``np.argmin``
    prefers a ``NaN`` to every real cost -- so the face would be textured from
    the one view that cannot see it, and the seed search's ``energy <
    best_energy`` comparison would stop working too (every comparison against
    ``NaN`` is False).
    """
    from gsplat.photogrammetry.texturing import _box_means, _gradient_summed_area

    rng = np.random.default_rng(0)
    # A detailed disc on an empty frame -- the shape of every image this runs
    # on, a lit subject against background. Boxes read out of the flat parts
    # cancel four large partial sums whose true difference is zero, and land
    # either side of it.
    size = 512
    ys, xs = np.mgrid[0:size, 0:size]
    disc = ((ys - size / 2) ** 2 + (xs - size / 2) ** 2) < (0.4 * size) ** 2
    image = np.zeros((size, size, 3))
    image[disc] = rng.random((int(disc.sum()), 3))
    table = _gradient_summed_area(image)

    y0, x0 = np.mgrid[0 : size - 1, 0 : size - 1].reshape(2, -1)
    y1, x1 = y0 + 1, x0 + 1
    raw = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    # Premise: the unclamped readout really does go negative, so the assertion
    # below is not vacuous. (A handful of texels out of 260k -- rare, but the
    # consequence is a NaN that wins every argmin it appears in.)
    assert raw.min() < 0, raw.min()
    assert (_box_means(table, x0, y0, x1, y1) >= 0).all()


def test_gradient_summed_area_is_accumulated_in_double_precision():
    """float32 images must not cost precision in the table.

    Training images arrive as float32 (that is what the dataset yields), and a
    cumulative sum over 200k entries in float32 leaves box readouts wrong by
    ~6e-5 -- the same order as the gradient magnitude over a flat patch, so the
    (face, view) quality it feeds is wrong exactly where views are hardest to
    tell apart.
    """
    from gsplat.photogrammetry.texturing import _gradient_summed_area

    rng = np.random.default_rng(4)
    image = rng.random((512, 512, 3))
    exact = _gradient_summed_area(image)
    from_float32 = _gradient_summed_area(image.astype(np.float32))

    # float32 *input* is fine -- the values differ by float32's own rounding of
    # the pixels. What must not happen is that error compounding across 260k
    # accumulation steps: accumulating in float32 puts this at 0.087, four
    # orders of magnitude worse than the 1.1e-5 the input rounding alone costs.
    assert np.abs(from_float32 - exact).max() < 1e-3, np.abs(from_float32 - exact).max()


def test_bake_mesh_texture_dispatches_to_view_selection_and_reports_it():
    """The CLIs' one entry point must reach the new path and hand back stats.

    ``stats_out`` is an out-parameter because several callers unpack this
    function's ``(mesh, texture)`` pair; a test that it is actually filled is
    what stops that contract quietly rotting.
    """
    from gsplat.photogrammetry.texturing import bake_mesh_texture

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    dataset = _SphereDataset(num_views=8)

    stats: dict = {}
    mesh, texture = bake_mesh_texture(
        _unit_sphere_mesh(resolution=8),
        dataset,
        mode="atlas",
        texture_size=128,
        view_selection=True,
        stats_out=stats,
    )
    assert texture.shape == (128, 128, 3)
    assert stats["mrf"]["num_faces"] == len(mesh.triangles)
    assert stats["num_texels_view_selected"] > 0

    # Blending leaves the out-dict untouched, so a caller can use "did I get
    # stats" to mean "did view selection actually run".
    blended_stats: dict = {}
    bake_mesh_texture(
        _unit_sphere_mesh(resolution=8),
        dataset,
        mode="atlas",
        texture_size=128,
        stats_out=blended_stats,
    )
    assert blended_stats == {}


def test_bake_mesh_texture_forwards_seam_smoothness_to_the_levelling():
    """The dispatcher must *carry* --texture_seam_smoothness, not just accept it.

    This pins the call site that shipped broken. ``bake_mesh_texture`` had no
    ``seam_smoothness`` parameter at all while ``examples/extract_mesh.py``
    passed one unconditionally, so every texture-baking run of that script --
    and therefore ``run_pipeline.py``'s whole delivery stage -- died with
    ``TypeError: bake_mesh_texture() got an unexpected keyword argument
    'seam_smoothness'``. Seam levelling was reachable from the library but not
    from the CLI, and nothing noticed because nothing drove either.

    Accepting the argument is not enough: silently dropping it would make the
    crash go away while leaving ``None`` (levelling off) indistinguishable from
    ``0.1`` (levelling on), because the callee supplies its own default. So the
    assertion is on the *effect* -- ``None`` must reach
    :func:`bake_texture_atlas_view_selected` and suppress the levelling stats
    that any other value produces.
    """
    from gsplat.photogrammetry.texturing import bake_mesh_texture

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    # Exposure drift is what levelling exists to remove; without it the solve
    # has almost nothing to do and the stats are less clearly attributable.
    dataset = _SphereDataset(num_views=8, exposure=0.15)

    levelled: dict = {}
    bake_mesh_texture(
        _unit_sphere_mesh(resolution=8),
        dataset,
        mode="atlas",
        texture_size=128,
        view_selection=True,
        seam_smoothness=0.1,
        stats_out=levelled,
    )
    assert "seam_levelling" in levelled
    assert levelled["seam_levelling"]["num_seam_edges"] > 0

    unlevelled: dict = {}
    bake_mesh_texture(
        _unit_sphere_mesh(resolution=8),
        dataset,
        mode="atlas",
        texture_size=128,
        view_selection=True,
        seam_smoothness=None,
        stats_out=unlevelled,
    )
    # Dropped on the floor instead of forwarded, this key would be present:
    # the callee's own default (0.1) would have levelled anyway.
    assert "seam_levelling" not in unlevelled

    # And it is accepted-and-ignored without view selection, the same contract
    # mrf_smoothness already has -- raising here would resurrect the crash on
    # the default blended path, which is what the CLI actually runs.
    blended: dict = {}
    _, texture = bake_mesh_texture(
        _unit_sphere_mesh(resolution=8),
        dataset,
        mode="atlas",
        texture_size=128,
        seam_smoothness=0.1,
        stats_out=blended,
    )
    assert texture.shape == (128, 128, 3)
    assert blended == {}


# ---------------------------------------------------------------------------
# Seam levelling
# ---------------------------------------------------------------------------


def test_conjugate_gradient_matches_a_dense_solve():
    """The hand-rolled solver against `np.linalg.solve` on a small system.

    Written by hand because `gsplat[mesh]` is deliberately just open3d and
    imageio, so scipy's `cg` is not available as a hard dependency. That makes
    an independent check of it non-optional.
    """
    from gsplat.photogrammetry.texturing import _conjugate_gradient

    rng = np.random.default_rng(5)
    root = rng.random((12, 12))
    matrix = root @ root.T + 0.5 * np.eye(12)  # symmetric positive definite
    rhs = rng.random((12, 3))

    solved, stats = _conjugate_gradient(
        lambda x: matrix @ x, rhs, max_iterations=200, tol=1e-12
    )
    np.testing.assert_allclose(solved, np.linalg.solve(matrix, rhs), atol=1e-8)
    assert stats["converged"]
    # CG on an n x n system is exact in at most n steps in exact arithmetic;
    # the extra iteration is the one that *observes* the residual collapsed,
    # since the tolerance is checked after each update rather than before.
    assert stats["iterations"] <= 13


def test_conjugate_gradient_anchors_a_singular_system():
    """The seam system is singular along the constants, and must be anchored.

    Its energy only sees *differences* of corrections, so adding a constant to
    every unknown changes nothing and the solution is a whole line. Without an
    anchor the solve drifts somewhere along it; `project_mean` picks the
    zero-mean point, which is the one that shifts the atlas least.
    """
    from gsplat.photogrammetry.texturing import _conjugate_gradient

    # A path graph's Laplacian: exactly the structure of the smoothness term,
    # singular along the constant vector.
    laplacian = np.array(
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]],
    )
    rhs = np.array([[1.0], [0.0], [-1.0]])  # consistent: sums to zero

    solved, stats = _conjugate_gradient(
        lambda x: laplacian @ x, rhs, project_mean=True, tol=1e-12
    )
    assert abs(float(solved.mean())) < 1e-9, solved
    # It still solves the system, up to that constant.
    np.testing.assert_allclose(laplacian @ solved, rhs, atol=1e-8)


def test_shared_edge_vertices_finds_the_two_vertices_of_each_shared_edge():
    from gsplat.photogrammetry.texturing import (
        _face_adjacency,
        _shared_edge_vertices,
    )

    triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    pairs, shared = _shared_edge_vertices(triangles, _face_adjacency(triangles))
    assert len(pairs) == 1
    assert set(shared[0]) == {1, 2}

    # Every interior edge of a closed mesh yields exactly two vertices.
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    triangles = np.asarray(sphere.triangles)
    pairs, shared = _shared_edge_vertices(triangles, _face_adjacency(triangles))
    assert shared.shape == (len(pairs), 2)
    assert (shared[:, 0] != shared[:, 1]).all()
    # And each shared vertex really is a corner of both faces.
    for (face_a, face_b), (v0, v1) in zip(pairs, shared):
        assert {v0, v1} <= set(triangles[face_a]) & set(triangles[face_b])


def test_seam_levelling_closes_the_steps_between_views():
    """Per-view exposure offsets make real seams; levelling must remove them.

    Each view reports the (unambiguous) surface colour shifted by its own
    constant, so best-view labelling produces genuine steps wherever
    neighbouring faces chose different photographs.
    """
    from gsplat.photogrammetry.texturing import bake_texture_atlas_view_selected

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    size = 256
    dataset = _SphereDataset(num_views=16, exposure=0.15)
    mesh = _unit_sphere_mesh(resolution=10)
    _, levelled, stats = bake_texture_atlas_view_selected(
        mesh, dataset, texture_size=size
    )

    before = stats["seam_discontinuity_before"]["mean"]
    after = stats["seam_discontinuity"]["mean"]

    # Premise 1: there are seams to level at all.
    assert stats["mrf"]["num_seams"] > 10, stats["mrf"]
    assert stats["seam_levelling"]["num_seam_edges"] > 10, stats["seam_levelling"]
    # Premise 2: the exposure offsets really do show up as steps. The metric
    # has a floor -- two samples either side of a border are different surface
    # points -- so this is measured against the *same scene without exposure
    # differences* rather than against zero.
    clean_mesh = _unit_sphere_mesh(resolution=10)
    _, _, clean = bake_texture_atlas_view_selected(
        clean_mesh, _SphereDataset(num_views=16), texture_size=size
    )
    floor = clean["seam_discontinuity_before"]["mean"]
    assert before > 1.3 * floor, (before, floor)

    # The steps are largely gone: measured ~2.1x on this scene.
    assert before / after > 1.5, (before, after)
    assert stats["seam_levelling"]["converged"], stats["seam_levelling"]

    # The correction is gauge-anchored: its mean is zero, so levelling closes
    # the seams without also shifting the whole atlas -- the energy only sees
    # differences of corrections, so the solution is a whole line and which
    # point on it is returned decides whether the atlas gains a colour cast.
    # (Measured: the projection is not what achieves this here, since the seam
    # system's right-hand side is already orthogonal to the constants. This
    # asserts the property, not the mechanism.)
    from gsplat.photogrammetry.texturing import (
        _face_adjacency,
        face_view_quality,
        level_seams,
        select_views_mrf,
    )

    quality = face_view_quality(mesh, dataset)
    mrf_labels, _ = select_views_mrf(
        quality, _face_adjacency(np.asarray(mesh.triangles))
    )
    correction = level_seams(mesh, dataset, mrf_labels)
    assert np.abs(correction.values.mean(axis=0)).max() < 1e-9, correction.values.mean(
        axis=0
    )
    # Premise: there is something to anchor -- the corrections are not all zero.
    assert correction.stats["mean_correction"] > 0.01, correction.stats

    # And levelling did not buy that by shifting everything: the atlas must be
    # no further from the mean-exposure ground truth than it was. (It is in
    # fact much closer -- ~0.078 to ~0.052 -- because removing the per-view
    # offsets is removing real error, not just hiding a boundary.)
    unlevelled_mesh = _unit_sphere_mesh(resolution=10)
    _, unlevelled, _ = bake_texture_atlas_view_selected(
        unlevelled_mesh, dataset, texture_size=size, seam_smoothness=None
    )
    truth, covered = _ground_truth_atlas(mesh, size, _default_pattern())
    truth_f = truth[covered] / 255.0
    unlevelled_truth, unlevelled_covered = _ground_truth_atlas(
        unlevelled_mesh, size, _default_pattern()
    )
    l1_levelled = np.abs(levelled[covered] / 255.0 - truth_f).mean()
    l1_unlevelled = np.abs(
        unlevelled[unlevelled_covered] / 255.0
        - unlevelled_truth[unlevelled_covered] / 255.0
    ).mean()
    assert l1_levelled < l1_unlevelled, (l1_levelled, l1_unlevelled)


def test_seam_discontinuity_reports_nothing_to_level_for_one_view():
    """A single-view labelling has no seams, which is not a levelled seam."""
    from gsplat.photogrammetry.metrics import seam_discontinuity
    from gsplat.photogrammetry.texturing import _unwrap_and_rasterize

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=6)
    atlas = _unwrap_and_rasterize(mesh, 64)
    texture = np.zeros((64, 64, 3), dtype=np.uint8)

    single = np.zeros(len(mesh.triangles), dtype=np.int64)
    stats = seam_discontinuity(mesh, texture, single, atlas.triangle_uvs)
    assert stats["num_seam_edges"] == 0
    assert stats["mean"] == 0.0

    # And an alternating labelling does find seams, so the zero above is not
    # the function simply never finding anything.
    alternating = np.arange(len(mesh.triangles)) % 3
    found = seam_discontinuity(mesh, texture, alternating, atlas.triangle_uvs)
    assert found["num_seam_edges"] > 0
    assert set(found) == set(stats)


# ---------------------------------------------------------------------------
# Source-image sampling
# ---------------------------------------------------------------------------


def _nearest(image, uv):
    """Nearest-neighbour sampling, written out as the comparison baseline.

    Deliberately independent of the module under test: a bilinear
    implementation that is quietly broken cannot satisfy a comparison against
    this by being broken in the same way.
    """
    height, width = image.shape[:2]
    px = np.clip(uv[:, 0].astype(np.int64), 0, width - 1)
    py = np.clip(uv[:, 1].astype(np.int64), 0, height - 1)
    return image[py, px]


def test_bilinear_reproduces_a_linear_image_exactly():
    """Bilinear interpolation of a linear ramp is exact, at any coordinate.

    Analytic ground truth that owes nothing to the implementation: if the
    image *is* a linear function of position, the interpolant must return that
    same function, to floating-point precision, everywhere.
    """
    from gsplat.photogrammetry.texturing import _bilinear

    height, width = 23, 31
    ys, xs = np.mgrid[0:height, 0:width]
    # Pixel centres sit at integer + 0.5 in the uv convention.
    ramp = 0.3 * (xs + 0.5) - 0.2 * (ys + 0.5) + 1.5
    image = np.stack([ramp, 2.0 * ramp, -ramp], axis=-1)

    rng = np.random.default_rng(0)
    uv = np.stack(
        [
            rng.uniform(0.5, width - 0.5, size=500),
            rng.uniform(0.5, height - 0.5, size=500),
        ],
        axis=1,
    )
    expected_ramp = 0.3 * uv[:, 0] - 0.2 * uv[:, 1] + 1.5
    expected = np.stack([expected_ramp, 2.0 * expected_ramp, -expected_ramp], axis=-1)
    np.testing.assert_allclose(_bilinear(image, uv), expected, atol=1e-12)


def test_bilinear_returns_the_pixel_exactly_at_its_centre():
    """At a pixel centre there is nothing to interpolate between."""
    from gsplat.photogrammetry.texturing import _bilinear

    rng = np.random.default_rng(1)
    image = rng.random((8, 9, 3))
    ys, xs = np.mgrid[0:8, 0:9]
    uv = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1).astype(np.float64)
    np.testing.assert_allclose(_bilinear(image, uv), image.reshape(-1, 3), atol=1e-12)


def test_bilinear_clamps_at_the_border_rather_than_wrapping():
    """A sample past the edge must take the edge pixel, not the opposite one.

    Wrapping would pull the far side of the image into a silhouette texel --
    the kind of artifact that looks like a reconstruction error rather than a
    sampling bug.
    """
    from gsplat.photogrammetry.texturing import _bilinear

    image = np.zeros((4, 4, 3))
    image[0, 0] = [1.0, 0.0, 0.0]
    image[3, 3] = [0.0, 0.0, 1.0]

    corners = np.array([[-5.0, -5.0], [0.0, 0.0], [99.0, 99.0], [4.0, 4.0]])
    sampled = _bilinear(image, corners)
    np.testing.assert_allclose(sampled[0], image[0, 0])
    np.testing.assert_allclose(sampled[1], image[0, 0])
    np.testing.assert_allclose(sampled[2], image[3, 3])
    np.testing.assert_allclose(sampled[3], image[3, 3])


def test_bilinear_sampling_beats_nearest_on_the_vertex_bake():
    """The reason the sampler is bilinear, measured end to end.

    A surface point almost never lands on a pixel centre, so rounding to one
    throws away up to half a pixel of the projection's accuracy -- and does so
    differently in each view, which is also what makes the views disagree about
    a point's colour by more than they need to.
    """
    from gsplat.photogrammetry import texturing

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    from test_mesh_extraction import _surface_pattern

    dataset = _SphereDataset(num_views=16)

    def bake_error(sampler):
        original = texturing._bilinear
        texturing._bilinear = sampler
        try:
            mesh = _unit_sphere_mesh(resolution=10)
            texturing.bake_texture(mesh, dataset)
        finally:
            texturing._bilinear = original
        vertices = np.asarray(mesh.vertices)
        truth = _surface_pattern(
            vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        )
        return float(np.abs(np.asarray(mesh.vertex_colors) - truth).mean())

    nearest_error = bake_error(_nearest)
    bilinear_error = bake_error(texturing._bilinear)

    # Premise: nearest-neighbour must be measurably wrong here, or there is
    # nothing for interpolation to recover. Measured ~0.0052.
    assert nearest_error > 0.004, nearest_error
    # Measured 0.0052 -> 0.0027, a 1.9x improvement.
    assert bilinear_error < nearest_error / 1.5, (bilinear_error, nearest_error)


# ---------------------------------------------------------------------------
# Sizing the atlas from the evidence
# ---------------------------------------------------------------------------


def test_projected_areas_match_the_analytic_silhouette():
    """A sphere's front faces tile its silhouette disc, whose area is known.

    For radius ``r`` at distance ``d`` with focal ``f``, the silhouette is a
    circle of image radius ``f*r/sqrt(d^2 - r^2)``. Summing the projected areas
    of the faces one view can see must reproduce its area -- a check on the
    projection maths that owes nothing to the implementation.
    """
    from gsplat.photogrammetry.texturing import (
        _project_triangles,
        _triangle_pixel_areas,
        face_visibility,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    radius, distance, focal = 1.0, 3.5, 260.0
    mesh = _unit_sphere_mesh(resolution=40)
    dataset = _SphereDataset(num_views=8, cam_dist=distance, focal=focal)

    visible = face_visibility(mesh, dataset)
    seen = visible[:, 0]
    view = dataset[0]
    uv = _project_triangles(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles)[seen],
        view["camtoworld"].numpy(),
        view["K"].numpy(),
    )
    measured = float(_triangle_pixel_areas(uv).sum())

    image_radius = focal * radius / np.sqrt(distance**2 - radius**2)
    analytic = np.pi * image_radius**2
    # Measured 18850.9 against an analytic 18877.5 -- 0.1%, the rest being the
    # polygon's chord error against the true sphere.
    assert measured == pytest.approx(analytic, rel=0.01), (measured, analytic)


def test_recommended_size_scales_with_the_evidence():
    """Four times the pixels must ask for twice the atlas, on every route.

    Three independent ways to change the amount of evidence -- shoot at higher
    resolution, move the camera closer, or ask for more texels per pixel -- and
    all three are exact factors, so `exact_size` (before the power-of-two
    rounding quantises it) has an exact expected ratio.
    """
    from gsplat.photogrammetry.texturing import recommended_texture_size

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=20)
    # Fix the packing so the comparison isolates the evidence: measuring it
    # re-unwraps, and this is not a test about the unwrapper.
    fixed = dict(packing_efficiency=0.5)

    base = recommended_texture_size(mesh, _SphereDataset(num_views=8), **fixed)[1]

    # 1. Same capture at twice the resolution: area scales as focal squared.
    bigger = recommended_texture_size(
        mesh, _SphereDataset(num_views=8, focal=520.0, width=384, height=384), **fixed
    )[1]
    assert bigger["exact_size"] / base["exact_size"] == pytest.approx(2.0, rel=0.02)

    # 2. Twice as far away is less evidence, but *how much* less is bracketed
    #    rather than exact, and the bracket is the interesting part.
    #
    #    The silhouette-disc law gives sqrt((d2^2 - r^2)/(d1^2 - r^2)) = 2.07.
    #    That is a lower bound here, not the answer: this sums each face's
    #    *best* view, and a face seen head-on projects like a patch at range
    #    (d - r), giving (d2 - r)/(d1 - r) = 2.40 as the upper bound. Faces are
    #    spread across the visible cap rather than all at its closest point, so
    #    the truth sits between. (Measured: 2.27.) Asserting the disc law alone
    #    fails, which is how this bracket came to be written down.
    farther = recommended_texture_size(
        mesh, _SphereDataset(num_views=8, cam_dist=7.0), **fixed
    )[1]
    ratio = base["exact_size"] / farther["exact_size"]
    disc_law = np.sqrt((7.0**2 - 1.0) / (3.5**2 - 1.0))
    head_on_law = (7.0 - 1.0) / (3.5 - 1.0)
    assert disc_law < head_on_law  # premise: the bracket is the right way round
    assert disc_law * 0.97 < ratio < head_on_law * 1.03, (
        ratio,
        disc_law,
        head_on_law,
    )

    # 3. Four texels per pixel is twice the linear size, exactly.
    supersampled = recommended_texture_size(
        mesh, _SphereDataset(num_views=8), texels_per_pixel=4.0, **fixed
    )[1]
    assert supersampled["exact_size"] / base["exact_size"] == pytest.approx(2.0)


def test_recommended_size_rounds_to_the_nearest_power_of_two():
    """Not the next one up: that quadruples the atlas for any overshoot.

    Measured on a test sphere, an exact size of 518.1 rounded *up* landed on
    1024 and baked 3.88x more texels than there were source pixels to fill
    them; nearest lands on 512 and covers 0.98x. Forced here by choosing the
    packing efficiency that puts `exact_size` just either side of the midpoint
    between two powers of two (512 * sqrt(2) = 724).
    """
    from gsplat.photogrammetry.texturing import recommended_texture_size

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=20)
    dataset = _SphereDataset(num_views=8)

    pixels = recommended_texture_size(mesh, dataset, packing_efficiency=1.0)[1][
        "total_source_pixels"
    ]
    for target, expected in ((600.0, 512), (800.0, 1024)):
        # exact = sqrt(pixels / packing)  =>  packing = pixels / target^2
        packing = pixels / target**2
        assert 0 < packing <= 1, packing
        size, stats = recommended_texture_size(
            mesh, dataset, packing_efficiency=packing
        )
        assert stats["exact_size"] == pytest.approx(target, rel=0.01)
        assert size == expected, (target, size, expected)


def test_recommended_size_measures_packing_on_the_mesh_by_default():
    """There is no defensible constant, so it is measured rather than assumed.

    Across test spheres the packing ranges 42.7%-73.2% and not monotonically in
    density, while five repeated unwraps of one mesh spread by at most 2.8%.
    That combination -- mesh-dependent but stable -- is what makes a probe
    worth an extra unwrap.
    """
    from gsplat.photogrammetry.texturing import recommended_texture_size

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    dataset = _SphereDataset(num_views=8)

    _, measured = recommended_texture_size(_unit_sphere_mesh(resolution=20), dataset)
    assert measured["packing_efficiency_measured"] is True
    assert 0.1 < measured["packing_efficiency"] < 1.0, measured

    _, given = recommended_texture_size(
        _unit_sphere_mesh(resolution=20), dataset, packing_efficiency=0.5
    )
    assert given["packing_efficiency_measured"] is False
    assert given["packing_efficiency"] == 0.5

    # Premise: the two routes really do disagree here, so the flag is not
    # distinguishing identical results.
    assert measured["packing_efficiency"] != pytest.approx(0.5, abs=0.02)


def test_recommended_size_rejects_nonsense():
    from gsplat.photogrammetry.texturing import (
        face_projected_areas,
        recommended_texture_size,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=6)
    dataset = _SphereDataset(num_views=4)

    with pytest.raises(ValueError, match="texels_per_pixel must be positive"):
        recommended_texture_size(mesh, dataset, texels_per_pixel=0.0)
    with pytest.raises(ValueError, match=r"packing_efficiency must be in \(0, 1\]"):
        recommended_texture_size(mesh, dataset, packing_efficiency=1.5)
    with pytest.raises(ValueError, match="probe_size must be positive"):
        recommended_texture_size(mesh, dataset, probe_size=0)
    with pytest.raises(ValueError, match="min_size <= max_size"):
        recommended_texture_size(mesh, dataset, min_size=4096, max_size=512)
    with pytest.raises(ValueError, match="no triangles"):
        face_projected_areas(o3d.geometry.TriangleMesh(), dataset)


def test_recommended_size_is_clamped_and_says_so():
    """A bound biting is reported, not silently applied."""
    from gsplat.photogrammetry.texturing import recommended_texture_size

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=8)
    dataset = _SphereDataset(num_views=8)

    size, stats = recommended_texture_size(
        mesh, dataset, packing_efficiency=0.5, min_size=4096, max_size=8192
    )
    assert size == 4096
    assert stats["clamped"] is True
    # And the unclamped truth is still reported, so a caller can see how far
    # off the bound was.
    assert stats["exact_size"] < 4096

    _, unclamped = recommended_texture_size(mesh, dataset, packing_efficiency=0.5)
    assert unclamped["clamped"] is False


def test_evidence_is_bounded_by_the_surface_not_the_view_count():
    """Photographing the same surface more times is not more detail.

    `face_projected_areas` takes the *maximum* over views, not the sum, because
    texture detail is limited by the best look the capture ever got at a
    surface. Summing would make the recommended atlas grow with the number of
    photographs -- shoot the same object twice as many times from the same
    distance and get an atlas twice the area, carrying no more detail.

    Ratio-based tests cannot catch that: a consistent over-count cancels out of
    every ratio. This pins the absolute behaviour.
    """
    from gsplat.photogrammetry.texturing import (
        face_projected_areas,
        face_visibility,
    )

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=20)

    few, many = _SphereDataset(num_views=8), _SphereDataset(num_views=24)
    # Premise: tripling the views really does mean each face is seen by far
    # more of them -- 2.9 on average rising to 8.6 -- so summing and maxing
    # would give very different answers here.
    seen_few = face_visibility(mesh, few).sum(axis=1).mean()
    seen_many = face_visibility(mesh, many).sum(axis=1).mean()
    assert seen_many > 2.5 * seen_few, (seen_few, seen_many)

    evidence_few = float(face_projected_areas(mesh, few).sum())
    evidence_many = float(face_projected_areas(mesh, many).sum())

    # More views do buy a little -- each face's *best* look improves -- but
    # nothing like proportionally. Measured 1.26x for 3x the photographs.
    assert 1.0 <= evidence_many / evidence_few < 1.6, (
        evidence_few,
        evidence_many,
    )


# ---------------------------------------------------------------------------
# Multi-page atlases
# ---------------------------------------------------------------------------


def _two_quads():
    """A near quad hiding the centre of a larger one behind it.

    The smallest scene that is *non-convex*: both quads face the cameras, so
    back-face rejection does not separate them and only an occlusion test can.
    A sphere cannot stand in for this -- it is convex, so every face the
    cameras should not see is also facing away from them, and a bake with no
    occlusion test at all still gets the right answer.
    """
    vertices, triangles = [], []
    for z, half in ((0.0, 0.5), (-1.0, 1.0)):
        base = len(vertices)
        vertices += [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ]
        triangles += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


class _QuadViews:
    """Cameras in front of :func:`_two_quads`; near quad red, far quad blue."""

    def __init__(self, mesh, num_views=6, distance=4.0, size=128, focal=160.0):
        torch = pytest.importorskip("torch")

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1.0]])
        self._items = []
        for i in range(num_views):
            angle = 0.25 * (i / max(num_views - 1, 1) - 0.5)
            camera = np.array([distance * np.sin(angle), 0.0, distance * np.cos(angle)])
            forward = -camera / np.linalg.norm(camera)
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            right /= np.linalg.norm(right)
            rotation = np.stack([right, -np.cross(right, forward), forward], axis=1)
            camtoworld = np.eye(4)
            camtoworld[:3, :3], camtoworld[:3, 3] = rotation, camera

            ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            dirs = np.stack(
                [
                    (xs - size / 2 + 0.5) / focal,
                    (ys - size / 2 + 0.5) / focal,
                    np.ones_like(xs),
                ],
                axis=-1,
            )
            dirs = np.einsum(
                "ij,hwj->hwi",
                rotation,
                dirs / np.linalg.norm(dirs, axis=-1, keepdims=True),
            )
            rays = np.concatenate(
                [np.broadcast_to(camera, (size, size, 3)), dirs], axis=-1
            ).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))
            primitive = hit["primitive_ids"].numpy()
            landed = np.isfinite(hit["t_hit"].numpy())

            image = np.zeros((size, size, 3))
            image[landed & (primitive < 2)] = [0.9, 0.1, 0.1]
            image[landed & (primitive >= 2) & (primitive < 4)] = [0.1, 0.1, 0.9]
            self._items.append(
                {
                    "camtoworld": torch.from_numpy(camtoworld),
                    "K": torch.from_numpy(K),
                    "image": torch.from_numpy((image * 255).astype(np.float32)),
                }
            )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_a_page_is_occluded_by_the_rest_of_the_mesh_not_just_itself():
    """The trap in baking one page of a multi-page atlas.

    A page ray-cast against its own geometry alone is blind to everything the
    rest of the surface puts in front of it -- it textures the far wall of a
    room straight through the near one. Here the far quad's centre is hidden
    in every view, so the only correct answer is "never observed"; casting
    against the page alone instead samples the *near* quad's red onto it.
    """
    from gsplat.photogrammetry.texturing import _bake_points_from_views

    mesh = _two_quads()
    dataset = _QuadViews(mesh)
    hidden_point = np.array([[0.0, 0.0, -1.0]])
    facing_camera = np.array([[0.0, 0.0, 1.0]])

    far_page = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[4:]),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )

    _, weight_whole = _bake_points_from_views(
        mesh, dataset, hidden_point, facing_camera, occluder=mesh
    )
    colors_page, weight_page = _bake_points_from_views(
        mesh, dataset, hidden_point, facing_camera, occluder=far_page
    )

    # Correct: hidden in every view, so nothing was accumulated.
    assert weight_whole[0] == 0.0, weight_whole

    # Premise: casting against the page alone does *not* merely lose accuracy,
    # it accepts the sample -- and the colour it accepts is the occluder's.
    assert weight_page[0] > 0.0, weight_page
    sampled = colors_page[0] / weight_page[0]
    assert sampled[0] > 0.7 and sampled[2] < 0.3, sampled  # the near quad's red


def test_partition_is_balanced_deterministic_and_compact():
    """Even face counts, reproducible, and groups that stay together in space."""
    from gsplat.photogrammetry.texturing import partition_faces

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=20)
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)

    spreads = []
    for pages in (1, 2, 4, 8):
        labels = partition_faces(mesh, pages)
        assert labels.min() >= 0 and labels.max() == pages - 1
        counts = np.bincount(labels, minlength=pages)
        # Median splits of powers of two are exactly even.
        assert counts.max() == counts.min(), (pages, counts)
        np.testing.assert_array_equal(labels, partition_faces(mesh, pages))
        spreads.append(
            float(
                np.mean(
                    [
                        np.linalg.norm(
                            centroids[labels == g] - centroids[labels == g].mean(0),
                            axis=1,
                        ).mean()
                        for g in range(pages)
                    ]
                )
            )
        )
    # Compactness: more pages must mean tighter groups, or the split is not
    # spatial at all. Measured 0.996 -> 0.890 -> 0.770 -> 0.476.
    assert spreads == sorted(spreads, reverse=True), spreads
    assert spreads[-1] < 0.6 * spreads[0], spreads

    # A count that is not a power of two still works, and is balanced within
    # 2x -- "split the biggest" cannot do better with median cuts.
    counts = np.bincount(partition_faces(mesh, 3))
    assert counts.max() / counts.min() <= 2.0, counts


def test_partition_rejects_more_pages_than_faces():
    from gsplat.photogrammetry.texturing import partition_faces

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    mesh = _unit_sphere_mesh(resolution=4)
    with pytest.raises(ValueError, match="num_pages must be positive"):
        partition_faces(mesh, 0)
    with pytest.raises(ValueError, match="page with no faces"):
        partition_faces(mesh, len(mesh.triangles) + 1)
    with pytest.raises(ValueError, match="no triangles"):
        partition_faces(o3d.geometry.TriangleMesh(), 2)


def _page_face_error(mesh, textures, pattern, size):
    """Mean colour error sampling each face through *its own* page and UVs."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    uvs = np.asarray(mesh.triangle_uvs)
    ids = np.asarray(mesh.triangle_material_ids)
    centroids = vertices[triangles].mean(axis=1)
    truth = pattern(centroids / np.linalg.norm(centroids, axis=1, keepdims=True))

    errors = []
    for face in range(len(triangles)):
        uv = uvs[3 * face : 3 * face + 3].mean(axis=0)
        col = int(np.clip(uv[0] * size, 0, size - 1))
        row = int(np.clip((1.0 - uv[1]) * size, 0, size - 1))
        errors.append(
            np.abs(textures[ids[face]][row, col] / 255.0 - truth[face]).mean()
        )
    return float(np.mean(errors))


def test_pages_buy_texel_budget_without_a_bigger_image():
    """N pages of size S carry what one page of the same total texels does.

    That is the whole point: past 8192 or 16384 a side an atlas stops being
    practical, and splitting is the only way to keep adding texels. Measured
    on a high-frequency pattern -- 4x256 lands at 0.0318 against 1x512's
    0.0310, where a single 256 manages only 0.0569.
    """
    from gsplat.photogrammetry.texturing import bake_texture_atlas_pages

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    dataset = _SphereDataset(num_views=16, pattern=_high_frequency_pattern)

    def bake(pages, size):
        mesh = _unit_sphere_mesh(resolution=30)
        mesh, textures, stats = bake_texture_atlas_pages(
            mesh, dataset, num_pages=pages, texture_size=size
        )
        assert stats["total_texels"] == pages * size * size
        assert len(textures) == pages
        return _page_face_error(mesh, textures, _high_frequency_pattern, size)

    one_small = bake(1, 256)
    four_small = bake(4, 256)
    one_big = bake(1, 512)

    # Premise: a single small page must actually be losing detail here.
    assert one_small > 1.4 * one_big, (one_small, one_big)
    # Four pages recover it, landing near the equal-budget single page.
    assert four_small < 0.75 * one_small, (four_small, one_small)
    assert four_small < 1.25 * one_big, (four_small, one_big)


def test_multi_page_mesh_writes_and_reads_back_as_multi_material_obj():
    """Every face must address the page its material id names, through disk.

    A UV scattered onto the wrong face, or a material id off by one, produces
    an asset that loads without complaint and is textured with someone else's
    surface -- so this checks the colours after a round trip, not just the
    counts.
    """
    from gsplat.photogrammetry.texturing import bake_texture_atlas_pages

    _SphereDataset, _unit_sphere_mesh = _sphere_fixtures()
    from test_mesh_extraction import _surface_pattern

    size = 256
    mesh = _unit_sphere_mesh(resolution=20)
    mesh, textures, stats = bake_texture_atlas_pages(
        mesh, _SphereDataset(num_views=16), num_pages=4, texture_size=size
    )
    assert stats["faces_per_page"] == [380, 380, 380, 380]
    assert _page_face_error(mesh, textures, _surface_pattern, size) < 0.02

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mesh.obj")
        assert o3d.io.write_triangle_mesh(path, mesh)
        written = sorted(os.listdir(directory))
        assert written == [
            "mesh.mtl",
            "mesh.obj",
            "mesh_0.png",
            "mesh_1.png",
            "mesh_2.png",
            "mesh_3.png",
        ], written

        loaded = o3d.io.read_triangle_mesh(path, True)
        assert len(loaded.textures) == 4
        assert len(np.unique(np.asarray(loaded.triangle_material_ids))) == 4


def test_page_bake_uses_the_whole_mesh_as_the_occluder():
    """The same trap as above, pinned where the pipeline actually hits it.

    `test_a_page_is_occluded_by_the_rest_of_the_mesh_not_just_itself` proves
    the sampler honours an occluder; this proves `bake_texture_atlas_pages`
    passes it one. Mutation checking found that gap: swapping the call site to
    `occluder=None` left the whole suite green, because the direct test
    supplies its own occluder either way.
    """
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas_pages,
        partition_faces,
    )

    size = 128
    mesh = _two_quads()
    # Premise: the split must actually separate the two quads, or there is no
    # cross-page occlusion to get wrong. Faces 0-1 are the near quad, 2-3 the far.
    labels = partition_faces(mesh, 2)
    assert labels[0] == labels[1] and labels[2] == labels[3], labels
    assert labels[0] != labels[2], labels

    mesh, textures, _ = bake_texture_atlas_pages(
        _two_quads(), _QuadViews(_two_quads()), num_pages=2, texture_size=size
    )
    uvs = np.asarray(mesh.triangle_uvs)
    ids = np.asarray(mesh.triangle_material_ids)

    sampled = []
    for face in (2, 3):  # the far quad, hidden at its centre in every view
        uv = uvs[3 * face : 3 * face + 3].mean(axis=0)
        col = int(np.clip(uv[0] * size, 0, size - 1))
        row = int(np.clip((1.0 - uv[1]) * size, 0, size - 1))
        sampled.append(textures[ids[face]][row, col] / 255.0)
    hidden = np.mean(sampled, axis=0)

    # The near quad is red. Whatever the hidden centre ends up as -- unobserved
    # and left to the dilation fill, or reached by the far quad's own visible
    # blue border -- it must not be the thing standing in front of it.
    # Measured: [0, 0, 0] correct, [0.898, 0.102, 0.102] with the occluder
    # dropped.
    assert not (hidden[0] > 0.5 and hidden[1] < 0.3), hidden


# ---------------------------------------------------------------------------
# Multi-view super-resolution
# ---------------------------------------------------------------------------


def _mesh_rendered_views(mesh, num_views=10, width=48, pose_error_arcmin=0.0):
    """Views ray-cast from `mesh` itself, so geometry is not a confound.

    `_SphereDataset` renders the *analytic* sphere while `_unit_sphere_mesh` is
    a polyhedron inscribed in it, and the gap between them is a sizeable
    fraction of `_high_frequency_pattern`'s wavelength. A deconvolution is
    exactly the wrong thing to measure through that gap: it would be charged
    for tessellation error it cannot fix. See
    `tests/test_photometric_alignment.py`, which learned this the hard way.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_photometric_alignment import _MeshViews

    return _MeshViews(
        mesh,
        num_views=num_views,
        width=width,
        pose_error_arcmin=pose_error_arcmin,
    )


def test_super_resolution_beats_blending_on_both_sharpness_and_error():
    """The one claim that holds unambiguously, and the reason this exists.

    Blending averages every view that sees a texel, which low-passes the result.
    Super-resolution instead models how the texture *became* each photograph --
    projected and blurred by the camera's PSF -- and solves for the texture
    whose reprojection explains all of them. Against blending that is a strict
    win on both axes at once, which nothing else in this module manages:
    view selection buys sharpness with pointwise accuracy (ISSUES.md § 4.1).

    Measured: contrast 63.6% -> 90.8% of the ground truth's, L1 0.1081 ->
    0.0759.
    """
    _, _unit_sphere_mesh = _sphere_fixtures()
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas,
        bake_texture_atlas_super_resolved,
    )

    size = 128
    mesh = _unit_sphere_mesh(resolution=16)
    dataset = _mesh_rendered_views(mesh)

    _, blended = bake_texture_atlas(mesh, dataset, texture_size=size)
    _, resolved, stats = bake_texture_atlas_super_resolved(
        mesh, dataset, texture_size=size
    )
    truth, covered = _ground_truth_atlas(mesh, size, _high_frequency_pattern)

    truth_grad = atlas_sharpness(truth, covered)["mean_gradient"]
    truth_flat = truth[covered] / 255.0
    blended_grad = atlas_sharpness(blended, covered)["mean_gradient"] / truth_grad
    resolved_grad = atlas_sharpness(resolved, covered)["mean_gradient"] / truth_grad
    blended_l1 = np.abs(blended[covered] / 255.0 - truth_flat).mean()
    resolved_l1 = np.abs(resolved[covered] / 255.0 - truth_flat).mean()

    # Premise: blending must actually be losing detail here.
    assert blended_grad < 0.75, blended_grad
    # Sharper *and* closer -- the whole point.
    assert resolved_grad > 1.25 * blended_grad, (resolved_grad, blended_grad)
    assert resolved_l1 < 0.85 * blended_l1, (resolved_l1, blended_l1)

    # The solve reports the same verdict from its own stats, so a caller need
    # not build a ground-truth atlas to see it.
    assert (
        stats["atlas_sharpness"]["mean_gradient"]
        > stats["blended_atlas_sharpness"]["mean_gradient"]
    )
    assert stats["solver"]["converged"], stats["solver"]


def test_super_resolution_is_sharper_than_view_selection_but_not_strictly_better():
    """The claim that does **not** hold, pinned so it is not quietly assumed.

    The pitch for this method is that it should be strictly better than both
    alternatives rather than another tradeoff. Against blending it is (above).
    Against view selection it is not: it is reliably sharper, but pointwise it
    wins in some regimes and loses in others. Measured against the same
    ground-truth atlas:

    ======================================  ==============  ==============
    Regime                                  view selection  super-resolved
    ======================================  ==============  ==============
    128 atlas, 48 px views (sigma 0.61 tx)  0.0785          0.0759
    192 atlas, 48 px views (sigma 0.89 tx)  0.0787          0.0784
    256 atlas, 64 px views (sigma 0.89 tx)  0.0770          0.0788
    384 atlas, 64 px views (sigma 1.54 tx)  0.0400          0.0493
    ======================================  ==============  ==============

    The reason is structural and worth keeping in mind before tuning it away:
    single-view sampling reads the source pixels with *no forward model at
    all*, while this one approximates the PSF as a per-view Gaussian in atlas
    space. Every approximation in that model shows up as pointwise error, and
    at wide PSFs it costs more than the deconvolution recovers.

    So the assertion here is the honest one -- sharper, and *comparable*
    pointwise -- and the feature stays opt-in. If a real capture ever shows a
    strict win, tighten this; do not tighten it on synthetic data alone.
    """
    _, _unit_sphere_mesh = _sphere_fixtures()
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas_super_resolved,
        bake_texture_atlas_view_selected,
    )

    size = 128
    mesh = _unit_sphere_mesh(resolution=16)
    dataset = _mesh_rendered_views(mesh)

    _, selected, _ = bake_texture_atlas_view_selected(mesh, dataset, texture_size=size)
    _, resolved, _ = bake_texture_atlas_super_resolved(mesh, dataset, texture_size=size)
    truth, covered = _ground_truth_atlas(mesh, size, _high_frequency_pattern)

    truth_grad = atlas_sharpness(truth, covered)["mean_gradient"]
    truth_flat = truth[covered] / 255.0
    selected_grad = atlas_sharpness(selected, covered)["mean_gradient"] / truth_grad
    resolved_grad = atlas_sharpness(resolved, covered)["mean_gradient"] / truth_grad
    selected_l1 = np.abs(selected[covered] / 255.0 - truth_flat).mean()
    resolved_l1 = np.abs(resolved[covered] / 255.0 - truth_flat).mean()

    # Reliably sharper. Measured 90.8% against 77.0%.
    assert resolved_grad > 1.1 * selected_grad, (resolved_grad, selected_grad)
    # And pointwise comparable, in *either* direction -- deliberately not
    # asserted as a win. A one-sided assertion here would be a claim the
    # measurements above do not support.
    assert 0.75 < resolved_l1 / selected_l1 < 1.35, (resolved_l1, selected_l1)


def test_the_deconvolution_stays_bounded_instead_of_ringing():
    """The prior enters the normal equations with a **plus**, and this pins it.

    `_laplacian` returns the positive semi-definite graph Laplacian, so the
    normal equations of ``||S T - c||^2 + lambda T^T L T`` are
    ``(S^T W S + lambda L) T = S^T W c``. Subtracting it instead -- an easy slip,
    and the one made first here -- leaves the operator *indefinite*: conjugate
    gradient has no descent direction to find and the "deconvolution" diverges.

    The tell is not a crash. It is an atlas that looks spectacularly sharp:
    measured **464% of the ground truth's contrast**, with L1 five times worse,
    and non-monotonic in ``regularization`` (0.02 -> 464%, 0.1 -> 168%,
    0.5 -> 375%), which is what finally gave it away. `atlas_sharpness` alone
    would have called that a triumph, so this asserts an upper bound too.
    """
    _, _unit_sphere_mesh = _sphere_fixtures()
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.texturing import bake_texture_atlas_super_resolved

    size = 128
    mesh = _unit_sphere_mesh(resolution=16)
    dataset = _mesh_rendered_views(mesh)

    # Bake once before building the ground truth, so both ride the same UV
    # layout -- `compute_uvatlas` is non-deterministic (ISSUES.md #10).
    grads = []
    for regularization in (0.05, 0.2, 0.8):
        _, resolved, _ = bake_texture_atlas_super_resolved(
            mesh, dataset, texture_size=size, regularization=regularization
        )
        grads.append((regularization, resolved))
    truth, covered = _ground_truth_atlas(mesh, size, _high_frequency_pattern)
    truth_grad = atlas_sharpness(truth, covered)["mean_gradient"]

    measured = []
    for regularization, resolved in grads:
        grad = atlas_sharpness(resolved, covered)["mean_gradient"] / truth_grad
        # Recovering a little more gradient energy than the truth is normal
        # (sampling noise is high-frequency); 464% is divergence.
        assert grad < 1.5, (regularization, grad)
        measured.append(grad)

    # More smoothing must move it monotonically back toward the blend. Note
    # this is asserted on *contrast*, not on L1: L1 against the truth is
    # U-shaped in `regularization` (it bottoms out near 0.1 and rises on both
    # sides), so a monotonicity assertion on L1 is simply false. Contrast falls
    # monotonically -- 108% -> 91% -> 78% -> 47% -> 13% across two decades --
    # and the indefinite operator's 464% -> 168% -> 375% does not.
    assert measured == sorted(measured, reverse=True), measured


def test_the_psf_width_is_measured_from_the_capture():
    """sigma is derived, not tuned: a finer atlas means a wider PSF in texels.

    The whole mechanism is that a view's *footprint* -- how much surface one of
    its pixels covers -- decides how much it can say about a texel, and the
    existing blend weight (``clamp(n.-d, 0, 1) / dist``) has no notion of it.
    Doubling the atlas halves the world size of a texel, so one source pixel
    spans twice as many of them and sigma must double.
    """
    _, _unit_sphere_mesh = _sphere_fixtures()
    from gsplat.photogrammetry.texturing import bake_texture_atlas_super_resolved

    mesh = _unit_sphere_mesh(resolution=16)
    dataset = _mesh_rendered_views(mesh)

    sigmas = {}
    for size in (64, 128):
        _, _, stats = bake_texture_atlas_super_resolved(
            _unit_sphere_mesh(resolution=16), dataset, texture_size=size
        )
        sigmas[size] = stats["mean_psf_sigma_texels"]
        assert stats["texel_world_size"] > 0.0

    ratio = sigmas[128] / sigmas[64]
    assert 1.7 < ratio < 2.3, (sigmas, ratio)
