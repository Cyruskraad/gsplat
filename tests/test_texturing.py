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
