"""Level-set surface extraction (GOF-style), against analytic fields.

The GPU half of this module -- evaluating a Gaussian opacity field on a real
trained checkpoint -- cannot run here. Everything else can, and does: the
tetrahedral grid, marching tetrahedra, the orchestration, every diagnostic, and
(on CPU, with synthetic Gaussians) the field adapter's own mathematics.

What remains genuinely unverified is narrow and is stated in the module
docstring: CUDA execution, and behaviour on a real scene's thousands of
anisotropic Gaussians.
"""

import os
import sys

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="open3d not installed")

from gsplat.photogrammetry.level_set import (  # noqa: E402
    extract_level_set,
    level_set_residual,
    marching_tetrahedra,
    probe_field,
    tetrahedral_grid,
    validate_level_set_pipeline,
)

BOX = np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])


def _sphere(points):
    return np.linalg.norm(points, axis=1) - 1.0


def test_a_sphere_comes_out_watertight_and_the_right_size():
    """Closed-form ground truth: radius 1, area 4pi, volume 4pi/3."""
    mesh, stats = extract_level_set(_sphere, BOX, resolution=32, probe=False)

    assert stats["is_watertight"], "the extracted sphere is not closed"
    radii = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    assert np.abs(radii - 1.0).mean() < 0.002
    assert mesh.get_surface_area() == pytest.approx(4 * np.pi, rel=0.005)
    assert mesh.get_volume() == pytest.approx(4 * np.pi / 3, rel=0.01)


def test_triangles_are_built_from_the_right_edges():
    """Connectivity, which vertex positions cannot check.

    Marching tetrahedra emits its corners in per-edge blocks, and reshaping
    those to triangles the wrong way builds each face from three *different*
    tetrahedra's corners. The vertices still land in exactly the right places,
    because each is interpolated along its own edge -- so a test that measures
    where the vertices are passes while the surface is shredded. Only
    connectivity sees it, which is why this is separate from the test above.
    """
    mesh, _ = extract_level_set(_sphere, BOX, resolution=16, probe=False, clean=False)

    assert mesh.is_edge_manifold(allow_boundary_edges=False), (
        "some edge is shared by more than two triangles -- the faces are not "
        "being assembled from the correct edge triples"
    )
    assert len(mesh.get_non_manifold_edges(allow_boundary_edges=False)) == 0


def test_halving_the_cell_size_quarters_the_error():
    """Linear interpolation on a smooth field is second-order accurate.

    A *linear* improvement would mean the crossing point is being placed
    wrongly along its edge -- a surface that still looks plausible.
    """
    errors = {}
    for resolution in (16, 32):
        mesh, _ = extract_level_set(_sphere, BOX, resolution=resolution, probe=False)
        radii = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
        errors[resolution] = float(np.abs(radii - 1.0).mean())

    ratio = errors[16] / errors[32]
    assert 2.5 < ratio < 6.0, f"error fell by {ratio:.2f}x, not the ~4x expected"


def test_an_open_surface_is_not_claimed_to_be_closed():
    """A plane across the box: area exactly 9, and not watertight."""
    mesh, stats = extract_level_set(lambda p: p[:, 2], BOX, resolution=24, probe=False)
    assert mesh.get_surface_area() == pytest.approx(9.0, abs=0.05)
    assert not stats["is_watertight"]


def test_normals_point_out_of_the_surface_whichever_way_the_field_runs():
    """Winding comes from the field, not from the case table.

    `_MT_CASES` deliberately does not try to get winding right; orientation is
    established afterwards by asking the field which side is outside. Negating
    the field must therefore produce the same surface with normals still
    pointing away from the solid.
    """
    normal_mesh, normal_stats = extract_level_set(
        _sphere, BOX, resolution=16, probe=False
    )
    flipped_mesh, flipped_stats = extract_level_set(
        lambda p: -_sphere(p), BOX, resolution=16, probe=False
    )

    def outwardness(mesh):
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        return float((normals * radial).sum(axis=1).mean())

    # Premise: the two really do come out oriented oppositely, or there is
    # nothing here to check. (Neither needed an explicit flip -- open3d's own
    # `orient_triangles` happens to agree with the field both times -- so the
    # flip mechanism itself is pinned separately, below.)
    assert normal_stats["orientation_confidence"] > 0.9
    assert flipped_stats["orientation_confidence"] > 0.9
    # For -f the solid is the *outside*, so "away from the solid" is inward.
    assert outwardness(normal_mesh) > 0.9
    assert outwardness(flipped_mesh) < -0.9


def test_a_reversed_mesh_is_flipped_back_against_the_field():
    """Pins the flip itself, which the test above cannot.

    `_orient_against_field` is the only thing standing between the case table's
    untrusted winding and an inside-out asset, but on these scenes open3d's
    `orient_triangles` already lands the right way round, so the flip branch
    never runs. Feeding it a deliberately reversed mesh is the only way to
    exercise it.
    """
    from gsplat.photogrammetry.level_set import _orient_against_field

    mesh, _ = extract_level_set(_sphere, BOX, resolution=12, probe=False)
    reversed_mesh = o3d.geometry.TriangleMesh(
        mesh.vertices, o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
    )

    def outwardness(m):
        m.compute_vertex_normals()
        vertices = np.asarray(m.vertices)
        normals = np.asarray(m.vertex_normals)
        radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        return float((normals * radial).sum(axis=1).mean())

    # Premise: it really is inside out to begin with.
    assert outwardness(reversed_mesh) < -0.9

    result = _orient_against_field(reversed_mesh, _sphere, 0.0, 0.05)
    assert result["flipped"] is True
    assert result["orientation_confidence"] > 0.9
    assert outwardness(reversed_mesh) > 0.9


@pytest.mark.parametrize(
    "field, level, expect",
    [
        (_sphere, 99.0, "outside the sampled field range"),
        (lambda p: np.full(len(p), np.nan), 0.0, "NaN or infinite"),
        (lambda p: np.ones(len(p)), 0.0, "constant"),
    ],
)
def test_every_silent_failure_names_its_cause(field, level, expect):
    """The three ways this returns nothing, each explained rather than blank.

    On a GPU run there is no way to step through, so an empty mesh has to
    arrive with a sentence saying which of the level, the bounds or the field
    is responsible.
    """
    mesh, stats = extract_level_set(field, BOX, resolution=8, level=level)

    assert stats["num_triangles"] == 0
    assert any(
        expect in w for w in stats["warnings"]
    ), f"no warning mentioned {expect!r}; got {stats['warnings']}"
    # The empty case must report the same keys as a successful one -- it is the
    # case a caller inspects precisely because something went wrong.
    for key in ("num_vertices", "num_triangles", "is_watertight", "residual"):
        assert key in stats


def test_probe_field_predicts_an_empty_extraction_before_it_happens():
    """The cheap check that is meant to run before spending GPU time."""
    good = probe_field(_sphere, BOX, level=0.0, resolution=8)
    assert good["crossable"] and not good["warnings"]

    bad = probe_field(_sphere, BOX, level=99.0, resolution=8)
    assert not bad["crossable"]
    assert bad["warnings"]
    # It must also suggest something usable rather than only complaining.
    assert bad["min"] < bad["suggested_level"] < bad["max"]


def test_the_residual_measures_faithfulness_without_ground_truth():
    """The self-check available on a real capture, where truth is unavailable."""
    fine, _ = extract_level_set(_sphere, BOX, resolution=32, probe=False)
    coarse, _ = extract_level_set(_sphere, BOX, resolution=4, probe=False)

    fine_residual = level_set_residual(fine, _sphere)["mean_abs"]
    coarse_residual = level_set_residual(coarse, _sphere)["mean_abs"]

    assert fine_residual < 0.01
    assert coarse_residual > fine_residual


def test_the_grid_refuses_a_degenerate_box():
    for bad in (np.array([[0.0, 0, 0], [0, 0, 0]]), np.array([[1.0, 1, 1], [0, 0, 0]])):
        with pytest.raises(ValueError, match="positive, finite box"):
            tetrahedral_grid(bad, 4)


def test_field_values_must_match_the_vertices():
    vertices, tets = tetrahedral_grid(BOX, 4)
    with pytest.raises(ValueError, match="one-to-one"):
        marching_tetrahedra(vertices, tets, np.zeros(len(vertices) + 1))


def test_the_self_test_passes_and_is_the_thing_to_run_first():
    """`validate_level_set_pipeline` is what a GPU user runs before anything."""
    results = validate_level_set_pipeline(resolution=12, verbose=False)
    assert results["passed"], results["failures"]
    assert results["checks"]["plane"]["area"] == pytest.approx(9.0, abs=0.05)


# --- The GPU adapter: its mathematics, executed on CPU ------------------------


def _single_gaussian(scale, quat=(1.0, 0.0, 0.0, 0.0), opacity_logit=4.0):
    torch = pytest.importorskip("torch", reason="torch not installed")
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
    return {
        "means": torch.zeros(1, 3),
        "quats": torch.tensor([list(quat)], dtype=torch.float32),
        "scales": torch.log(torch.tensor([scale.tolist()], dtype=torch.float32)),
        "opacities": torch.tensor([opacity_logit]),
    }


def test_the_gaussian_field_matches_its_closed_form():
    """One isotropic Gaussian has an analytic 0.5-opacity radius.

    This is the GPU adapter's actual arithmetic, run on CPU. What it cannot
    check is CUDA execution or a real scene's thousands of Gaussians -- but it
    does rule out the arithmetic being wrong, which is most of what would be.
    """
    from gsplat.photogrammetry.level_set import gaussian_opacity_field

    sigma, logit = 0.3, 4.0
    opacity = 1.0 / (1.0 + np.exp(-logit))
    # 0.5 - o * exp(-r^2 / 2 sigma^2) = 0  =>  r = sigma * sqrt(2 ln(2 o))
    expected = sigma * np.sqrt(2.0 * np.log(2.0 * opacity))

    field_fn, info = gaussian_opacity_field(_single_gaussian(sigma), device="cpu")
    assert info["executed"] is False, "this adapter must not claim to be verified"

    at_radius = field_fn(np.array([[expected, 0.0, 0.0]]))
    assert float(at_radius[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(field_fn(np.zeros((1, 3)))[0]) < 0.0  # inside

    mesh, stats = extract_level_set(
        field_fn, np.array(info["bounds"]), resolution=48, probe=False
    )
    assert stats["is_watertight"]
    radii = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    assert radii.mean() == pytest.approx(expected, rel=0.005)


@pytest.mark.parametrize(
    "quat, longest",
    [
        ((1.0, 0.0, 0.0, 0.0), 0),
        ((0.70710678, 0.0, 0.0, 0.70710678), 1),  # 90 deg about z: x -> y
        ((0.70710678, 0.0, 0.70710678, 0.0), 2),  # 90 deg about y: x -> z
    ],
)
def test_the_gaussian_field_uses_the_right_rotation_convention(quat, longest):
    """A transposed rotation or a mis-ordered quaternion is silent.

    It produces a plausible ellipsoid pointing the wrong way, which no
    watertightness or convergence check can see. An anisotropic Gaussian
    elongated along local x must end up along the axis the quaternion sends x to.
    """
    from gsplat.photogrammetry.level_set import gaussian_opacity_field

    field_fn, info = gaussian_opacity_field(
        _single_gaussian([0.40, 0.10, 0.10], quat=quat), device="cpu"
    )
    mesh, _ = extract_level_set(
        field_fn, np.array(info["bounds"]), resolution=48, probe=False
    )
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(axis=0) - vertices.min(axis=0)

    assert int(np.argmax(extent)) == longest, (
        f"the ellipsoid's long axis came out along {'xyz'[int(np.argmax(extent))]}, "
        f"expected {'xyz'[longest]} -- the rotation convention is wrong"
    )
    # A rotation cannot change the lengths, only which axis they lie along.
    assert sorted(extent) == pytest.approx(sorted([0.9298, 0.233, 0.233]), rel=0.05)


def test_the_gaussian_field_rejects_a_checkpoint_it_cannot_read():
    from gsplat.photogrammetry.level_set import gaussian_opacity_field

    pytest.importorskip("torch", reason="torch not installed")
    with pytest.raises(ValueError, match="missing"):
        gaussian_opacity_field({"means": None}, device="cpu")


def test_the_rotation_is_not_its_own_transpose():
    """Distinguishes R from R^T, which the axis-aligned cases above cannot.

    For a 90-degree rotation about a single axis, R and R^T produce
    mirror-image ellipsoids -- and an axis-aligned bounding box is invariant
    under that mirror, so the extents are *identical* and the test above passes
    either way. Transposing the rotation is exactly the kind of silent
    convention error this module is most likely to contain.

    A 120-degree rotation about (1,1,1) permutes the axes cyclically, so with
    three distinct scales R sends the long axis to y and R^T sends it to z.
    """
    from gsplat.photogrammetry.level_set import gaussian_opacity_field

    field_fn, info = gaussian_opacity_field(
        _single_gaussian([0.40, 0.20, 0.10], quat=(0.5, 0.5, 0.5, 0.5)), device="cpu"
    )
    mesh, _ = extract_level_set(
        field_fn, np.array(info["bounds"]), resolution=48, probe=False
    )
    extent = np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(
        axis=0
    )

    # Premise: the three extents really are distinguishable, or a permutation
    # of them could not be detected.
    ordered = np.sort(extent)
    assert ordered[1] / ordered[0] > 1.5 and ordered[2] / ordered[1] > 1.5

    assert int(np.argmax(extent)) == 1, (
        f"the long axis landed on {'xyz'[int(np.argmax(extent))]}, expected y. "
        "A 120-degree rotation about (1,1,1) sends local x to world y; landing "
        "on z means the rotation is being applied transposed."
    )
    assert int(np.argmin(extent)) == 0
