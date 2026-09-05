"""Photometric mesh refinement (Vu et al., TPAMI 2012).

The claims pinned here are deliberately two-sided: this reduces surface error
when there is error, and *adds* about 0.15 source pixels of noise when there is
not. Asserting only the first would let a later reader believe it is free.

Contrast `tests/test_photometric_alignment.py`, which pins a method that does
not work. The difference that decided one shipped and the other did not: this
one's correction is proportional to the input error and exceeds the noise it
adds above ~0.4 px, where the alignment converged to an attractor independent
of the input.
"""

import os
import sys

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="open3d not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))

CAM_DIST = 3.5
FOCAL = 210.0
PIXEL_WORLD = CAM_DIST / FOCAL  # world size of one source pixel


class _Views:
    def __init__(self, capture):
        import imageio.v2 as imageio

        self._c = capture["camtoworlds"]
        self._K = capture["K"]
        self._images = [
            imageio.imread(os.path.join(capture["images_dir"], name)).astype(np.float32)
            for name in capture["image_names"]
        ]

    def __len__(self):
        return len(self._c)

    def __getitem__(self, index):
        return {
            "camtoworld": torch.tensor(self._c[index], dtype=torch.float32),
            "K": torch.tensor(self._K, dtype=torch.float32),
            "image": torch.tensor(self._images[index]),
        }


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from make_synthetic_capture import Config, build

    # 10 views is deliberate, not incidental: measured, 8 views recovers only
    # 4-12% of the same error where 10 recovers 26.7%. A cheaper fixture would
    # make this file test a regime the method does not work in.
    capture = build(
        Config(
            out_dir=str(tmp_path_factory.mktemp("refine")),
            num_views=10,
            width=160,
            height=160,
            focal=FOCAL,
            cam_dist=CAM_DIST,
            num_points=120,
            write_dense=False,
            mesh_resolution=20,
            seed=1,
        )
    )
    truth = o3d.io.read_triangle_mesh(capture["mesh_path"])
    truth.compute_vertex_normals()
    return capture, truth, _Views(capture)


def _radial_error_px(mesh):
    """Mean |distance from origin - 1| in source pixels."""
    radii = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    return float(np.abs(radii - 1.0).mean()) / PIXEL_WORLD


def _perturbed(truth, scale, seed=4):
    vertices = np.asarray(truth.vertices).copy()
    mesh = o3d.geometry.TriangleMesh(truth)
    if scale > 0:
        offsets = np.random.default_rng(seed).normal(
            scale=scale, size=(len(vertices), 1)
        )
        directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector(vertices + offsets * directions)
    mesh.compute_vertex_normals()
    return mesh


def test_refinement_pulls_a_noisy_surface_back_toward_the_truth(scene):
    """The claim it exists for, against an analytic sphere."""
    from gsplat.photogrammetry import refine_mesh_photometric

    _capture, truth, views = scene
    noisy = _perturbed(truth, 0.02)

    # Premise: the perturbation is real and well above the ~0.4 px break-even,
    # or "it improved" would be measuring noise.
    before = _radial_error_px(noisy)
    assert before > 0.7, f"perturbation only {before:.2f} px -- too small to test"

    refined, stats = refine_mesh_photometric(
        noisy, views, iterations=20, outer_rounds=4, smoothness=1.0
    )
    after = _radial_error_px(refined)

    assert after < before * 0.9, (
        f"refinement recovered only {100 * (1 - after / before):.1f}% of "
        f"{before:.2f} px of error"
    )
    # It must also improve the thing it actually optimises.
    assert stats["photoconsistency_after"] < stats["photoconsistency_before"]
    # And it must not have simply collapsed the mesh toward the origin.
    assert stats["max_abs_displacement"] <= stats["max_displacement"] + 1e-12


def test_refinement_costs_accuracy_on_an_already_correct_surface(scene):
    """The other half of the truth: below break-even this adds noise.

    Documented as ~0.15 source pixels. Pinned as a *bound*, not as "no harm",
    because claiming no harm would be false.
    """
    from gsplat.photogrammetry import refine_mesh_photometric

    _capture, truth, views = scene

    # Premise: the input really is the ground-truth surface.
    assert _radial_error_px(truth) < 1e-6

    refined, _ = refine_mesh_photometric(
        o3d.geometry.TriangleMesh(truth), views, iterations=20, outer_rounds=4
    )
    damage = _radial_error_px(refined)

    assert damage > 0.01, (
        "refinement no longer perturbs a correct surface -- the documented "
        "break-even may be obsolete; re-measure the table in the module docstring"
    )
    assert damage < 0.4, (
        f"refinement added {damage:.2f} px to a correct surface, well past the "
        "~0.15 px documented; something regressed"
    )


def test_an_oversized_patch_biases_the_surface_inward(scene):
    """A flat tangent patch is a chord, so a big patch shrinks a convex object.

    This is why `patch_spacing` is derived from the source pixel size rather
    than left to taste, and it is the failure mode a later reader would create
    by enlarging the patch "for robustness".
    """
    from gsplat.photogrammetry import refine_mesh_photometric

    _capture, truth, views = scene
    mean_radius = lambda m: float(np.linalg.norm(np.asarray(m.vertices), axis=1).mean())

    derived, _ = refine_mesh_photometric(
        o3d.geometry.TriangleMesh(truth), views, iterations=15, outer_rounds=2
    )
    oversized, _ = refine_mesh_photometric(
        o3d.geometry.TriangleMesh(truth),
        views,
        iterations=15,
        outer_rounds=2,
        patch_spacing=6.0 * PIXEL_WORLD,
    )

    assert mean_radius(oversized) < mean_radius(derived), (
        "an oversized patch no longer biases inward -- the derived patch "
        "spacing may no longer be load-bearing; re-measure"
    )
    assert mean_radius(oversized) < 1.0


def test_refinement_needs_views_that_can_disagree(scene):
    """One view cannot be cross-checked against anything."""
    from gsplat.photogrammetry import refine_mesh_photometric

    _capture, truth, views = scene
    with pytest.raises(ValueError, match="at least"):
        refine_mesh_photometric(truth, views, max_views=1)


def test_refinement_survives_per_view_exposure_differences(tmp_path):
    """The z-normalisation is the whole reason this works where 2a did not.

    `photometric_alignment` compares each view against one fused colour, so a
    view that is simply brighter looks like a view whose geometry is wrong, and
    the optimiser moves the surface to explain the brightness. Z-normalising
    each patch before comparison makes the objective a correlation, so a
    per-view gain and offset cancel exactly.

    Without exposure variation the two are indistinguishable -- which is why
    this test exists separately and builds its own capture. Dropping the
    normalisation passes every other test in this file.
    """
    from gsplat.photogrammetry import refine_mesh_photometric
    from make_synthetic_capture import Config, build

    capture = build(
        Config(
            out_dir=str(tmp_path / "exposed"),
            num_views=10,
            width=160,
            height=160,
            focal=FOCAL,
            cam_dist=CAM_DIST,
            num_points=120,
            write_dense=False,
            mesh_resolution=20,
            exposure=0.15,
            seed=1,
        )
    )
    truth = o3d.io.read_triangle_mesh(capture["mesh_path"])
    truth.compute_vertex_normals()
    views = _Views(capture)
    noisy = _perturbed(truth, 0.02)

    # Premise: the views really do disagree on brightness, or there is nothing
    # for the normalisation to cancel and this repeats an earlier test.
    means = [float(np.asarray(views[i]["image"]).mean()) for i in range(len(views))]
    assert max(means) - min(means) > 2.0, (
        f"per-view brightness spread is only {max(means) - min(means):.2f}/255 "
        "-- the capture has no exposure variation to be robust to"
    )

    before = _radial_error_px(noisy)
    refined, _stats = refine_mesh_photometric(
        noisy, views, iterations=20, outer_rounds=4, smoothness=1.0
    )
    after = _radial_error_px(refined)

    assert after < before * 0.9, (
        f"with per-view exposure the refinement recovered only "
        f"{100 * (1 - after / before):.1f}% of {before:.2f} px -- exposure is "
        "being blamed on geometry, so the patches are not being normalised"
    )
