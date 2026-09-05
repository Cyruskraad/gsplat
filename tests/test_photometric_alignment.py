"""Pins a documented negative result, so a future fix announces itself.

`gsplat.photogrammetry.photometric_alignment` implements Zhou & Koltun's
color-map optimization and **does not work well enough to ship** -- it is not
exported and not wired to any CLI. The module docstring carries the full
measurement trail. These tests pin the two claims that diagnosis rests on, so
that if someone repairs the appearance model the tests fail and say so, rather
than the finding quietly rotting into folklore.
"""

import os
import sys

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="open3d not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))


class _Views:
    """The duck-typed dataset contract: len + {camtoworld, K, image}."""

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


def _capture(tmp_path, pose_error_arcmin):
    from make_synthetic_capture import Config, build

    return build(
        Config(
            out_dir=str(tmp_path / f"cap{int(pose_error_arcmin)}"),
            num_views=10,
            width=96,
            height=96,
            focal=130.0,
            num_points=120,
            write_dense=False,
            mesh_resolution=14,
            pose_error_arcmin=pose_error_arcmin,
            seed=1,
        )
    )


def _mean_pose_error_arcmin(a, b):
    errors = []
    for i in range(len(a)):
        delta = a[i][:3, :3].T @ b[i][:3, :3]
        cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(np.degrees(np.arccos(cos)) * 60.0)
    return float(np.mean(errors))


def test_photometric_refinement_degrades_already_correct_poses(tmp_path):
    """The "do no harm" criterion, which this method fails.

    Given *exactly* correct cameras there is nothing to correct, so a sound
    refinement leaves them alone. This one moves them, because its objective's
    minimum is not at the truth. If this test starts failing, the appearance
    model was fixed and the module should be reconsidered for shipping --
    read the module docstring before deleting this.
    """
    from gsplat.photogrammetry.photometric_alignment import (
        refine_camera_poses_photometric,
    )

    capture = _capture(tmp_path, 0.0)
    mesh = o3d.io.read_triangle_mesh(capture["mesh_path"])
    mesh.compute_vertex_normals()

    # Premise: the input poses really are the ground truth, so any movement is
    # the method's own bias and not a recovered error. The bound is 1e-3' and
    # not zero because `arccos` near 1 amplifies float error -- identical
    # matrices score ~2e-5' through this formula, which is noise, not rotation.
    assert (
        _mean_pose_error_arcmin(capture["true_camtoworlds"], capture["camtoworlds"])
        < 1e-3
    )

    refined, stats = refine_camera_poses_photometric(
        mesh, _Views(capture), num_levels=2, iterations_per_level=30, pose_prior=1.0
    )
    drift = _mean_pose_error_arcmin(capture["true_camtoworlds"], refined)

    assert drift > 0.5, (
        "photometric refinement no longer degrades correct poses "
        f"(drift {drift:.2f}'). That is the documented failure being fixed -- "
        "re-measure the table in the module docstring and reconsider exporting it."
    )
    assert stats["mean_pose_correction_arcmin"] > 0.0


def test_the_objective_prefers_a_pose_that_is_not_the_truth(tmp_path):
    """The diagnosis itself: the minimum is in the wrong place.

    This is what rules out "the optimiser is broken" and rules in "the
    objective is wrong", so it is the claim most worth pinning. Scored with the
    fused colour **re-estimated at each pose**, which is the true joint
    objective rather than the one-sided one the inner loop sees.
    """
    from gsplat.photogrammetry import photometric_alignment as pa
    from gsplat.photogrammetry.texturing import _bake_points_from_views

    capture = _capture(tmp_path, 0.0)
    mesh = o3d.io.read_triangle_mesh(capture["mesh_path"])
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    K = capture["K"]
    views = _Views(capture)

    def joint_objective(poses):
        dataset = pa._PoseOverride(views, poses)
        color, weight = _bake_points_from_views(
            mesh, dataset, points, normals, max_views=None
        )
        seen = weight > 0
        target = np.zeros_like(color)
        target[seen] = color[seen] / weight[seen][:, None]
        visible = pa._visible_point_indices(mesh, dataset, points, normals, None)
        total, count = 0.0, 0
        for view, (idx, _w) in visible.items():
            idx = idx[seen[idx]]
            if idx.size == 0:
                continue
            c2w = poses[view]
            local = (points[idx] - c2w[:3, 3]) @ c2w[:3, :3]
            uv = (local @ K.T)[:, :2] / local[:, 2:3]
            image = np.asarray(views[view]["image"], dtype=np.float64) / 255.0
            sampled = pa._bilinear_torch(torch.tensor(image), torch.tensor(uv)).numpy()
            total += np.abs(sampled - target[idx]).sum()
            count += idx.size * 3
        return total / max(count, 1)

    truth = capture["true_camtoworlds"]
    refined, _ = pa.refine_camera_poses_photometric(
        mesh, views, num_levels=2, iterations_per_level=30, pose_prior=0.0
    )

    at_truth = joint_objective(truth)
    at_refined = joint_objective(refined)

    # Premise: the refined poses really did move, or the comparison is trivial.
    assert _mean_pose_error_arcmin(truth, refined) > 1.0
    assert at_refined < at_truth, (
        f"the objective now prefers the truth ({at_truth:.5f}) over the pose it "
        f"converges to ({at_refined:.5f}). The documented diagnosis no longer "
        "holds -- re-read the module docstring."
    )
