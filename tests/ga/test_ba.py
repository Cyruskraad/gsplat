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
"""Bundle adjustment, and the control that makes it mean something.

The load-bearing test here is :class:`TestGateOne`. Motors are isomorphic to
unit dual quaternions and PGA bivectors are exactly se(3), so the two arms are
the same estimator in different coordinates and **must** reach the same optimum.
If they ever disagree beyond the gauge, the geometric-algebra code has a bug --
there is no "GA behaves differently" explanation available.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import camera as cam
from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga.baseline import ba as qt_ba
from gsplat.contrib.ga.sfm import ba as ga_ba
from tests.ga._helpers import bundle_problem

DTYPE = torch.float64


def _camera_centres(poses: torch.Tensor) -> torch.Tensor:
    matrices = qt_ba.poses_to_matrices(poses)
    return -torch.einsum(
        "vij,vj->vi", matrices[:, :3, :3].transpose(-2, -1), matrices[:, :3, 3]
    )


class TestJacobians:
    def test_motor_pose_jacobian_matches_autograd(self):
        """``dp/dw = 2 skew(p)`` and ``dp/dv = -2 I`` for a left increment."""
        torch.manual_seed(0)
        base = mot.motor_exp(torch.randn(6, dtype=DTYPE) * 0.3)
        points = torch.randn(5, 3, dtype=DTYPE)

        def fn(delta):
            return mot.motor_apply_point(
                mot.motor_compose(mot.motor_exp(delta), base).expand(5, 8), points
            )

        auto = torch.autograd.functional.jacobian(fn, torch.zeros(6, dtype=DTYPE))
        cam_points = mot.motor_apply_point(base.expand(5, 8), points)
        eye = torch.eye(3, dtype=DTYPE).expand(5, 3, 3)
        predicted = torch.cat([2.0 * ga_ba._skew(cam_points), -2.0 * eye], dim=-1)
        torch.testing.assert_close(auto, predicted, atol=1e-12, rtol=0)

    def test_baseline_pose_jacobian_matches_autograd(self):
        """``dp/domega = -skew(p)`` and ``dp/du = I`` for the se(3) increment."""
        torch.manual_seed(1)
        quat = torch.randn(4, dtype=DTYPE)
        quat = quat / quat.norm()
        translation = torch.randn(3, dtype=DTYPE)
        points = torch.randn(5, 3, dtype=DTYPE)

        def fn(delta):
            omega, u = delta[:3], delta[3:]
            dq = qt_ba.rotation_from_axis_angle(omega)
            new_q = qt_ba.quat_multiply(dq, quat)
            new_t = qt_ba.quat_rotate(dq, translation) + u
            return qt_ba.quat_rotate(new_q.expand(5, 4), points) + new_t

        auto = torch.autograd.functional.jacobian(fn, torch.zeros(6, dtype=DTYPE))
        cam_points = qt_ba.quat_rotate(quat.expand(5, 4), points) + translation
        eye = torch.eye(3, dtype=DTYPE).expand(5, 3, 3)
        predicted = torch.cat([-qt_ba._skew(cam_points), eye], dim=-1)
        torch.testing.assert_close(auto, predicted, atol=1e-12, rtol=0)


class TestBaselineIsIndependent:
    """The control is written from scratch; these check it against the GA code.

    A control that merely wraps the code under test cannot detect a bug in it,
    so the quaternion path has its own product, rotation and exponential. That
    independence is only worth having if both agree on ground truth.
    """

    def test_quaternion_rotation_matches_motor_rotation(self):
        torch.manual_seed(2)
        motors = mot.motor_exp(torch.randn(16, 6, dtype=DTYPE) * 0.4)
        poses = qt_ba.poses_from_matrices(mot.motor_to_matrix(motors))
        points = torch.randn(16, 3, dtype=DTYPE)
        torch.testing.assert_close(
            qt_ba.quat_rotate(poses[:, :4], points) + poses[:, 4:],
            cam.world_to_camera(motors, points),
            atol=1e-10,
            rtol=0,
        )

    def test_pose_matrix_round_trip(self):
        torch.manual_seed(3)
        motors = mot.motor_exp(torch.randn(32, 6, dtype=DTYPE) * 0.4)
        matrices = mot.motor_to_matrix(motors)
        torch.testing.assert_close(
            qt_ba.poses_to_matrices(qt_ba.poses_from_matrices(matrices)),
            matrices,
            atol=1e-10,
            rtol=0,
        )

    def test_both_arms_start_from_identical_geometry(self):
        ga_problem, qt_problem, _ = bundle_problem(views=5, points=64)
        torch.testing.assert_close(
            ga_ba.reprojection_residuals(ga_problem),
            qt_ba.reprojection_residuals(qt_problem),
            atol=1e-9,
            rtol=0,
        )


class TestConvergence:
    def test_recovers_ground_truth_from_a_perturbed_start(self):
        ga_problem, _, (_, _, world) = bundle_problem(views=8, points=300)
        start = float(ga_ba.reprojection_residuals(ga_problem).pow(2).sum(-1).mean().sqrt())
        assert start > 1.0  # the start really is perturbed

        refined, stats = ga_ba.bundle_adjust(ga_problem, iterations=50)
        assert stats["rmse"] < 1e-9
        _, point_error = ga_ba.align_similarity(refined.points, world)
        assert point_error < 1e-9

    def test_cost_is_monotone(self):
        ga_problem, _, _ = bundle_problem(views=6, points=128)
        _, stats = ga_ba.bundle_adjust(ga_problem, iterations=30)
        costs = stats["cost"]
        assert all(b <= a for a, b in zip(costs, costs[1:]))

    def test_fixed_camera_does_not_move(self):
        """The gauge anchor must be held exactly, or the comparison drifts."""
        ga_problem, _, _ = bundle_problem(views=6, points=128)
        before = ga_problem.motors[0].clone()
        refined, _ = ga_ba.bundle_adjust(ga_problem, iterations=20, fixed_cameras=(0,))
        torch.testing.assert_close(refined.motors[0], before, atol=1e-12, rtol=0)

    def test_noisy_observations_converge_near_the_noise_floor(self):
        sigma = 1.0
        ga_problem, _, _ = bundle_problem(views=8, points=400, pixel_noise=sigma)
        _, stats = ga_ba.bundle_adjust(ga_problem, iterations=50)
        # Cannot fit below the noise; should not sit far above it either.
        assert 0.5 * sigma < stats["rmse"] < 2.0 * sigma


class TestGateOne:
    """GA and the quaternion control must reach the same optimum."""

    def test_arms_reach_the_same_cost(self):
        ga_problem, qt_problem, _ = bundle_problem(views=8, points=300)
        _, ga_stats = ga_ba.bundle_adjust(ga_problem, iterations=50)
        _, qt_stats = qt_ba.bundle_adjust(qt_problem, iterations=50)
        assert abs(ga_stats["final_cost"] - qt_stats["final_cost"]) < 1e-12

    def test_arms_agree_on_rotations(self):
        """Rotations are gauge-free -- a similarity leaves them untouched -- so
        they can be compared directly, with no alignment to hide behind."""
        ga_problem, qt_problem, _ = bundle_problem(views=8, points=300)
        ga_out, _ = ga_ba.bundle_adjust(ga_problem, iterations=50)
        qt_out, _ = qt_ba.bundle_adjust(qt_problem, iterations=50)
        torch.testing.assert_close(
            mot.motor_to_matrix(ga_out.motors)[:, :3, :3],
            qt_ba.poses_to_matrices(qt_out.poses)[:, :3, :3],
            atol=1e-10,
            rtol=0,
        )

    def test_arms_agree_on_structure_up_to_the_gauge(self):
        ga_problem, qt_problem, _ = bundle_problem(views=8, points=300)
        ga_out, _ = ga_ba.bundle_adjust(ga_problem, iterations=50)
        qt_out, _ = qt_ba.bundle_adjust(qt_problem, iterations=50)
        _, error = ga_ba.align_similarity(ga_out.points, qt_out.points)
        assert error < 1e-10

    def test_the_only_camera_disagreement_is_scale(self):
        """Pinning camera 0 removes six of the seven similarity degrees of
        freedom; scale is the one left. So raw camera centres may differ, but
        only by a constant factor, and alignment must remove it entirely.
        Recorded because the raw difference (~2e-5) looks alarming next to the
        structure agreement (~1e-16) until you know why.

        The ratio spread is checked loosely and the alignment tightly, on
        purpose. Both arms drive the cost to ~1e-23, and near that optimum the
        scale direction of the Hessian is almost flat, so the scale gauge is
        only loosely pinned and tiny numerical differences move it. A genuine
        non-scale disagreement would show up as ratios differing by orders of
        magnitude, not in the fifth decimal."""
        ga_problem, qt_problem, _ = bundle_problem(views=8, points=300)
        ga_out, _ = ga_ba.bundle_adjust(ga_problem, iterations=50)
        qt_out, _ = qt_ba.bundle_adjust(qt_problem, iterations=50)

        ga_centres = cam.camera_center(ga_out.motors)
        qt_centres = _camera_centres(qt_out.poses)
        ratios = qt_centres[1:].norm(dim=-1) / ga_centres[1:].norm(dim=-1)
        assert float(ratios.max() - ratios.min()) < 1e-3  # one scale factor, not many
        assert abs(float(ratios.mean()) - 1.0) < 1e-3

        _, error = ga_ba.align_similarity(ga_centres, qt_centres)
        assert error < 1e-10


class TestAlignSimilarity:
    def test_recovers_a_known_similarity(self):
        torch.manual_seed(7)
        target = torch.randn(64, 3, dtype=DTYPE)
        rotation = mot.motor_to_matrix(mot.motor_exp(torch.randn(6, dtype=DTYPE) * 0.5))[:3, :3]
        source = 3.7 * (target @ rotation.T) + torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
        aligned, error = ga_ba.align_similarity(source, target)
        assert error < 1e-10
        torch.testing.assert_close(aligned, target, atol=1e-9, rtol=0)

    def test_identity_alignment_is_a_no_op(self):
        torch.manual_seed(8)
        points = torch.randn(32, 3, dtype=DTYPE)
        aligned, error = ga_ba.align_similarity(points, points)
        assert error < 1e-12
        torch.testing.assert_close(aligned, points, atol=1e-10, rtol=0)
