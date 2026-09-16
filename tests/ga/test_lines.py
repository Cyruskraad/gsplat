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
"""Line features: claim C1, that points and lines need only one residual path.

The argument for building structure from motion in geometric algebra is not
accuracy -- motors are isomorphic to dual quaternions, so no accuracy is
available. It is that ``plane ^ entity`` is one expression that serves a point
and a line alike, differing only in the grade of its operand. These tests hold
that claim to something checkable: the same wedge, the same solver, the same
motor increment, and no Pluecker bookkeeping or second Jacobian derivation
anywhere.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import camera as cam
from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga import primitives as prim
from gsplat.contrib.ga.sfm import ba
from tests.ga._helpers import line_scene, recover_similarity

DTYPE = torch.float64


class TestLinePlaneIncidence:
    def test_offset_equals_the_geometric_distance(self):
        """For a normalized plane and line, the wedge magnitude *is* the offset."""
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE)  # z = 0
        origin = torch.tensor([1.0, 2.0, 0.0], dtype=DTYPE)
        direction = torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE)
        for offset in (0.0, 0.1, 0.5, 2.0):
            shifted = origin + torch.tensor([0.0, 0.0, offset], dtype=DTYPE)
            line = prim.line_from_point_direction(shifted, direction)
            got = float(prim.line_plane_distance(line, plane))
            assert abs(got - offset) < 1e-12

    def test_residual_is_smooth_at_incidence(self):
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE)
        origin = torch.tensor([1.0, 2.0, 0.0], dtype=DTYPE, requires_grad=True)
        direction = torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE)
        line = prim.line_from_point_direction(origin, origin + direction - origin)
        prim.line_plane_residual(line, plane).pow(2).sum().backward()
        assert torch.isfinite(origin.grad).all()

    def test_incidence_residual_dispatches_on_grade(self):
        """One entry point; a point gives one number, a line gives four."""
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE)
        point = torch.tensor([1.0, 2.0, 0.7], dtype=DTYPE)
        line = prim.line_from_point_direction(
            torch.tensor([1.0, 2.0, 0.3], dtype=DTYPE),
            torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE),
        )
        assert prim.incidence_residual(plane, point).shape == (1,)
        assert prim.incidence_residual(plane, line).shape == (4,)
        assert abs(float(prim.incidence_residual(plane, point)) - 0.7) < 1e-12
        assert abs(float(prim.incidence_residual(plane, line).norm()) - 0.3) < 1e-12

    def test_rejects_an_unknown_grade(self):
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE)
        try:
            prim.incidence_residual(plane, torch.zeros(5, dtype=DTYPE))
        except ValueError as error:
            assert "point" in str(error) and "line" in str(error)
        else:
            raise AssertionError("expected a ValueError for an unsupported shape")


class TestLineProjection:
    def test_projection_round_trips_through_the_interpretation_plane(self):
        problem, motors, world_lines = line_scene(views=4, lines=12)
        planes = cam.image_line_plane(
            problem.intrinsics[problem.camera_idx], problem.image_lines
        )
        cam_lines = mot.motor_apply_line(
            problem.motors[problem.camera_idx], world_lines[problem.line_idx]
        )
        assert float(prim.line_plane_distance(cam_lines, planes).abs().max()) < 1e-9

    def test_projected_points_of_a_line_lie_on_the_projected_line(self):
        problem, motors, world_lines = line_scene(views=3, lines=8)
        line = world_lines[0]
        direction = prim.line_direction(prim.normalize_line(line))
        foot = torch.linalg.cross(direction, prim.line_moment(prim.normalize_line(line)))
        samples = torch.stack([foot + t * direction for t in (-1.5, 0.0, 2.0)])

        for v in range(3):
            image_line = cam.project_line(problem.motors[v], problem.intrinsics[v], line)
            pixels, valid = cam.project(
                problem.motors[v].expand(3, 8), problem.intrinsics[v].expand(3, 3, 3), samples
            )
            residual = pixels @ image_line[:2] + image_line[2]
            assert float(residual[valid].abs().max()) < 1e-8


class TestLineBundleAdjustment:
    @staticmethod
    def _perturbed(seed: int = 0, views: int = 6, lines: int = 40):
        problem, motors, world_lines = line_scene(views=views, lines=lines, seed=seed)
        gen = torch.Generator().manual_seed(seed + 99)
        delta_cam = torch.randn(views, 6, generator=gen, dtype=DTYPE) * 0.015
        delta_cam[0] = 0.0
        delta_line = torch.randn(lines, 6, generator=gen, dtype=DTYPE) * 0.03
        from dataclasses import replace

        start = replace(
            problem,
            motors=mot.motor_compose(mot.motor_exp(delta_cam), problem.motors),
            line_motors=mot.motor_compose(mot.motor_exp(delta_line), problem.line_motors),
        )
        return start, motors, world_lines

    def test_residual_vanishes_at_ground_truth(self):
        problem, _, _ = line_scene(views=6, lines=40)
        assert float(ba.line_residuals(problem).abs().max()) < 1e-9

    def test_converges(self):
        start, _, _ = self._perturbed()
        before = float(ba.line_residuals(start).pow(2).sum(-1).mean().sqrt())
        assert before > 0.01
        _, stats = ba.bundle_adjust_lines(start, iterations=40)
        assert stats["rmse"] < 1e-9

    def test_recovers_geometry_up_to_a_similarity(self):
        """The gauge lesson, made a test.

        Line-only bundle adjustment fixes geometry only up to a global
        similarity, and pinning one camera does not pin the scale. Comparing raw
        coordinates against ground truth therefore measures the gauge: here the
        lines look ~0.57 off until the similarity is removed, after which they
        agree to ~1e-13. The similarity is recovered from the *cameras* and then
        applied to the *lines*, so the two must be consistent -- a much stronger
        check than aligning each independently.
        """
        start, true_motors, true_lines = self._perturbed()
        refined, stats = ba.bundle_adjust_lines(start, iterations=40)
        assert stats["rmse"] < 1e-9

        recovered_centres = cam.camera_center(refined.motors)
        true_centres = cam.camera_center(true_motors)
        _, error = ba.align_similarity(recovered_centres, true_centres)
        assert error < 1e-9

        rotation, scale, src_mean, dst_mean = recover_similarity(
            recovered_centres, true_centres
        )

        def transform(points):
            return scale * ((points - src_mean) @ rotation.T) + dst_mean

        recovered = prim.normalize_line(refined.world_lines())
        directions = prim.line_direction(recovered)
        feet = torch.linalg.cross(directions, prim.line_moment(recovered), dim=-1)
        moved = prim.normalize_line(
            prim.line_from_point_direction(
                transform(feet), transform(feet + directions) - transform(feet)
            )
        )
        truth = prim.normalize_line(true_lines)
        gap = torch.minimum(
            (moved - truth).abs().amax(-1), (moved + truth).abs().amax(-1)
        )
        assert float(gap.max()) < 1e-9

    def test_line_directions_are_gauge_free_and_exact(self):
        """Directions are scale-invariant, so they must match with no alignment."""
        start, _, true_lines = self._perturbed()
        refined, _ = ba.bundle_adjust_lines(start, iterations=40)
        got = prim.line_direction(prim.normalize_line(refined.world_lines()))
        want = prim.line_direction(prim.normalize_line(true_lines))
        agree = torch.minimum((got - want).abs().amax(-1), (got + want).abs().amax(-1))
        assert float(agree.max()) < 1e-9

    def test_cost_is_monotone(self):
        start, _, _ = self._perturbed(views=4, lines=16)
        _, stats = ba.bundle_adjust_lines(start, iterations=25)
        costs = stats["cost"]
        assert all(b <= a for a, b in zip(costs, costs[1:]))


class TestConditioning:
    def test_null_space_is_exactly_the_expected_redundancy(self):
        """The system is rank-deficient by design, and by exactly how much.

        One direction is the global scale (the similarity gauge, less the six
        removed by pinning camera 0), plus two per line: a PGA line has four
        degrees of freedom but is carried by a six-parameter motor, and the
        screw motions that slide a line along itself leave it unchanged. Any
        *excess* nullity would mean a genuine geometric ambiguity rather than a
        parameterization artifact, so this is the test that distinguishes the
        two.
        """
        for views, lines in ((6, 40), (3, 20), (10, 60)):
            problem, _, _ = line_scene(views=views, lines=lines)
            residual, jac_cam, jac_line = ba._line_residual_and_jacobians(problem)
            assert float(residual.abs().max()) < 1e-9

            num = residual.shape[0]
            full = torch.zeros(num * 4, 6 * views + 6 * lines, dtype=DTYPE)
            for m in range(num):
                c = int(problem.camera_idx[m])
                l = int(problem.line_idx[m])
                full[m * 4 : (m + 1) * 4, 6 * c : 6 * c + 6] = jac_cam[m]
                full[m * 4 : (m + 1) * 4, 6 * views + 6 * l : 6 * views + 6 * l + 6] = jac_line[m]

            keep = torch.ones(full.shape[1], dtype=torch.bool)
            keep[:6] = False  # camera 0 is pinned, as in the solver
            singular = torch.linalg.svdvals(full[:, keep])
            nullity = int((singular < singular.max() * 1e-10).sum())
            assert nullity == 1 + 2 * lines, (views, lines, nullity)
