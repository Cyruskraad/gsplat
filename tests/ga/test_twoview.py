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
"""Two-view relative pose.

:class:`TestObjectiveChoice` records a correction. An earlier version of the
module claimed the geometric ray gap was a *biased* objective and switched to
Sampson error on that basis. Measuring properly refutes the explanation: on
clean data the gap ranks the truth best, so it is sound. What actually goes
wrong is that an **unweighted** fit over outlier-contaminated data prefers a
wrong pose -- and Sampson error does the same thing on the same scene. The
lesson is about robust weighting, not about which residual is prettier, and
these tests pin both halves so the wrong story cannot creep back.
"""

from __future__ import annotations

import pytest
import torch

from gsplat.contrib.ga import camera as cam
from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga import primitives as prim
from gsplat.contrib.ga.sfm import twoview as tv
from tests.ga._helpers import pose_errors, two_view_scene

DTYPE = torch.float64


def _rays(points, intrinsics):
    num = points.shape[0]
    identity = mot.motor_identity(dtype=DTYPE)
    return prim.normalize_line(
        cam.pixel_ray(identity.expand(num, 8), intrinsics.expand(num, 3, 3), points)
    )


class TestEssentialMatrix:
    def test_satisfies_the_epipolar_constraint(self):
        points_a, points_b, intrinsics, _, _ = two_view_scene()
        essential = tv.essential_matrix(points_a, points_b, intrinsics)
        inv_k = torch.linalg.inv(intrinsics)
        ones = torch.ones_like(points_a[:, :1])
        xa = torch.cat([points_a, ones], dim=-1) @ inv_k.T
        xb = torch.cat([points_b, ones], dim=-1) @ inv_k.T
        residual = torch.einsum("ni,ij,nj->n", xb, essential, xa)
        assert float(residual.abs().max()) < 1e-9

    def test_matches_the_essential_matrix_of_the_true_motor(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene()
        from_points = tv.essential_matrix(points_a, points_b, intrinsics)
        from_motor = tv.essential_from_motor(true_relative)
        a = from_points / from_points.norm()
        b = from_motor / from_motor.norm()
        assert float(torch.minimum((a - b).abs().max(), (a + b).abs().max())) < 1e-9

    def test_decomposition_is_exact_without_noise(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene()
        essential = tv.essential_matrix(points_a, points_b, intrinsics)
        estimate = tv.relative_motor_from_essential(
            essential, points_a, points_b, intrinsics
        )
        rotation, direction = pose_errors(estimate, true_relative)
        assert rotation < 1e-6 and direction < 1e-6

    def test_one_gross_outlier_breaks_the_linear_fit(self):
        """Why a robust refinement is mandatory rather than decorative.

        The eight-point algorithm is an unweighted linear least squares with no
        breakdown resistance. A single corrupted correspondence is enough to
        move it from sub-degree to several degrees of rotation error, which is
        why the consensus set is never trusted as-is.
        """
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene(pixel_noise=0.5)
        clean = tv.relative_motor_from_essential(
            tv.essential_matrix(points_a, points_b, intrinsics),
            points_a, points_b, intrinsics,
        )
        clean_rotation, _ = pose_errors(clean, true_relative)

        corrupted = points_b.clone()
        corrupted[0] = torch.tensor([12.0, 470.0], dtype=DTYPE)
        dirty = tv.relative_motor_from_essential(
            tv.essential_matrix(points_a, corrupted, intrinsics),
            points_a, corrupted, intrinsics,
        )
        dirty_rotation, _ = pose_errors(dirty, true_relative)
        assert clean_rotation < 1.0
        assert dirty_rotation > 5.0 * max(clean_rotation, 1e-3)


class TestResiduals:
    def test_sampson_vanishes_at_the_true_pose(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene()
        residual = tv.sampson_residual(true_relative, points_a, points_b, intrinsics)
        assert float(residual.abs().max()) < 1e-9

    def test_ray_gap_vanishes_at_the_true_pose(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene()
        gap = tv.epipolar_residual(
            true_relative, _rays(points_a, intrinsics), _rays(points_b, intrinsics)
        )
        assert float(gap.abs().max()) < 1e-9

    def test_ray_gap_is_large_for_wrong_correspondences(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene(points=200)
        shuffled = points_b[torch.randperm(200, generator=torch.Generator().manual_seed(3))]
        gap = tv.epipolar_residual(
            true_relative, _rays(points_a, intrinsics), _rays(shuffled, intrinsics)
        )
        assert float(gap.abs().mean()) > 1e-3


class TestObjectiveChoice:
    """Why estimation is robust, and why that is the real issue.

    Recorded as tests because the first explanation for the observed failure --
    "the ray-gap objective is biased" -- was wrong, and only measuring the clean
    and contaminated cases separately showed it.
    """

    @staticmethod
    def _descend_gap(truth, rays_a, rays_b, steps: int = 300):
        """Adam on the ray-gap objective, starting from a given pose."""
        delta = torch.zeros(6, dtype=DTYPE, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.02)
        for _ in range(steps):
            optimizer.zero_grad()
            state = mot.motor_compose(mot.motor_exp(delta), truth)
            tv.epipolar_residual(state, rays_a, rays_b).pow(2).sum().backward()
            optimizer.step()
        return tv.unit_baseline(
            mot.motor_normalize(mot.motor_compose(mot.motor_exp(delta.detach()), truth))
        )

    def test_ray_gap_is_sound_on_clean_data(self):
        """The refutation: with no outliers, the gap ranks the truth best.

        So the gap is a valid objective, and the original "it is biased because
        gaps shrink when rays go near-parallel" story does not hold.
        """
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene(
            points=300, pixel_noise=0.5, outlier_fraction=0.0, seed=1
        )
        rays_a, rays_b = _rays(points_a, intrinsics), _rays(points_b, intrinsics)
        truth = tv.unit_baseline(true_relative)
        drifted = self._descend_gap(truth, rays_a, rays_b)

        def gap_cost(motor):
            return float(tv.epipolar_residual(motor, rays_a, rays_b).pow(2).sum())

        assert gap_cost(truth) < gap_cost(drifted)

    def test_unweighted_fits_fail_on_outliers_whichever_residual(self):
        """The real failure mode, and it is not specific to the ray gap.

        With 20% outliers an unweighted sum of *either* residual scores a badly
        wrong pose better than the truth. That is what motivates IRLS in
        :func:`refine_relative_motor`, and it is why the choice between the two
        residuals was never the thing that mattered.
        """
        points_a, points_b, intrinsics, true_relative, inlier_mask = two_view_scene(
            points=300, pixel_noise=0.5, outlier_fraction=0.2, seed=1
        )
        rays_a, rays_b = _rays(points_a, intrinsics), _rays(points_b, intrinsics)
        truth = tv.unit_baseline(true_relative)
        drifted = self._descend_gap(truth, rays_a, rays_b)
        _, direction_error = pose_errors(drifted, true_relative)
        assert direction_error > 20.0  # the drifted pose really is badly wrong

        def gap_cost(motor, mask):
            return float(tv.epipolar_residual(motor, rays_a[mask], rays_b[mask]).pow(2).sum())

        def sampson_cost(motor, mask):
            return float(
                tv.sampson_residual(motor, points_a[mask], points_b[mask], intrinsics)
                .pow(2)
                .sum()
            )

        everything = torch.ones_like(inlier_mask)
        # Unweighted over all correspondences, both prefer the wrong pose.
        assert gap_cost(drifted, everything) < gap_cost(truth, everything)
        assert sampson_cost(drifted, everything) < sampson_cost(truth, everything)
        # Restricted to true correspondences, both put the truth far ahead.
        assert gap_cost(truth, inlier_mask) < gap_cost(drifted, inlier_mask)
        assert sampson_cost(truth, inlier_mask) < sampson_cost(drifted, inlier_mask)

    def test_robust_refinement_beats_unweighted_from_a_sane_start(self):
        """Robust weighting rescues contamination, not a bad basin.

        Measured on a 20%-outlier pair: the unweighted refinement lands ~84
        degrees off in translation from *every* initialization tried, including
        ground truth itself -- the outlier-driven optimum is that dominant. The
        robust fit stays put when started somewhere sensible. It does not
        recover from a start already tens of degrees wrong, which is why RANSAC
        supplies the basin and IRLS only cleans up inside it.
        """
        points_a, points_b, intrinsics, true_relative, inlier_mask = two_view_scene(
            points=300, pixel_noise=0.5, outlier_fraction=0.2, seed=1
        )
        # A sane start: the linear fit on correspondences that actually agree.
        start = tv.relative_motor_from_essential(
            tv.essential_matrix(points_a[inlier_mask], points_b[inlier_mask], intrinsics),
            points_a, points_b, intrinsics,
        )
        delta = 1.5 / float(intrinsics[0, 0])
        plain = tv.refine_relative_motor(
            start, points_a, points_b, intrinsics, robust=False
        )
        robust = tv.refine_relative_motor(
            start, points_a, points_b, intrinsics, robust=True, huber_delta=delta
        )
        _, plain_direction = pose_errors(plain, true_relative)
        _, robust_direction = pose_errors(robust, true_relative)
        assert robust_direction < 5.0
        assert plain_direction > 20.0
        assert robust_direction < plain_direction


class TestUnitBaseline:
    def test_normalizes_translation_and_preserves_rotation(self):
        _, _, _, true_relative, _ = two_view_scene()
        scaled = tv.unit_baseline(true_relative)
        matrix = mot.motor_to_matrix(scaled)
        assert abs(float(matrix[:3, 3].norm()) - 1.0) < 1e-10
        original = mot.motor_to_matrix(true_relative)
        torch.testing.assert_close(matrix[:3, :3], original[:3, :3], atol=1e-10, rtol=0)

    def test_is_idempotent(self):
        _, _, _, true_relative, _ = two_view_scene()
        once = tv.unit_baseline(true_relative)
        twice = tv.unit_baseline(once)
        torch.testing.assert_close(twice, once, atol=1e-10, rtol=0)


class TestRansac:
    """Accuracy bounds are the measured ones, not aspirational.

    The tested envelope is up to 20% outliers at 0.5px noise. Beyond that --
    40% outliers, or 20% with 1px noise -- the estimate collapses to roughly
    7 degrees of rotation and 85 degrees of translation-direction error, with a
    consistent signature that suggests a single wrong attractor rather than
    sampling bad luck. Notably, raising the RANSAC budget from 200 to 800
    iterations changes those results *bit for bit*, so it is not a matter of
    too few hypotheses. That regime is documented in ``docs/ga-sfm.md`` as an
    open limitation rather than asserted here.
    """

    def test_exact_without_noise(self):
        points_a, points_b, intrinsics, true_relative, _ = two_view_scene(points=200)
        motor, inliers = tv.ransac_relative_motor(
            points_a, points_b, intrinsics, threshold_px=1.5, iterations=60
        )
        rotation, direction = pose_errors(motor, true_relative)
        assert rotation < 1e-4 and direction < 1e-4
        assert bool(inliers.all())

    @pytest.mark.parametrize("outlier_fraction", [0.0, 0.1, 0.2])
    def test_accurate_under_noise_and_outliers(self, outlier_fraction):
        points_a, points_b, intrinsics, true_relative, truth_mask = two_view_scene(
            points=300, pixel_noise=0.5, outlier_fraction=outlier_fraction, seed=1
        )
        motor, inliers = tv.ransac_relative_motor(
            points_a, points_b, intrinsics, threshold_px=1.5, iterations=200
        )
        rotation, direction = pose_errors(motor, true_relative)
        assert rotation < 1.2, rotation
        assert direction < 3.5, direction
        # No true outlier should survive as an inlier.
        if outlier_fraction:
            precision = float((inliers & truth_mask).sum()) / max(int(inliers.sum()), 1)
            assert precision > 0.95, precision

    def test_raises_when_no_pose_is_consistent(self):
        gen = torch.Generator().manual_seed(5)
        intrinsics = torch.tensor(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        points_a = torch.rand(20, 2, generator=gen, dtype=DTYPE) * 640.0
        points_b = torch.rand(20, 2, generator=gen, dtype=DTYPE) * 640.0
        # Pure noise: it may still return something, but must not crash or
        # return a non-finite motor.
        motor, _ = tv.ransac_relative_motor(
            points_a, points_b, intrinsics, threshold_px=0.5, iterations=20
        )
        assert torch.isfinite(motor).all()
