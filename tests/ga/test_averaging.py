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
"""View-graph averaging in the bivector algebra.

Rotation and translation are averaged *jointly* here, in one linear system,
because a motor logarithm is a single 6-vector -- where a vector-algebra
pipeline runs rotation averaging and translation averaging as separate stages
with separate machinery. The Jacobians come from autograd straight through
``motor_log``, so no linearization was derived by hand.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga.sfm import averaging as avg
from tests.ga._helpers import view_graph

DTYPE = torch.float64


def _pose_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((mot.motor_to_matrix(a) - mot.motor_to_matrix(b)).abs().max())


class TestResidual:
    def test_vanishes_at_ground_truth(self):
        truth, observed, edge_i, edge_j = view_graph()
        residual = avg.edge_residuals(truth, observed, edge_i, edge_j)
        assert float(residual.abs().max()) < 1e-12

    def test_relative_motor_round_trips(self):
        truth, observed, edge_i, edge_j = view_graph()
        rebuilt = avg.relative_motor(truth[edge_i], truth[edge_j])
        torch.testing.assert_close(rebuilt, observed, atol=1e-12, rtol=0)

    def test_residual_is_nonzero_for_a_wrong_estimate(self):
        truth, observed, edge_i, edge_j = view_graph()
        gen = torch.Generator().manual_seed(1)
        wrong = mot.motor_compose(
            mot.motor_exp(torch.randn(truth.shape[0], 6, generator=gen, dtype=DTYPE) * 0.2),
            truth,
        )
        residual = avg.edge_residuals(wrong, observed, edge_i, edge_j)
        assert float(residual.abs().max()) > 1e-3


class TestMotorAveraging:
    def test_recovers_global_poses_exactly(self):
        truth, observed, edge_i, edge_j = view_graph(views=10)
        estimate, stats = avg.average_motors(
            observed, edge_i, edge_j, truth.shape[0], iterations=40
        )
        assert stats["rmse"] < 1e-12
        assert _pose_gap(estimate, truth) < 1e-12

    def test_fixed_camera_does_not_move(self):
        truth, observed, edge_i, edge_j = view_graph()
        estimate, _ = avg.average_motors(
            observed, edge_i, edge_j, truth.shape[0], iterations=20, fixed=(0,)
        )
        torch.testing.assert_close(
            estimate[0], mot.motor_identity(dtype=DTYPE), atol=1e-12, rtol=0
        )

    def test_degrades_gracefully_with_edge_noise(self):
        truth, observed, edge_i, edge_j = view_graph()
        gen = torch.Generator().manual_seed(2)
        previous = 0.0
        for sigma in (0.0, 0.01, 0.05):
            noisy = mot.motor_compose(
                mot.motor_exp(
                    torch.randn(observed.shape[0], 6, generator=gen, dtype=DTYPE) * sigma
                ),
                observed,
            )
            estimate, _ = avg.average_motors(
                noisy, edge_i, edge_j, truth.shape[0], iterations=40
            )
            gap = _pose_gap(estimate, truth)
            assert gap >= previous - 1e-9  # error grows with noise, not erratically
            assert gap < 20.0 * sigma + 1e-9
            previous = gap

    def test_cost_is_monotone(self):
        truth, observed, edge_i, edge_j = view_graph()
        gen = torch.Generator().manual_seed(3)
        noisy = mot.motor_compose(
            mot.motor_exp(
                torch.randn(observed.shape[0], 6, generator=gen, dtype=DTYPE) * 0.03
            ),
            observed,
        )
        _, stats = avg.average_motors(noisy, edge_i, edge_j, truth.shape[0], iterations=30)
        costs = stats["cost"]
        assert all(b <= a for a, b in zip(costs, costs[1:]))

    def test_a_disconnected_graph_leaves_its_far_component_free(self):
        """Averaging determines poses only within the fixed camera's component.

        Worth pinning because the failure is silent: the residual goes to
        machine zero -- every edge is satisfied -- while the component that does
        not contain the pinned camera sits at an arbitrary global offset. During
        development a same-parity edge rule split the graph exactly this way and
        looked like a solver bug.
        """
        truth, observed, edge_i, edge_j = view_graph(views=10, connected=False)
        estimate, stats = avg.average_motors(
            observed, edge_i, edge_j, truth.shape[0], iterations=40
        )
        assert stats["rmse"] < 1e-12  # every edge is satisfied
        assert _pose_gap(estimate, truth) > 0.1  # yet the poses are not the truth

        # The component containing camera 0 (even indices) *is* recovered.
        even = torch.arange(0, truth.shape[0], 2)
        assert _pose_gap(estimate[even], truth[even]) < 1e-9


class TestRotationAveraging:
    def test_ignores_unreliable_translations(self):
        """The reason this entry point exists.

        Two-view geometry gives translation only up to scale, so the translation
        in a relative motor is not trustworthy. Scrambling it completely must
        leave the recovered rotations untouched.
        """
        truth, observed, edge_i, edge_j = view_graph()
        gen = torch.Generator().manual_seed(4)
        scrambled = mot.motor_log(observed).clone()
        scrambled[:, 3:] = torch.randn(observed.shape[0], 3, generator=gen, dtype=DTYPE) * 3.0

        estimate, stats = avg.average_rotations(
            mot.motor_exp(scrambled), edge_i, edge_j, truth.shape[0], iterations=40
        )
        got = mot.motor_to_matrix(estimate)[:, :3, :3]
        want = mot.motor_to_matrix(truth)[:, :3, :3]
        assert stats["rmse"] < 1e-12
        assert float((got - want).abs().max()) < 1e-9

    def test_returns_pure_rotations(self):
        truth, observed, edge_i, edge_j = view_graph()
        estimate, _ = avg.average_rotations(
            observed, edge_i, edge_j, truth.shape[0], iterations=20
        )
        translation = mot.motor_to_matrix(estimate)[:, :3, 3]
        assert float(translation.abs().max()) < 1e-9

    def test_matches_motor_averaging_on_rotation(self):
        """With trustworthy translations both routes must agree on rotation."""
        truth, observed, edge_i, edge_j = view_graph()
        joint, _ = avg.average_motors(observed, edge_i, edge_j, truth.shape[0], iterations=40)
        rotation_only, _ = avg.average_rotations(
            observed, edge_i, edge_j, truth.shape[0], iterations=40
        )
        torch.testing.assert_close(
            mot.motor_to_matrix(joint)[:, :3, :3],
            mot.motor_to_matrix(rotation_only)[:, :3, :3],
            atol=1e-9,
            rtol=0,
        )
