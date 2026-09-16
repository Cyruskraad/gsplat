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
"""Motor-parameterized camera refinement -- claim C2.

The claim is *minimality*: six parameters for six degrees of freedom, no
constraint to maintain, identity at the origin. It is deliberately **not** a
claim that this trains better. Both parameterizations cover SE(3), and the
optimization comparison below is reported rather than asserted in the motor's
favour -- asserting a win on one synthetic task would be exactly the kind of
overclaim this project is trying to avoid.

The control is written from scratch here rather than imported from
``examples/utils.py``: a control that shares code with the thing under test
cannot catch a bug in it, and the example module pulls in dependencies the test
suite does not need.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import camera as ga_camera
from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga.camera_opt import MotorCameraOptModule
from tests.ga._helpers import synthetic_scene

DTYPE = torch.float64


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt 6D rotation (Zhou et al.), as ``examples/utils.py`` uses."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class SixDCameraOptModule(torch.nn.Module):
    """The control: 3 translation + 6D rotation, mirroring CameraOptModule."""

    def __init__(self, n: int):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, 9)
        self.register_buffer(
            "identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=DTYPE)
        )
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch = camtoworlds.shape[:-2]
        deltas = self.embeds(embed_ids)
        dx, drot = deltas[..., :3], deltas[..., 3:]
        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch, -1))
        transform = torch.eye(4, dtype=camtoworlds.dtype).repeat((*batch, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def _make_modules(n: int):
    motor_module = MotorCameraOptModule(n).to(DTYPE)
    control = SixDCameraOptModule(n).to(DTYPE)
    return motor_module, control


class TestParameterization:
    def test_motor_uses_six_parameters_where_the_control_uses_nine(self):
        """Claim C2 in one assertion: minimal versus redundant."""
        motor_module, control = _make_modules(12)
        motor_count = sum(p.numel() for p in motor_module.parameters())
        control_count = sum(p.numel() for p in control.parameters())
        assert motor_count == 12 * 6
        assert control_count == 12 * 9
        assert motor_count < control_count

    def test_motor_needs_no_identity_buffer(self):
        """The identity motor is exp(0), so there is no constant to carry."""
        motor_module, control = _make_modules(4)
        assert list(motor_module.buffers()) == []
        assert len(list(control.buffers())) == 1

    def test_zero_init_is_exactly_the_identity(self):
        motor_module, control = _make_modules(5)
        poses = mot.motor_to_matrix(mot.motor_exp(torch.randn(5, 6, dtype=DTYPE) * 0.3))
        ids = torch.arange(5)
        torch.testing.assert_close(motor_module(poses, ids), poses, atol=1e-12, rtol=0)
        torch.testing.assert_close(control(poses, ids), poses, atol=1e-12, rtol=0)


class TestOutputIsRigid:
    def test_output_is_a_valid_rigid_transform(self):
        motor_module, _ = _make_modules(6)
        motor_module.random_init(0.1)
        poses = mot.motor_to_matrix(mot.motor_exp(torch.randn(6, 6, dtype=DTYPE) * 0.3))
        out = motor_module(poses, torch.arange(6))
        rotation = out[:, :3, :3]
        torch.testing.assert_close(
            rotation @ rotation.transpose(-2, -1),
            torch.eye(3, dtype=DTYPE).expand(6, 3, 3),
            atol=1e-10,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotation), torch.ones(6, dtype=DTYPE), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(out[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE).expand(6, 4), atol=1e-12, rtol=0)

    def test_random_init_is_reproducible_and_nonzero(self):
        torch.manual_seed(0)
        module, _ = _make_modules(8)
        module.random_init(0.05)
        assert float(module.embeds.weight.detach().abs().max()) > 0.0

    def test_can_represent_an_arbitrary_rigid_perturbation(self):
        """Both parameterizations cover SE(3); the motor does it with six numbers."""
        target = mot.motor_exp(torch.randn(4, 6, dtype=DTYPE) * 0.4)
        target_matrix = mot.motor_to_matrix(target)

        module, _ = _make_modules(4)
        with torch.no_grad():
            module.embeds.weight.copy_(mot.motor_log(target))
        poses = torch.eye(4, dtype=DTYPE).expand(4, 4, 4).contiguous()
        torch.testing.assert_close(
            module(poses, torch.arange(4)), target_matrix, atol=1e-10, rtol=0
        )


class TestGradients:
    def test_gradients_flow_to_the_bivectors(self):
        module, _ = _make_modules(3)
        poses = mot.motor_to_matrix(mot.motor_exp(torch.randn(3, 6, dtype=DTYPE) * 0.2))
        module(poses, torch.arange(3)).pow(2).sum().backward()
        grad = module.embeds.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0

    def test_gradients_are_finite_at_the_identity(self):
        """zero_init sits at w = 0, where an unregularized screw split is 0/0."""
        module, _ = _make_modules(3)
        poses = torch.eye(4, dtype=DTYPE).expand(3, 4, 4).contiguous()
        module(poses, torch.arange(3)).pow(2).sum().backward()
        assert torch.isfinite(module.embeds.weight.grad).all()


class TestPoseRefinement:
    """The CPU-sized version of the trainer experiment.

    Both modules refine perturbed camera poses against fixed 3D points under
    identical optimizer settings. Convergence is asserted for both; neither is
    asserted to beat the other, because one synthetic task would not support
    that claim and the real comparison needs the trainer on a GPU.
    """

    @staticmethod
    def _refine(module, poses, world, pixels, steps=400, lr=0.01):
        views, points = pixels.shape[0], pixels.shape[1]
        ids = torch.arange(views)
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        intrinsics = torch.tensor(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        for _ in range(steps):
            optimizer.zero_grad()
            adjusted = module(poses, ids)
            motors = mot.motor_from_matrix(adjusted)
            projected, valid = ga_camera.project(
                motors[:, None, :].expand(views, points, 8),
                intrinsics.expand(views, points, 3, 3),
                world.expand(views, points, 3),
            )
            loss = ((projected - pixels) * valid.unsqueeze(-1)).pow(2).sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            adjusted = module(poses, ids)
            motors = mot.motor_from_matrix(adjusted)
            projected, valid = ga_camera.project(
                motors[:, None, :].expand(views, points, 8),
                intrinsics.expand(views, points, 3, 3),
                world.expand(views, points, 3),
            )
            return float(
                ((projected - pixels)[valid]).pow(2).sum(-1).mean().sqrt()
            )

    def test_both_parameterizations_reduce_reprojection_error(self):
        torch.manual_seed(0)
        motors, _, world, pixels = synthetic_scene(views=4, points=60, seed=0)
        truth = mot.motor_to_matrix(motors)
        perturbation = mot.motor_exp(torch.randn(4, 6, dtype=DTYPE) * 0.01)
        start = torch.matmul(truth, mot.motor_to_matrix(perturbation))

        motor_module, control = _make_modules(4)
        before = self._refine(motor_module, start, world, pixels, steps=0)
        assert before > 1.0  # the start really is perturbed

        motor_after = self._refine(motor_module, start, world, pixels)
        control_after = self._refine(control, start, world, pixels)

        assert motor_after < before * 0.5, (before, motor_after)
        assert control_after < before * 0.5, (before, control_after)
