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
"""Motor-parameterized camera pose refinement -- claim C2.

A drop-in alternative to ``examples/utils.py::CameraOptModule``, matching its
constructor, ``zero_init`` / ``random_init`` and ``forward`` signature so it can
be swapped into ``examples/simple_trainer.py`` under ``--pose_opt``.

The difference is the parameterization of the per-camera correction:

============ ============== =====================================================
              parameters     per forward pass
============ ============== =====================================================
6D + delta    9 for 6 dof    Gram-Schmidt orthogonalization; identity is the
                             non-zero constant ``[1,0,0,0,1,0]``, so it needs a
                             buffer and ``zero_init`` is only identity by
                             construction around it
motor         6 for 6 dof    one ``exp`` from the tangent algebra; identity *is*
                             the zero vector, so ``zero_init`` is exactly
                             identity with nothing to remember
============ ============== =====================================================

The motor parameterization is *minimal*: six numbers for six degrees of freedom,
with no constraint to maintain and no redundant directions in the gradient. That
is a real structural difference and it is what claim C2 asserts.

What it does **not** assert is that this trains better. Both parameterizations
cover SE(3) and both are differentiable; whether minimality helps a particular
optimizer on a particular scene is an empirical question that needs the real
trainer on a GPU. ``tests/ga/test_camera_opt.py`` runs the CPU-sized version of
that comparison; the end-to-end run is documented in ``docs/ga-sfm.md`` and has
not been executed.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import motor as _mot

__all__ = ["MotorCameraOptModule"]


class MotorCameraOptModule(torch.nn.Module):
    """Per-camera pose correction parameterized by a PGA bivector.

    Args:
        n: number of cameras to hold corrections for.

    The correction is applied on the right, exactly as ``CameraOptModule`` does
    (``camtoworlds @ transform``), so it is a perturbation in each camera's own
    frame and the two modules are interchangeable.
    """

    def __init__(self, n: int):
        super().__init__()
        # Six numbers per camera: the bivector [wx, wy, wz, vx, vy, vz].
        # No identity buffer -- the identity motor is exp(0).
        self.embeds = torch.nn.Embedding(n, 6)
        self.zero_init()

    def zero_init(self) -> None:
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float) -> None:
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera poses by the learned corrections.

        Args:
            camtoworlds: ``(..., 4, 4)``
            embed_ids: ``(...,)``

        Returns:
            updated ``camtoworlds`` of the same shape.
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        bivectors = self.embeds(embed_ids)  # (..., 6)
        transform = _mot.motor_to_matrix(_mot.motor_exp(bivectors))
        return torch.matmul(camtoworlds, transform.to(camtoworlds.dtype))
