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
"""Shared helpers for the geometric-algebra tests.

These tests are CPU-only by design: the GA layer is plain PyTorch, so it must be
verifiable without a GPU (unlike ``gsplat.geometry``, whose operators are
CUDA-only and whose suites skip without a device).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["expm", "se3_matrix", "skew", "random_bivectors", "MAX_THETA"]

#: Motors double-cover SE(3); ``motor_log`` returns the ``scalar >= 0`` branch,
#: which caps the recoverable rotation magnitude at ``pi/2`` in bivector terms
#: (a half-turn of the underlying rotation).
MAX_THETA = np.pi / 2


def skew(w: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix of a 3-vector."""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=float
    )


def expm(a: np.ndarray, terms: int = 60) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring plus a Taylor series.

    Written out rather than taken from SciPy so the oracle stays independent of
    the optional ``scipy`` extra, and so the test does not lean on a library
    that could share a bug with the code under test.
    """
    scale = max(0, int(np.ceil(np.log2(max(np.abs(a).max(), 1e-12)))) + 4)
    small = a / 2**scale
    out = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for k in range(1, terms):
        term = term @ small / k
        out = out + term
    for _ in range(scale):
        out = out @ out
    return out


def se3_matrix(biv: torch.Tensor) -> np.ndarray:
    """The 4x4 rigid transform that a PGA bivector's motor should reproduce.

    The motor convention carries the quaternion half-angle factor, so the
    bivector ``[w, v]`` corresponds to the ``se(3)`` twist ``(-2w, -2v)``.
    """
    b = biv.detach().cpu().numpy().astype(float)
    a = np.zeros((4, 4))
    a[:3, :3] = skew(-2.0 * b[:3])
    a[:3, 3] = -2.0 * b[3:]
    return expm(a)


def random_bivectors(n: int, generator: torch.Generator, trans_scale: float = 2.0):
    """Random bivectors inside the principal branch of ``motor_log``."""
    w = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    norms = w.norm(dim=-1, keepdim=True)
    capped = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (MAX_THETA * 0.999)
    w = w / norms * capped
    v = torch.randn(n, 3, generator=generator, dtype=torch.float64) * trans_scale
    return torch.cat([w, v], dim=-1)
