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
"""Incidence constructions and metric residuals.

Each test states the Euclidean fact the GA construction is supposed to
reproduce, and checks against a direct numpy computation rather than against
another GA expression -- so a shared convention error cannot pass.
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat.contrib.ga import primitives as prim

DTYPE = torch.float64


def _rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestDistances:
    def test_point_line_distance_matches_numpy(self):
        gen = _rng(0)
        origin = torch.randn(64, 3, generator=gen, dtype=DTYPE)
        direction = torch.randn(64, 3, generator=gen, dtype=DTYPE)
        point = torch.randn(64, 3, generator=gen, dtype=DTYPE)
        line = prim.line_from_point_direction(origin, direction)

        unit = (direction / direction.norm(dim=-1, keepdim=True)).numpy()
        want = np.linalg.norm(np.cross((point - origin).numpy(), unit), axis=-1)
        got = prim.point_line_distance(point, line).numpy()
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)

    def test_point_line_residual_norm_is_the_distance(self):
        gen = _rng(1)
        line = prim.line_from_point_direction(
            torch.randn(32, 3, generator=gen, dtype=DTYPE),
            torch.randn(32, 3, generator=gen, dtype=DTYPE),
        )
        point = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        residual = prim.point_line_residual(point, line)
        torch.testing.assert_close(
            torch.linalg.vector_norm(residual, dim=-1),
            prim.point_line_distance(point, line),
            atol=1e-12,
            rtol=0,
        )

    def test_point_line_residual_is_smooth_at_zero(self):
        """The residual is used as a least-squares term, so it must be
        differentiable *on* the line -- which is where a distance norm is not."""
        line = prim.line_from_point_direction(
            torch.zeros(3, dtype=DTYPE), torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        )
        point = torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE, requires_grad=True)
        prim.point_line_residual(point, line).pow(2).sum().backward()
        assert torch.isfinite(point.grad).all()

    def test_point_plane_distance_matches_numpy(self):
        gen = _rng(2)
        plane = torch.randn(64, 4, generator=gen, dtype=DTYPE)
        point = torch.randn(64, 3, generator=gen, dtype=DTYPE)
        normals = plane[:, :3].numpy()
        want = ((point.numpy() * normals).sum(-1) + plane[:, 3].numpy()) / np.linalg.norm(
            normals, axis=-1
        )
        got = prim.point_plane_distance(point, plane).numpy()
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)

    def test_point_plane_distance_is_signed(self):
        plane = torch.tensor([0.0, 0.0, 1.0, -3.0], dtype=DTYPE)
        above = prim.point_plane_distance(torch.tensor([0.0, 0.0, 9.0], dtype=DTYPE), plane)
        below = prim.point_plane_distance(torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE), plane)
        assert float(above) > 0 and float(below) < 0


class TestConstructions:
    def test_plane_from_points_contains_them(self):
        gen = _rng(3)
        a, b, c = (torch.randn(32, 3, generator=gen, dtype=DTYPE) for _ in range(3))
        plane = prim.plane_from_points(a, b, c)
        for vertex in (a, b, c):
            torch.testing.assert_close(
                prim.point_plane_distance(vertex, plane),
                torch.zeros(32, dtype=DTYPE),
                atol=1e-12,
                rtol=0,
            )

    def test_join_of_points_contains_them(self):
        gen = _rng(4)
        a = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        b = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        line = prim.join_points(a, b)
        for vertex in (a, b):
            torch.testing.assert_close(
                prim.point_line_distance(vertex, line),
                torch.zeros(32, dtype=DTYPE),
                atol=1e-12,
                rtol=0,
            )

    def test_line_direction_matches_the_generating_direction(self):
        gen = _rng(5)
        origin = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        direction = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        line = prim.normalize_line(prim.line_from_point_direction(origin, direction))
        unit = direction / direction.norm(dim=-1, keepdim=True)
        recovered = prim.line_direction(line)
        # A line has no intrinsic orientation, so match up to sign.
        agree = torch.minimum(
            (recovered - unit).abs().amax(-1), (recovered + unit).abs().amax(-1)
        )
        assert float(agree.max()) < 1e-12

    def test_closest_point_lies_on_the_line_and_is_closest(self):
        gen = _rng(6)
        line = prim.line_from_point_direction(
            torch.randn(64, 3, generator=gen, dtype=DTYPE),
            torch.randn(64, 3, generator=gen, dtype=DTYPE),
        )
        point = torch.randn(64, 3, generator=gen, dtype=DTYPE)
        foot = prim.closest_point_on_line(point, line)

        torch.testing.assert_close(
            prim.point_line_distance(foot, line),
            torch.zeros(64, dtype=DTYPE),
            atol=1e-10,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(foot - point, dim=-1),
            prim.point_line_distance(point, line),
            atol=1e-10,
            rtol=0,
        )

    def test_normalize_plane_preserves_the_surface(self):
        gen = _rng(7)
        plane = torch.randn(16, 4, generator=gen, dtype=DTYPE)
        scaled = prim.normalize_plane(plane)
        torch.testing.assert_close(
            torch.linalg.vector_norm(scaled[:, :3], dim=-1),
            torch.ones(16, dtype=DTYPE),
            atol=1e-12,
            rtol=0,
        )
        point = torch.randn(16, 3, generator=gen, dtype=DTYPE)
        torch.testing.assert_close(
            prim.point_plane_distance(point, plane),
            prim.point_plane_distance(point, scaled),
            atol=1e-12,
            rtol=0,
        )
