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
"""PGA encoding conventions and cross-library conformance.

The conventions asserted here are the contract documented in
:mod:`gsplat.contrib.ga.algebra`; the rest of the GA code is written against
them, so a convention drifting silently would be the worst kind of bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat.contrib.ga import algebra as alg

torch.manual_seed(0)
DTYPE = torch.float64


def _mv_scalar(mv, name: str) -> float:
    value = getattr(mv, name, 0.0)
    return float(value)


class TestConventions:
    """The encoding contract: which grade holds what, and what incidence means."""

    def test_point_is_grade_three_and_round_trips(self):
        pts = torch.randn(17, 3, dtype=DTYPE)
        mv = alg.point_mv(pts)
        assert {bin(k).count("1") for k in mv.keys()} == {3}
        back = alg.mv_to_point(mv, like=pts[..., 0])
        torch.testing.assert_close(back, pts)

    def test_plane_is_grade_one_and_round_trips(self):
        planes = torch.randn(11, 4, dtype=DTYPE)
        mv = alg.plane_mv(planes)
        assert {bin(k).count("1") for k in mv.keys()} == {1}
        torch.testing.assert_close(alg.mv_to_plane(mv, like=planes[..., 0]), planes)

    def test_line_is_grade_two_and_round_trips(self):
        lines = torch.randn(7, 6, dtype=DTYPE)
        mv = alg.line_mv(lines)
        assert {bin(k).count("1") for k in mv.keys()} == {2}
        torch.testing.assert_close(alg.mv_to_line(mv, like=lines[..., 0]), lines)

    def test_bivector_round_trips(self):
        biv = torch.randn(5, 6, dtype=DTYPE)
        mv = alg.bivector_mv(biv)
        torch.testing.assert_close(alg.mv_to_bivector(mv, like=biv[..., 0]), biv)

    def test_plane_wedge_point_is_signed_distance(self):
        """For a normalized plane and point, ``plane ^ point`` is the signed distance.

        This is the residual the bundle adjuster uses for plane features, so it
        is worth pinning to an explicit geometric meaning rather than trusting
        that "the wedge vanishes on incidence" is enough.
        """
        # plane z = 3, unit normal
        plane = torch.tensor([[0.0, 0.0, 1.0, -3.0]], dtype=DTYPE)
        for z, expected in [(3.0, 0.0), (9.0, 6.0), (-1.0, -4.0)]:
            pt = torch.tensor([[1.0, 2.0, z]], dtype=DTYPE)
            wedge = alg.plane_mv(plane) ^ alg.point_mv(pt)
            got = float(np.asarray(wedge.e0123).reshape(-1)[0])
            assert got == pytest.approx(expected, abs=1e-12)

    def test_join_of_points_equals_meet_of_planes(self):
        """The z axis, built two ways: joining two points, meeting two planes."""
        a = alg.point_mv(torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE))
        b = alg.point_mv(torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE))
        join = a & b
        x0 = alg.plane_mv(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE))
        y0 = alg.plane_mv(torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=DTYPE))
        meet = x0 ^ y0
        like = torch.zeros(1, dtype=DTYPE)
        torch.testing.assert_close(
            alg.mv_to_line(join, like=like), alg.mv_to_line(meet, like=like)
        )

    def test_rotational_blades_map_to_axes(self):
        """``e23 -> x``, ``-e13 -> y``, ``e12 -> z`` (right-handed)."""
        from gsplat.contrib.ga import motor as mot

        quarter = np.pi / 4  # bivector magnitude for a 90-degree rotation
        probe = torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE)
        expected = {
            0: torch.tensor([1.0, -1.0, 1.0], dtype=DTYPE),  # about x
            1: torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE),  # about y
            2: torch.tensor([-1.0, 1.0, 1.0], dtype=DTYPE),  # about z
        }
        for axis, want in expected.items():
            biv = torch.zeros(6, dtype=DTYPE)
            biv[axis] = -quarter
            got = mot.motor_apply_point(mot.motor_exp(biv), probe)
            torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


class TestCliffordConformance:
    """Cross-check kingdon's products against the independent ``clifford`` library.

    Two implementations of the same algebra agreeing is a much stronger signal
    than either agreeing with itself. The libraries order and name their bases
    differently, so blades are matched by their *set* of basis indices with an
    explicit permutation sign.
    """

    @staticmethod
    def _clifford_layout():
        clifford = pytest.importorskip("clifford")
        layout, blades = clifford.Cl(3, 0, 1)
        # clifford orders the degenerate basis vector *first* (sig == [0,1,1,1]),
        # so its e1 is kingdon's e0 and the index mapping is a shift by one.
        assert list(layout.sig) == [0, 1, 1, 1]
        return layout, blades

    @staticmethod
    def _kingdon_name_to_clifford(name: str) -> tuple[str, float]:
        """Map a kingdon blade name to (clifford blade name, permutation sign)."""
        if name == "e":
            return "", 1.0
        idx = [int(c) for c in name[1:]]
        mapped = [i + 1 for i in idx]
        sign = 1.0
        arr = list(mapped)
        for i in range(len(arr)):  # bubble sort, counting transpositions
            for j in range(len(arr) - 1 - i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    sign = -sign
        return "e" + "".join(str(i) for i in arr), sign

    def _to_clifford(self, mv, blades, layout):
        out = layout.scalar * 0.0
        for key, value in zip(mv.keys(), mv.values()):
            name = alg.ALGEBRA.bin2canon[key]
            cname, sign = self._kingdon_name_to_clifford(name)
            basis = layout.scalar if cname == "" else blades[cname]
            out = out + float(value) * sign * basis
        return out

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_geometric_product_matches_clifford(self, seed):
        layout, blades = self._clifford_layout()
        rng = np.random.default_rng(seed)
        names = list(alg.ALGEBRA.canon2bin)
        a = alg.ALGEBRA.multivector({n: float(rng.normal()) for n in names})
        b = alg.ALGEBRA.multivector({n: float(rng.normal()) for n in names})

        kingdon_product = a * b
        clifford_product = self._to_clifford(a, blades, layout) * self._to_clifford(
            b, blades, layout
        )
        reference = self._to_clifford(kingdon_product, blades, layout)
        assert abs((clifford_product - reference).value).max() < 1e-10
