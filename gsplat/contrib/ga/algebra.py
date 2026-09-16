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
"""Projective Geometric Algebra (PGA) backend adapter.

This module is the single place that knows *which* geometric algebra is in use
and how geometric objects are encoded in it. Everything above it works with
plain ``torch.Tensor`` in the canonical layouts documented below, so the
algebra backend (currently `kingdon <https://github.com/tBuLi/kingdon>`_) can be
replaced by generated kernels without touching pipeline code.

Algebra
-------
3D PGA, Clifford algebra :math:`Cl(3,0,1)`: basis vectors ``e1, e2, e3`` square
to ``+1`` and the degenerate ``e0`` squares to ``0``.

Conventions (verified numerically in ``tests/ga/test_algebra.py``)
-----------------------------------------------------------------
======== ====== ===========================================================
Object   Grade  Encoding
======== ====== ===========================================================
plane    1      ``a*e1 + b*e2 + c*e3 + d*e0`` for the plane ``ax+by+cz+d=0``
line     2      bivector; the join of two points, or the meet of two planes
point    3      ``e123 - x*e023 + y*e013 - z*e012`` for the point ``(x,y,z)``
motor    even   ``scalar + bivector + pseudoscalar`` (8 coefficients)
======== ====== ===========================================================

The rotational bivector blades map to axes as ``e23 -> x``, ``-e13 -> y``,
``e12 -> z``. Incidence follows from the products: ``plane ^ point`` is the
signed distance (times the pseudoscalar) when both are normalized, ``plane ^
plane`` meets two planes in their common line, and ``point & point`` joins two
points into the line through them.

Canonical tensor layouts
------------------------
- bivector ``(..., 6)``: ``[wx, wy, wz, vx, vy, vz]`` -- rotational part first.
- motor ``(..., 8)``: ``[1, e01, e02, e03, e12, e13, e23, e0123]``.
- point ``(..., 3)``: Euclidean ``[x, y, z]``.
- plane ``(..., 4)``: ``[a, b, c, d]`` for ``ax+by+cz+d=0``.
- line ``(..., 6)``: ``[e01, e02, e03, e12, e13, e23]``.

Intrinsics are deliberately *not* modelled in the algebra: a pinhole projection
is projective, not a versor, so it cannot be a sandwich product. Extrinsics are
motors; intrinsics stay a separate linear map (see :mod:`gsplat.contrib.ga.camera`).
"""

from __future__ import annotations

import torch
from kingdon import Algebra

__all__ = [
    "ALGEBRA",
    "BIVECTOR_BLADES",
    "MOTOR_BLADES",
    "blade",
    "bivector_mv",
    "mv_to_bivector",
    "motor_mv",
    "mv_to_motor",
    "point_mv",
    "mv_to_point",
    "plane_mv",
    "mv_to_plane",
    "line_mv",
    "mv_to_line",
]

#: The PGA algebra Cl(3,0,1). Shared module-level singleton: kingdon caches its
#: symbolically-optimized operator code per algebra instance, so reusing one
#: instance keeps the generated-code cache warm.
ALGEBRA = Algebra(3, 0, 1)

#: Bivector blade names in canonical tensor order ``[wx, wy, wz, vx, vy, vz]``.
#: The rotational entries carry the sign that makes them right-handed about the
#: matching axis (``-e13`` for y); ``_ROT_SIGNS`` applies it.
BIVECTOR_BLADES = ("e23", "e13", "e12", "e01", "e02", "e03")
_ROT_SIGNS = (1.0, -1.0, 1.0)

#: Motor blade names in canonical tensor order.
MOTOR_BLADES = ("e", "e01", "e02", "e03", "e12", "e13", "e23", "e0123")

_POINT_BLADES = ("e023", "e013", "e012")
_POINT_SIGNS = (-1.0, 1.0, -1.0)
_PLANE_BLADES = ("e1", "e2", "e3", "e0")
_LINE_BLADES = ("e01", "e02", "e03", "e12", "e13", "e23")


def blade(name: str):
    """Return the unit basis blade ``name`` (e.g. ``"e12"``) as a multivector."""
    return ALGEBRA.blades[name]


def _coeffs(mv, names: tuple[str, ...], like: torch.Tensor) -> torch.Tensor:
    """Stack the named blade coefficients of ``mv`` into ``(..., len(names))``.

    Blades that are absent from a sparse multivector read back as zeros shaped
    like ``like``; kingdon drops structurally-zero blades, so this is the normal
    path rather than an error case.
    """
    out = []
    for name in names:
        value = getattr(mv, name, None)
        if not isinstance(value, torch.Tensor):
            value = torch.full_like(like, float(value if value is not None else 0.0))
        out.append(value)
    return torch.stack(torch.broadcast_tensors(*out), dim=-1)


def bivector_mv(biv: torch.Tensor):
    """``(..., 6)`` ``[wx, wy, wz, vx, vy, vz]`` -> bivector multivector."""
    parts = {
        name: (sign * biv[..., i] if sign != 1.0 else biv[..., i])
        for i, (name, sign) in enumerate(zip(BIVECTOR_BLADES, _ROT_SIGNS + (1.0, 1.0, 1.0)))
    }
    return ALGEBRA.multivector(parts)


def mv_to_bivector(mv, like: torch.Tensor) -> torch.Tensor:
    """Bivector multivector -> ``(..., 6)`` ``[wx, wy, wz, vx, vy, vz]``."""
    raw = _coeffs(mv, BIVECTOR_BLADES, like)
    signs = torch.tensor(_ROT_SIGNS + (1.0, 1.0, 1.0), dtype=raw.dtype, device=raw.device)
    return raw * signs


def motor_mv(motor: torch.Tensor):
    """``(..., 8)`` -> motor multivector."""
    return ALGEBRA.multivector({n: motor[..., i] for i, n in enumerate(MOTOR_BLADES)})


def mv_to_motor(mv, like: torch.Tensor) -> torch.Tensor:
    """Motor multivector -> ``(..., 8)``."""
    return _coeffs(mv, MOTOR_BLADES, like)


def point_mv(xyz: torch.Tensor):
    """``(..., 3)`` Euclidean point -> normalized grade-3 point multivector."""
    parts = {"e123": torch.ones_like(xyz[..., 0])}
    for i, (name, sign) in enumerate(zip(_POINT_BLADES, _POINT_SIGNS)):
        parts[name] = sign * xyz[..., i]
    return ALGEBRA.multivector(parts)


def mv_to_point(mv, like: torch.Tensor) -> torch.Tensor:
    """Grade-3 point multivector -> ``(..., 3)``, dehomogenized by the ``e123`` weight."""
    raw = _coeffs(mv, _POINT_BLADES, like)
    signs = torch.tensor(_POINT_SIGNS, dtype=raw.dtype, device=raw.device)
    weight = _coeffs(mv, ("e123",), like)
    return raw * signs / weight


def plane_mv(plane: torch.Tensor):
    """``(..., 4)`` ``[a, b, c, d]`` -> grade-1 plane multivector."""
    return ALGEBRA.multivector({n: plane[..., i] for i, n in enumerate(_PLANE_BLADES)})


def mv_to_plane(mv, like: torch.Tensor) -> torch.Tensor:
    """Grade-1 plane multivector -> ``(..., 4)`` ``[a, b, c, d]``."""
    return _coeffs(mv, _PLANE_BLADES, like)


def line_mv(line: torch.Tensor):
    """``(..., 6)`` -> grade-2 line multivector."""
    return ALGEBRA.multivector({n: line[..., i] for i, n in enumerate(_LINE_BLADES)})


def mv_to_line(mv, like: torch.Tensor) -> torch.Tensor:
    """Grade-2 line multivector -> ``(..., 6)``."""
    return _coeffs(mv, _LINE_BLADES, like)
