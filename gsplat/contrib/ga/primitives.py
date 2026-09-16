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
"""Incidence: building geometric objects from each other, and measuring between them.

This is where the practical payoff of the algebra shows up. Constructions that
are separate special cases in a vector-algebra pipeline -- a line through two
points, a line where two planes cross, the distance from a point to a line --
are all products here, and every one of them is differentiable and batched.

Residuals are returned as *vectors* rather than norms wherever possible. A norm
has a kink at zero, exactly where a well-fit residual sits, which makes it a
poor least-squares term; the underlying vector is smooth there.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import algebra as _alg

__all__ = [
    "join_points",
    "line_from_point_direction",
    "meet_planes",
    "plane_from_points",
    "normalize_plane",
    "normalize_line",
    "line_direction",
    "line_moment",
    "point_plane_distance",
    "point_line_residual",
    "point_line_distance",
    "closest_point_on_line",
    "line_plane_residual",
    "line_plane_distance",
    "incidence_residual",
    "mv_join_point_line",
]

_EPS = 1e-12


def _coeff(mv, name: str, like: torch.Tensor) -> torch.Tensor:
    """One blade coefficient of ``mv``, broadcast to ``like``'s shape.

    kingdon drops structurally-zero blades, so a missing blade is the normal
    case rather than an error.
    """
    value = getattr(mv, name, None)
    if not isinstance(value, torch.Tensor):
        return torch.full_like(like, float(value if value is not None else 0.0))
    return torch.broadcast_to(value, like.shape)


def _like(*tensors: torch.Tensor) -> torch.Tensor:
    out = tensors[0][..., 0]
    for t in tensors[1:]:
        out = out * t[..., 0]
    return out


def join_points(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The line through two Euclidean points ``(..., 3)`` -> line ``(..., 6)``."""
    mv = _alg.point_mv(a) & _alg.point_mv(b)
    return _alg.mv_to_line(mv, like=_like(a, b))


def meet_planes(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The line where two planes ``(..., 4)`` cross -> line ``(..., 6)``.

    The dual construction to :func:`join_points`, and the same line comes out
    either way (asserted in ``tests/ga/test_algebra.py``).
    """
    mv = _alg.plane_mv(p) ^ _alg.plane_mv(q)
    return _alg.mv_to_line(mv, like=_like(p, q))


def plane_from_points(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The plane through three Euclidean points -> plane ``(..., 4)``."""
    mv = _alg.point_mv(a) & _alg.point_mv(b) & _alg.point_mv(c)
    return _alg.mv_to_plane(mv, like=_like(a, b, c))


def line_from_point_direction(point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """The line through ``point`` along ``direction`` (need not be unit)."""
    return join_points(point, point + direction)


def line_direction(line: torch.Tensor) -> torch.Tensor:
    """Direction 3-vector of a line, using the blade-to-axis map of the algebra."""
    return torch.stack([line[..., 5], -line[..., 4], line[..., 3]], dim=-1)


def line_moment(line: torch.Tensor) -> torch.Tensor:
    """Moment 3-vector of a line (the ``e01, e02, e03`` block)."""
    return line[..., 0:3]


def normalize_plane(plane: torch.Tensor) -> torch.Tensor:
    """Scale a plane so its normal is a unit vector.

    Only then is :func:`point_plane_distance` a metric distance rather than a
    value scaled by an arbitrary homogeneous factor.
    """
    norm = torch.linalg.vector_norm(plane[..., :3], dim=-1, keepdim=True)
    return plane / norm.clamp_min(_EPS)


def normalize_line(line: torch.Tensor) -> torch.Tensor:
    """Scale a line so its direction is a unit vector."""
    norm = torch.linalg.vector_norm(line_direction(line), dim=-1, keepdim=True)
    return line / norm.clamp_min(_EPS)


def point_plane_distance(point: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """Signed distance from points ``(..., 3)`` to planes ``(..., 4)``.

    This is the wedge ``plane ^ point``, whose single pseudoscalar coefficient
    *is* the signed distance once the plane is normalized -- not merely zero on
    incidence. The plane is normalized here, so the result is metric.
    """
    plane = normalize_plane(plane)
    mv = _alg.plane_mv(plane) ^ _alg.point_mv(point)
    return _coeff(mv, "e0123", _like(point, plane))


def point_line_residual(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """Vector residual ``(..., 3)`` whose norm is the point-to-line distance.

    The join of a point and a line is the plane containing both; once the line
    is normalized, the magnitude of that plane's normal is exactly the
    point-line distance. Returning the normal itself rather than its length
    keeps the term smooth at zero, which is where a converged residual lives.

    This is the triangulation and bundle-adjustment residual for point features:
    a camera ray is a line, and a reconstructed point should lie on it.
    """
    line = normalize_line(line)
    mv = _alg.point_mv(point) & _alg.line_mv(line)
    plane = _alg.mv_to_plane(mv, like=_like(point, line))
    return plane[..., :3]


def point_line_distance(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """Distance from points ``(..., 3)`` to lines ``(..., 6)``."""
    return torch.linalg.vector_norm(point_line_residual(point, line), dim=-1)


def closest_point_on_line(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of ``point`` onto ``line``."""
    line = normalize_line(line)
    direction = line_direction(line)
    # A point on the line: direction x moment recovers the foot of the
    # perpendicular from the origin for a normalized line.
    origin = torch.linalg.cross(direction, line_moment(line), dim=-1)
    delta = point - origin
    return origin + (delta * direction).sum(-1, keepdim=True) * direction


def line_plane_residual(line: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """Vector residual ``(..., 4)`` measuring how far a line is from lying in a plane.

    The wedge ``plane ^ line`` is a grade-3 object that vanishes exactly when the
    line lies in the plane, and whose magnitude is the geometric offset once both
    operands are normalized. As with :func:`point_line_residual`, the underlying
    coefficients are returned rather than their norm, so the term stays smooth at
    zero.

    Note this is *the same wedge* as :func:`point_plane_distance`, with an
    operand of a different grade. That is the whole argument for doing structure
    from motion in this algebra: point and line features are not two residuals
    to derive and maintain separately, they are one expression evaluated on
    different objects.
    """
    plane = normalize_plane(plane)
    line = normalize_line(line)
    mv = _alg.plane_mv(plane) ^ _alg.line_mv(line)
    like = _like(plane, line)
    return torch.stack(
        [_coeff(mv, name, like) for name in ("e012", "e013", "e023", "e123")], dim=-1
    )


def line_plane_distance(line: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """Offset between a line and a plane; zero exactly when the line lies in it."""
    return torch.linalg.vector_norm(line_plane_residual(line, plane), dim=-1)


def incidence_residual(plane: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
    """Residual of ``plane ^ entity`` for a point ``(..., 3)`` or a line ``(..., 6)``.

    One entry point for both feature types, dispatching only on the trailing
    dimension. Adding plane features later means adding a branch here, not a new
    Jacobian derivation.
    """
    if entity.shape[-1] == 3:
        return point_plane_distance(entity, plane).unsqueeze(-1)
    if entity.shape[-1] == 6:
        return line_plane_residual(entity, plane)
    raise ValueError(
        f"expected a point (..., 3) or a line (..., 6), got shape {tuple(entity.shape)}"
    )


def mv_join_point_line(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """The plane ``(..., 4)`` spanned by a point and a line.

    The same join that :func:`point_line_residual` takes the normal of; exposed
    separately because projecting a 3D line into an image wants the whole plane,
    not just its distance interpretation.
    """
    mv = _alg.point_mv(point) & _alg.line_mv(line)
    return _alg.mv_to_plane(mv, like=_like(point, line))
