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
"""Motors: the PGA representation of rigid motion.

A *motor* is an even-grade PGA element that acts on any geometric object by the
sandwich product :math:`X \\mapsto M X \\widetilde{M}`. Motors are isomorphic to
unit dual quaternions, and the PGA bivectors are exactly :math:`\\mathfrak{se}(3)`,
so this module is the geometric-algebra spelling of the usual Lie-group pose
machinery -- with one practical advantage: the sandwich is *the same code* for
points, lines and planes, and the 6-component bivector parameterization carries
no normalization constraint and no quaternion sign ambiguity.

Exponential and logarithm
-------------------------
A general PGA bivector is a *screw*, not a simple bivector, so it cannot be
exponentiated by the usual :math:`\\cos + \\sin` formula directly (and kingdon's
generic ``exp`` rejects it). We use the invariant (screw) decomposition: split
the bivector into

* ``B_s = omega + v_perp`` -- simple, squares to the scalar ``-theta**2``; and
* ``B_p = v_par``          -- null, squares to zero,

which commute, so ``exp(B) = exp(B_s) exp(B_p)`` with
``exp(B_s) = cos(theta) + sinc(theta) * B_s`` and ``exp(B_p) = 1 + B_p``.
This is verified against an independent 4x4 matrix exponential in
``tests/ga/test_motor.py``, including the pure-rotation, pure-translation,
near-zero and near-pi cases.

The logarithm inverts that split. It canonicalizes the motor to the ``s >= 0``
branch first (motors double-cover SE(3)), which puts ``theta`` in
``[0, pi/2]``. The screw-parallel component is then recovered from the
*pseudoscalar* coefficient rather than from ``cos(theta)``: the ``cos(theta)``
route divides by zero at a 180-degree rotation, which is an ordinary pose, not
an edge case.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import algebra as _alg

__all__ = [
    "motor_identity",
    "motor_exp",
    "motor_log",
    "motor_compose",
    "motor_inverse",
    "motor_normalize",
    "motor_apply_point",
    "motor_apply_plane",
    "motor_apply_line",
]

#: Clamp floor for divisions that are already guarded by a branch.
_EPS = 1e-12

#: Regularizer *squared*, added under the square root that forms ``theta``. It
#: only has to keep ``theta`` and the axis direction differentiable through
#: ``w = 0`` (pure translation), where the unregularized screw split is 0/0.
#: Because it sits inside the root it perturbs an ordinary ``theta`` by only
#: ``_THETA_EPS_SQ / (2 * theta)``, so it is kept far below ``_EPS``: at 1e-12
#: it would put a ~1e-12 error floor on every rotation, large enough to show up
#: in exp/log round trips.
_THETA_EPS_SQ = 1e-24

#: Sign flips of the reverse (grade involution) on the motor blade layout: the
#: six bivector coefficients negate, scalar and pseudoscalar do not.
_REVERSE_SIGNS = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])


def motor_identity(
    *batch: int, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor:
    """Identity motor(s) of shape ``(*batch, 8)``."""
    out = torch.zeros(*batch, 8, dtype=dtype, device=device)
    out[..., 0] = 1.0
    return out


def _screw_split(biv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a bivector into ``(theta, v_perp, v_par)``.

    ``theta`` is regularized so the axis direction stays finite at ``w = 0``;
    there ``v_par -> 0`` and ``v_perp -> v``, which is the correct pure-translation
    limit rather than a special case.
    """
    w, v = biv[..., :3], biv[..., 3:]
    theta = torch.sqrt((w * w).sum(-1, keepdim=True) + _THETA_EPS_SQ)
    axis = w / theta
    v_par = (v * axis).sum(-1, keepdim=True) * axis
    return theta, v - v_par, v_par


def motor_exp(biv: torch.Tensor) -> torch.Tensor:
    """Exponentiate a bivector ``(..., 6)`` into a motor ``(..., 8)``.

    ``biv`` is ``[wx, wy, wz, vx, vy, vz]``. A rigid motion of rotation angle
    ``alpha`` about a unit axis ``n`` through the origin, followed by a
    translation ``t``, has ``w = -alpha/2 * n`` and ``v = -t/2`` (the
    half-angle convention motors share with quaternions).
    """
    theta, v_perp, v_par = _screw_split(biv)
    simple = _alg.bivector_mv(torch.cat([biv[..., :3], v_perp], dim=-1))
    null = _alg.bivector_mv(
        torch.cat([torch.zeros_like(v_par), v_par], dim=-1)
    )
    sinc = torch.sin(theta) / theta
    # exp(B_s) = cos(theta) + sinc(theta) * B_s ;  exp(B_p) = 1 + B_p
    exp_s = torch.cos(theta).squeeze(-1) + sinc.squeeze(-1) * simple
    return _alg.mv_to_motor(exp_s * (1 + null), like=biv[..., 0])


def motor_log(motor: torch.Tensor) -> torch.Tensor:
    """Invert :func:`motor_exp`: motor ``(..., 8)`` -> bivector ``(..., 6)``.

    Returns the principal branch. Motors double-cover SE(3), so ``M`` and
    ``-M`` denote the same rigid motion; the sign is canonicalized to the
    ``scalar >= 0`` hemisphere, giving ``theta in [0, pi/2]``.
    """
    motor = motor_normalize(motor)
    sign = torch.where(motor[..., :1] < 0, -1.0, 1.0)
    motor = motor * sign

    scalar = motor[..., 0:1]
    pseudo = motor[..., 7:8]
    # rotational blades [e12, e13, e23] -> axis order [x, y, z] is [e23, -e13, e12]
    rot = torch.stack([motor[..., 6], -motor[..., 5], motor[..., 4]], dim=-1)
    trans = motor[..., 1:4]

    sin_theta = torch.linalg.vector_norm(rot, dim=-1, keepdim=True)
    theta = torch.atan2(sin_theta, scalar.clamp(min=0.0))

    # theta / sin(theta), continued smoothly to 1 at theta = 0.
    small = sin_theta < 1e-8
    ratio = torch.where(small, torch.ones_like(theta), theta / sin_theta.clamp_min(_EPS))
    w = rot * ratio
    axis = torch.where(small, torch.zeros_like(rot), rot / sin_theta.clamp_min(_EPS))

    # Perpendicular part: trans_perp = sinc(theta) * v_perp.
    trans_par_mag = (trans * axis).sum(-1, keepdim=True)
    trans_perp = trans - trans_par_mag * axis
    v_perp = trans_perp * ratio

    # Parallel part from the pseudoscalar: the pseudoscalar coefficient of a
    # screw is exactly ``pitch * sin(theta)`` (verified in
    # ``test_motor.py::test_pitch_from_pseudoscalar``). Reading the pitch here
    # rather than from ``cos(theta)`` keeps a 180-degree rotation -- an ordinary
    # pose, where ``cos(theta)`` vanishes -- from dividing by zero.
    pitch = pseudo / sin_theta.clamp_min(_EPS)
    v_par = pitch * axis

    # With no rotation there is no screw axis: the whole translation is the
    # null part, and splitting it perp/par would double-count it.
    v = torch.where(small, trans, v_perp + v_par)
    return torch.cat([w, v], dim=-1)


def motor_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose motors: the result applies ``b`` first, then ``a``."""
    mv = _alg.motor_mv(a) * _alg.motor_mv(b)
    return _alg.mv_to_motor(mv, like=a[..., 0] * b[..., 0])


def motor_inverse(motor: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit motor -- its reverse, negating the bivector grade."""
    return motor_normalize(motor) * _REVERSE_SIGNS.to(
        dtype=motor.dtype, device=motor.device
    )


def motor_normalize(motor: torch.Tensor) -> torch.Tensor:
    """Project a motor back onto the unit (rigid-motion) manifold.

    A motor is a rigid motion exactly when ``M ~M == 1``. For any even element
    ``M ~M`` is ``a + c * e0123`` -- a scalar plus a pseudoscalar, because the
    pseudoscalar is nilpotent and central in the even subalgebra. So rescaling
    by ``sqrt(a)`` fixes the rotor norm, and right-multiplying by
    ``1 - (c/2) * e0123`` clears the residual dual part:
    ``(1 + c I)(1 - (c/2) I)**2 = 1`` since ``I**2 = 0``.

    Deriving the correction from the product this way avoids hand-maintaining
    the Study-condition sign table, whose blade pairing (``e01`` with ``e23``,
    not with itself) is easy to get wrong.
    """
    rev = motor * _REVERSE_SIGNS.to(dtype=motor.dtype, device=motor.device)
    mm = motor_compose(motor, rev)
    scale = torch.sqrt(mm[..., 0:1].clamp_min(_EPS))
    motor = motor / scale
    rev = motor * _REVERSE_SIGNS.to(dtype=motor.dtype, device=motor.device)
    dual = motor_compose(motor, rev)[..., 7:8]
    corrector = torch.zeros_like(motor)
    corrector[..., 0] = 1.0
    corrector = corrector - torch.cat(
        [torch.zeros_like(motor[..., :7]), dual / 2], dim=-1
    )
    return motor_compose(motor, corrector)


def _sandwich(motor: torch.Tensor, mv, like: torch.Tensor):
    return _alg.motor_mv(motor).sw(mv)


def motor_apply_point(motor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform Euclidean points ``(..., 3)`` by a motor."""
    like = points[..., 0] * motor[..., 0]
    return _alg.mv_to_point(_sandwich(motor, _alg.point_mv(points), like), like=like)


def motor_apply_plane(motor: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Transform planes ``(..., 4)`` by a motor -- same sandwich, different grade."""
    like = planes[..., 0] * motor[..., 0]
    return _alg.mv_to_plane(_sandwich(motor, _alg.plane_mv(planes), like), like=like)


def motor_apply_line(motor: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
    """Transform lines ``(..., 6)`` by a motor -- same sandwich, different grade."""
    like = lines[..., 0] * motor[..., 0]
    return _alg.mv_to_line(_sandwich(motor, _alg.line_mv(lines), like), like=like)


def motor_to_matrix(motor: torch.Tensor) -> torch.Tensor:
    """Motor ``(..., 8)`` -> homogeneous transform ``(..., 4, 4)``.

    Built by pushing the origin and the three basis points through the motor,
    so it reuses the verified sandwich rather than re-deriving a coefficient
    formula that could disagree with it.
    """
    batch = motor.shape[:-1]
    eye = torch.eye(3, dtype=motor.dtype, device=motor.device)
    origin = torch.zeros(*batch, 3, dtype=motor.dtype, device=motor.device)

    translation = motor_apply_point(motor, origin)
    columns = []
    for axis in range(3):
        basis = eye[axis].expand(*batch, 3)
        columns.append(motor_apply_point(motor, basis) - translation)
    rotation = torch.stack(columns, dim=-1)

    out = torch.zeros(*batch, 4, 4, dtype=motor.dtype, device=motor.device)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1.0
    return out


def motor_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Homogeneous transform ``(..., 4, 4)`` -> motor ``(..., 8)``.

    The rotation is routed through a quaternion (Shepperd's method, picking the
    largest denominator) because extracting an axis directly is ill-conditioned
    near a half-turn. The motor is then assembled as ``translator * rotor``,
    using only the verified :func:`motor_exp` and :func:`motor_compose`.
    """
    rotation, translation = matrix[..., :3, :3], matrix[..., :3, 3]
    m = [[rotation[..., i, j] for j in range(3)] for i in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]

    def branch(w, x, y, z, scale):
        return torch.stack([w, x, y, z], dim=-1) / scale.unsqueeze(-1)

    # Four algebraically equivalent forms; each is stable where its own
    # denominator is largest.
    s0 = torch.sqrt((trace + 1.0).clamp_min(_EPS)) * 2.0
    q0 = branch(0.25 * s0 * s0, m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], s0)
    s1 = torch.sqrt((1.0 + m[0][0] - m[1][1] - m[2][2]).clamp_min(_EPS)) * 2.0
    q1 = branch(m[2][1] - m[1][2], 0.25 * s1 * s1, m[0][1] + m[1][0], m[0][2] + m[2][0], s1)
    s2 = torch.sqrt((1.0 - m[0][0] + m[1][1] - m[2][2]).clamp_min(_EPS)) * 2.0
    q2 = branch(m[0][2] - m[2][0], m[0][1] + m[1][0], 0.25 * s2 * s2, m[1][2] + m[2][1], s2)
    s3 = torch.sqrt((1.0 - m[0][0] - m[1][1] + m[2][2]).clamp_min(_EPS)) * 2.0
    q3 = branch(m[1][0] - m[0][1], m[0][2] + m[2][0], m[1][2] + m[2][1], 0.25 * s3 * s3, s3)

    pick_0 = (trace > 0).unsqueeze(-1)
    pick_1 = ((m[0][0] >= m[1][1]) & (m[0][0] >= m[2][2])).unsqueeze(-1)
    pick_2 = (m[1][1] >= m[2][2]).unsqueeze(-1)
    quat = torch.where(
        pick_0, q0, torch.where(pick_1, q1, torch.where(pick_2, q2, q3))
    )
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(_EPS)

    # Rotor: rotation of angle alpha about unit axis n has bivector -alpha/2 * n.
    vec = quat[..., 1:]
    sin_half = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half, quat[..., 0:1])
    axis = torch.where(sin_half > 1e-12, vec / sin_half.clamp_min(_EPS), torch.zeros_like(vec))
    zeros = torch.zeros_like(translation)
    rotor = motor_exp(torch.cat([-half_angle * axis, zeros], dim=-1))
    translator = motor_exp(torch.cat([zeros, -translation / 2.0], dim=-1))
    return motor_compose(translator, rotor)
