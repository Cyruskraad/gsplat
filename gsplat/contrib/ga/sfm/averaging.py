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
"""View-graph averaging: pairwise relative motors to global poses.

Given noisy relative motors on the edges of a view graph, recover one motor per
camera. The objective is

    minimize  sum over edges  || log( R_obs^-1 . (M_i . M_j^-1) ) ||^2

which is Gauss-Newton in **bivector coordinates**: the residual of an edge is
the logarithm of how far the observed relative motor is from the one the current
global estimate implies, and a logarithm of a motor is a bivector -- an ordinary
6-vector with no constraint attached. Rotation and translation are averaged
jointly in one linear system rather than in the two separate stages a
vector-algebra pipeline uses, because in this algebra they are one object.

That is the last place in this package where the formulation earns something
concrete, and the dividend is the same as for line features: the Jacobians come
from autograd straight through :func:`gsplat.contrib.ga.motor.motor_log` and
friends, so no linearization had to be derived by hand.

Two entry points, because two-view geometry gives you different things:

- :func:`average_motors` when the relative motors carry true scale.
- :func:`average_rotations` when they do not, which is the usual case --
  two-view translation is known only up to scale, so rotations are averaged
  first and translations are a separate problem.

Gauge: the objective is invariant to a global motor applied to every camera, so
the system is rank-deficient by 6 without a constraint. Cameras listed in
``fixed`` are held.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import motor as _mot

__all__ = [
    "relative_motor",
    "edge_residuals",
    "average_motors",
    "average_rotations",
]

_EPS = 1e-12


def relative_motor(motor_i: torch.Tensor, motor_j: torch.Tensor) -> torch.Tensor:
    """The motor carrying frame *j* into frame *i*, given camera-from-world motors.

    With ``X_i = M_i X`` for a world point ``X``, we have
    ``X_i = (M_i M_j^-1) X_j``, so the relative motor is ``M_i . M_j^-1``.
    """
    return _mot.motor_compose(motor_i, _mot.motor_inverse(motor_j))


def edge_residuals(
    motors: torch.Tensor,
    observed: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
) -> torch.Tensor:
    """Per-edge residual bivectors ``(E, 6)``.

    Zero exactly when the global estimate reproduces the observed relative
    motor. Being a logarithm, the residual is already in the tangent algebra and
    needs no further chart.
    """
    implied = relative_motor(motors[edge_i], motors[edge_j])
    discrepancy = _mot.motor_compose(_mot.motor_inverse(observed), implied)
    return _mot.motor_log(discrepancy)


def _solve_gauss_newton(
    residual_fn,
    num_views: int,
    block: int,
    fixed: tuple[int, ...],
    damping: float,
    iterations: int,
    dtype: torch.dtype,
    device,
):
    """Shared Gauss-Newton loop over per-camera increments.

    ``residual_fn(delta) -> (E, block)`` must be differentiable; the Jacobian is
    taken by autograd rather than derived, which is the point.
    """
    from torch.func import jacrev

    delta_shape = (num_views, block)
    mask = torch.zeros(num_views, dtype=torch.bool, device=device)
    for index in fixed:
        mask[index] = True
    gauge = mask.repeat_interleave(block)

    total = num_views * block
    eye = torch.eye(total, dtype=dtype, device=device)
    accumulated = torch.zeros(delta_shape, dtype=dtype, device=device)
    lam = damping

    zeros = torch.zeros(delta_shape, dtype=dtype, device=device)
    cost = residual_fn(accumulated).pow(2).sum()
    history = [float(cost)]

    for _ in range(iterations):
        def at(step: torch.Tensor) -> torch.Tensor:
            return residual_fn(accumulated + step)

        jac = jacrev(at)(zeros).reshape(-1, total)
        residual = at(zeros).reshape(-1)

        improved = False
        for _ in range(10):
            lhs = jac.T @ jac + lam * eye
            rhs = jac.T @ residual
            # Gauge fixing: pin the held cameras by replacing their rows/columns.
            lhs = lhs.clone()
            lhs[gauge, :] = 0.0
            lhs[:, gauge] = 0.0
            lhs[gauge, gauge] = 1.0
            rhs = rhs.clone()
            rhs[gauge] = 0.0
            try:
                step = torch.linalg.solve(lhs, -rhs.unsqueeze(-1)).squeeze(-1)
            except Exception:
                lam *= 10.0
                continue
            if not torch.isfinite(step).all():
                lam *= 10.0
                continue
            candidate = accumulated + step.reshape(delta_shape)
            trial = residual_fn(candidate).pow(2).sum()
            if torch.isfinite(trial) and trial < cost:
                accumulated, cost = candidate, trial
                lam = max(lam * 0.3, 1e-12)
                history.append(float(cost))
                improved = True
                break
            lam *= 10.0
        if not improved:
            break

    return accumulated, {"cost": history, "final_cost": float(cost), "iterations": len(history) - 1}


def average_motors(
    observed: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    num_views: int,
    initial: torch.Tensor | None = None,
    iterations: int = 30,
    fixed: tuple[int, ...] = (0,),
    damping: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Global camera-from-world motors from relative motors that carry true scale.

    Args:
        observed: ``(E, 8)`` relative motors, edge ``e`` carrying frame
            ``edge_j[e]`` into frame ``edge_i[e]``.
        edge_i, edge_j: ``(E,)`` camera indices.
        num_views: number of cameras.
        initial: ``(V, 8)`` starting motors; identity when omitted.

    Returns ``(motors, stats)``.
    """
    dtype, device = observed.dtype, observed.device
    if initial is None:
        initial = _mot.motor_identity(num_views, dtype=dtype, device=device)
    base = initial

    def residual_fn(delta: torch.Tensor) -> torch.Tensor:
        motors = _mot.motor_compose(_mot.motor_exp(delta), base)
        return edge_residuals(motors, observed, edge_i, edge_j)

    delta, stats = _solve_gauss_newton(
        residual_fn, num_views, 6, fixed, damping, iterations, dtype, device
    )
    motors = _mot.motor_normalize(_mot.motor_compose(_mot.motor_exp(delta), base))
    stats["rmse"] = float(
        (torch.as_tensor(stats["final_cost"]) / max(observed.shape[0], 1)).sqrt()
    )
    return motors, stats


def average_rotations(
    observed: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    num_views: int,
    initial: torch.Tensor | None = None,
    iterations: int = 30,
    fixed: tuple[int, ...] = (0,),
    damping: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Global rotations from relative motors whose translation is unreliable.

    The usual case after two-view estimation, where translation is known only up
    to scale. Only the rotational half of each bivector is optimized and only
    the rotational half of each residual is scored, so the arbitrary translation
    in ``observed`` is ignored rather than allowed to corrupt the rotations.

    Returns motors with the averaged rotation and zero translation.
    """
    dtype, device = observed.dtype, observed.device
    if initial is None:
        initial = _mot.motor_identity(num_views, dtype=dtype, device=device)
    base = initial
    # Strip translation from the observations so only rotation is compared.
    rotation_only = _mot.motor_exp(
        torch.cat(
            [_mot.motor_log(observed)[..., :3], torch.zeros_like(observed[..., :3])],
            dim=-1,
        )
    )

    def residual_fn(delta: torch.Tensor) -> torch.Tensor:
        full = torch.cat([delta, torch.zeros_like(delta)], dim=-1)
        motors = _mot.motor_compose(_mot.motor_exp(full), base)
        return edge_residuals(motors, rotation_only, edge_i, edge_j)[..., :3]

    delta, stats = _solve_gauss_newton(
        residual_fn, num_views, 3, fixed, damping, iterations, dtype, device
    )
    full = torch.cat([delta, torch.zeros_like(delta)], dim=-1)
    motors = _mot.motor_normalize(_mot.motor_compose(_mot.motor_exp(full), base))
    stats["rmse"] = float(
        (torch.as_tensor(stats["final_cost"]) / max(observed.shape[0], 1)).sqrt()
    )
    return motors, stats
