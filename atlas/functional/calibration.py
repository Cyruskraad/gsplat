# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Recovering where the flash was, and removing the light that was already there.

Handheld OLAT capture has two calibration problems that a light stage does not.
The flash moves, so its position is unknown per shot; and the room is not dark,
so every frame carries ambient illumination that is not the light being
measured.

Both are solved by the capture protocol rather than by inference:

**Where the flash was.** A chrome sphere of known radius and position turns
every frame into a measurement of the light's *direction*. The camera ray that
lands on the sphere's specular highlight reflects, by the law of reflection,
straight at the light, so :func:`light_ray_from_highlight` returns a ray the
light lies on. Turning a bundle of such rays into a position is a separate
question, and the answer is not the obvious one -- see the caveat below.

**What the room contributed.** Shoot every frame twice, once with the flash and
once without, and subtract. The difference is the response to the flash alone,
which is precisely the OLAT observation. It costs a second exposure and it
removes an entire class of error that no amount of modelling would.

The no-flash half has a second use that is worth more than the subtraction:
because it is lit identically in every frame, it is what structure-from-motion
should run on. Changing illumination is the standard reason flash photogrammetry
fails to register, and shooting pairs removes the reason for free.

A measured caveat, and it changed the capture protocol
------------------------------------------------------

Triangulating a *position* from one small chrome ball does not work, and the
reason is geometric rather than numerical. A ray-direction error ``delta`` at
the camera displaces the hit point by ``standoff * delta``, which rotates the
sphere normal by ``standoff * delta / radius`` and the reflected ray by twice
that. The triangulation baseline, meanwhile, is only the ball's *diameter*. So

    position error  ~  (range / baseline) * range * 2 * standoff * delta / radius

For a 40 mm ball at 0.9 m with a half-pixel ray error on a 4000 px sensor, that
is a metre. Measured: 784 mm worst case over 30 seeds. Widening the baseline to
two balls 600 mm apart brings it to 27 mm, and 1.2 m apart to 17 mm. It never
approaches the sub-millimetre figure the design originally assumed.

Direction, by contrast, survives: about 1 degree at the same noise, which is the
quantity the atom basis actually consumes.

So the protocol mounts the flash on a bracket rigidly attached to the camera and
fits **one** camera-frame offset across every shot
(:func:`solve_flash_offset`), instead of triangulating a fresh position per
shot. That replaces a badly conditioned three-unknowns-per-shot problem with a
well-conditioned three-unknowns-in-total one, and the per-shot light position
then inherits the accuracy of the camera pose, which structure-from-motion
already gives.

:func:`closest_point_to_rays` is kept because it is the right tool when the
baseline really is wide, and because its residual is what says whether it was.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "FlashOffset",
    "reflect",
    "ray_sphere_intersection",
    "light_ray_from_highlight",
    "closest_point_to_rays",
    "solve_flash_offset",
    "subtract_ambient",
]


def reflect(incident: Tensor, normal: Tensor) -> Tensor:
    """Mirror ``incident`` about ``normal``: ``2 (n . v) n - v``.

    Both arguments point *away* from the surface, which is the convention that
    makes reflection its own inverse -- feeding the result back in returns the
    input. That symmetry is what lets a test construct an exactly consistent
    camera/sphere/light configuration without solving Alhazen's problem.

    Args:
        incident: ``[..., 3]`` unit vector from the surface towards the viewer.
        normal: ``[..., 3]`` unit surface normal.

    Returns:
        ``[..., 3]`` unit vector from the surface towards the mirror direction.
    """
    if incident.shape[-1] != 3 or normal.shape[-1] != 3:
        raise ValueError(
            f"incident and normal must end in 3, got {tuple(incident.shape)} "
            f"and {tuple(normal.shape)}"
        )
    dot = (normal * incident).sum(dim=-1, keepdim=True)
    return 2.0 * dot * normal - incident


def ray_sphere_intersection(
    origin: Tensor,
    direction: Tensor,
    center: Tensor,
    radius: float,
) -> Tuple[Tensor, Tensor]:
    """Nearest intersection of a ray with a sphere.

    Args:
        origin: ``[..., 3]`` ray origins.
        direction: ``[..., 3]`` unit ray directions.
        center: ``[3]`` sphere centre.
        radius: Sphere radius, positive.

    Returns:
        ``(points, hit)``. ``points`` is ``[..., 3]`` and is only meaningful
        where ``hit`` is True; misses are returned as the ray origin so that
        downstream shapes stay uniform, and the caller is expected to consult
        ``hit`` rather than to trust the point.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if center.shape != (3,):
        raise ValueError(f"center must be [3], got {tuple(center.shape)}")
    offset = origin - center.to(origin.dtype).to(origin.device)
    b = (offset * direction).sum(dim=-1)
    c = (offset * offset).sum(dim=-1) - radius * radius
    discriminant = b * b - c
    hit = discriminant >= 0.0
    root = torch.sqrt(torch.clamp(discriminant, min=0.0))
    # Near root first; fall back to the far one if the near one is behind.
    t_near = -b - root
    t_far = -b + root
    t = torch.where(t_near > 0.0, t_near, t_far)
    hit = hit & (t > 0.0)
    points = origin + t.unsqueeze(-1) * direction
    return torch.where(hit.unsqueeze(-1), points, origin), hit


def light_ray_from_highlight(
    camera_position: Tensor,
    ray_direction: Tensor,
    sphere_center: Tensor,
    sphere_radius: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Turn a chrome-sphere highlight into a ray that points at the light.

    Args:
        camera_position: ``[..., 3]`` camera centres.
        ray_direction: ``[..., 3]`` unit directions from the camera through the
            highlight pixel, in world space.
        sphere_center: ``[3]``.
        sphere_radius: Positive.

    Returns:
        ``(point, direction, hit)``. The light lies along
        ``point + t * direction`` for some ``t > 0``, wherever ``hit``.
    """
    point, hit = ray_sphere_intersection(
        camera_position, ray_direction, sphere_center, sphere_radius
    )
    normal = (point - sphere_center.to(point.dtype).to(point.device)) / sphere_radius
    towards_camera = -ray_direction
    direction = reflect(towards_camera, normal)
    return point, direction, hit


def closest_point_to_rays(
    origins: Tensor,
    directions: Tensor,
    *,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Least-squares point of closest approach to a bundle of rays.

    Minimises the sum of squared perpendicular distances, which has the closed
    form ``(sum_i (I - d_i d_i^T))^-1 sum_i (I - d_i d_i^T) o_i``.

    Two rays suffice in principle. In practice use more: the conditioning
    degrades badly as the rays become parallel, and the returned residual is
    what says whether they did.

    Args:
        origins: ``[R, 3]`` ray origins, ``R >= 2``.
        directions: ``[R, 3]`` unit ray directions.
        weights: ``[R]`` optional non-negative weights.

    Returns:
        ``(point [3], rms_distance)``. The residual is the root-mean-square
        perpendicular distance from the solution to the rays, in world units --
        the number the sub-millimetre gate is stated against.
    """
    if origins.ndim != 2 or origins.shape[-1] != 3:
        raise ValueError(f"origins must be [R, 3], got {tuple(origins.shape)}")
    if directions.shape != origins.shape:
        raise ValueError(
            f"directions must match origins {tuple(origins.shape)}, got "
            f"{tuple(directions.shape)}"
        )
    if origins.shape[0] < 2:
        raise ValueError("at least two rays are needed to locate a point")

    unit = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(
        min=1e-12
    )
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)
    projectors = eye - unit.unsqueeze(-1) * unit.unsqueeze(-2)  # [R, 3, 3]
    if weights is not None:
        if weights.shape != origins.shape[:1]:
            raise ValueError(f"weights must be [R], got {tuple(weights.shape)}")
        if bool((weights < 0).any()):
            raise ValueError("weights must be non-negative")
        projectors = projectors * weights[:, None, None]
    lhs = projectors.sum(dim=0)
    rhs = torch.einsum("rij,rj->i", projectors, origins)
    point = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)

    offset = point.unsqueeze(0) - origins
    perpendicular = offset - (offset * unit).sum(dim=-1, keepdim=True) * unit
    rms = torch.sqrt((perpendicular**2).sum(dim=-1).mean())
    return point, rms


class FlashOffset(NamedTuple):
    """A fitted bracket offset, with the two numbers that say whether to trust it."""

    offset: Tensor  # [3], in camera coordinates
    rms: Tensor  # scalar, perpendicular distance from fitted lights to their rays
    condition: Tensor  # scalar, condition number of the normal equations


def solve_flash_offset(
    camera_positions: Tensor,
    camera_rotations: Tensor,
    ray_points: Tensor,
    ray_directions: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Fit one camera-frame flash offset across every shot.

    With the flash on a bracket the light sits at a fixed offset ``o`` in camera
    coordinates, so its world position in shot ``i`` is ``c_i + R_i o``. Each
    highlight says that point lies on a ray, which is *linear* in ``o``:

        (I - l_i l_i^T) (c_i + R_i o - x_i) = 0

    Stacking those over every shot gives three unknowns constrained by hundreds
    of observations, instead of three unknowns per shot constrained by a handful.
    The conditioning improves for a second reason too: the camera rotates
    substantially around the subject over a capture, so the same offset is
    observed from many orientations, where a per-shot triangulation only ever
    had the ball's diameter to work with.

    Args:
        camera_positions: ``[N, 3]`` camera centres in world space.
        camera_rotations: ``[N, 3, 3]`` camera-to-world rotations.
        ray_points: ``[N, 3]`` a point on each shot's light ray, from
            :func:`light_ray_from_highlight`.
        ray_directions: ``[N, 3]`` the corresponding ray directions.

    A regular orbit does not constrain this, and the failure is silent. If the
    camera always sits at the same distance and orientation relative to the
    sphere, then the sphere is stationary in camera coordinates, every shot
    contributes the *same* rank-2 constraint, and the offset is only determined
    up to a slide along the light ray. The fit then returns a confident answer
    with a **zero** residual that is over a metre wrong -- measured, on a
    perfectly circular 60-shot orbit.

    So vary the capture: change the standoff, the elevation, and the roll. And
    read ``condition`` before believing ``offset``; above roughly ``1e6`` the
    geometry did not determine the answer, whatever the residual says.

    Returns:
        A :class:`FlashOffset`. ``rms`` is the root-mean-square perpendicular
        distance from the fitted light positions to their rays, and
        ``condition`` is the condition number of the normal equations.
    """
    if camera_positions.ndim != 2 or camera_positions.shape[-1] != 3:
        raise ValueError(
            f"camera_positions must be [N, 3], got {tuple(camera_positions.shape)}"
        )
    num = camera_positions.shape[0]
    if camera_rotations.shape != (num, 3, 3):
        raise ValueError(
            f"camera_rotations must be [{num}, 3, 3], got "
            f"{tuple(camera_rotations.shape)}"
        )
    if ray_points.shape != camera_positions.shape:
        raise ValueError(
            f"ray_points must be [{num}, 3], got {tuple(ray_points.shape)}"
        )
    if ray_directions.shape != camera_positions.shape:
        raise ValueError(
            f"ray_directions must be [{num}, 3], got {tuple(ray_directions.shape)}"
        )
    if num < 2:
        raise ValueError("at least two observations are needed to fit an offset")

    unit = ray_directions / torch.linalg.norm(
        ray_directions, dim=-1, keepdim=True
    ).clamp(min=1e-12)
    eye = torch.eye(3, device=unit.device, dtype=unit.dtype)
    # P is idempotent and symmetric, so A^T A reduces to R^T P R.
    projectors = eye - unit.unsqueeze(-1) * unit.unsqueeze(-2)  # [N, 3, 3]
    lhs = torch.einsum(
        "nji,njk,nkl->il", camera_rotations, projectors, camera_rotations
    )
    residual_target = ray_points - camera_positions  # [N, 3]
    rhs = torch.einsum("nji,njk,nk->i", camera_rotations, projectors, residual_target)
    offset = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)

    eigenvalues = torch.linalg.eigvalsh(0.5 * (lhs + lhs.transpose(0, 1)))
    condition = eigenvalues.abs().max() / eigenvalues.abs().min().clamp(min=1e-300)

    predicted = camera_positions + torch.einsum("nij,j->ni", camera_rotations, offset)
    delta = predicted - ray_points
    perpendicular = delta - (delta * unit).sum(dim=-1, keepdim=True) * unit
    rms = torch.sqrt((perpendicular**2).sum(dim=-1).mean())
    return FlashOffset(offset=offset, rms=rms, condition=condition)


def subtract_ambient(
    flash_image: Tensor,
    ambient_image: Tensor,
    *,
    flash_exposure: float = 1.0,
    ambient_exposure: float = 1.0,
    clamp_negative: bool = True,
) -> Tensor:
    """Isolate the flash's contribution from a flash / no-flash pair.

    Both images must be *linear* radiance. Applying this to gamma-encoded data
    is not an approximation, it is simply wrong: the difference of two encoded
    values is not the encoding of the difference.

    Args:
        flash_image: ``[..., C]`` linear radiance with the flash fired.
        ambient_image: ``[..., C]`` linear radiance without it.
        flash_exposure: Relative exposure of the flash frame.
        ambient_exposure: Relative exposure of the ambient frame.
        clamp_negative: Clamp the result at zero. Negative values are sensor
            noise in regions the flash did not reach; they are physically
            meaningless but their *statistics* are the honest estimate of the
            noise floor, so leave this off when measuring it.

    Returns:
        ``[..., C]`` the flash-only observation, normalised to unit exposure.
    """
    if flash_image.shape != ambient_image.shape:
        raise ValueError(
            f"flash and ambient must have the same shape, got "
            f"{tuple(flash_image.shape)} and {tuple(ambient_image.shape)}"
        )
    if flash_exposure <= 0.0 or ambient_exposure <= 0.0:
        raise ValueError(
            f"exposures must be > 0, got {flash_exposure} and {ambient_exposure}"
        )
    difference = flash_image / flash_exposure - ambient_image / ambient_exposure
    return torch.clamp(difference, min=0.0) if clamp_negative else difference
