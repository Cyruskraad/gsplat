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

"""The P0 calibration gates, and the one that had to be rewritten.

The fixture exploits the fact that mirror reflection is its own inverse. Rather
than solving Alhazen's problem to find where a highlight lands, it starts from a
point on the sphere and *derives* a camera that would see its highlight -- which
produces an exactly consistent camera/sphere/light configuration, so the
noiseless case is an equality rather than a tolerance.

The plan declared a gate of "light positions recovered from synthetic sphere
highlights to under 2 mm". Running it showed that is not attainable and not a
matter of tuning: the triangulation baseline from a single ball is its own
diameter, and the sphere's curvature amplifies ray noise by ``standoff /
radius``. Measured worst case over 30 seeds at a half-pixel ray error: **784
mm** with one 40 mm ball, 27 mm with two balls 600 mm apart, 17 mm at 1.2 m.

What survives is *direction*, at about 1 degree, and that is the quantity the
atom basis consumes. The position comes instead from a single camera-frame
offset fitted across every shot, which is the gate below that does hold.
"""

import math

import pytest
import torch

from atlas.functional import (
    closest_point_to_rays,
    fibonacci_sphere,
    light_ray_from_highlight,
    ray_sphere_intersection,
    reflect,
    solve_flash_offset,
    subtract_ambient,
)

DTYPE = torch.float64
SPHERE_CENTER = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)
SPHERE_RADIUS = 0.02  # a 40 mm chrome ball, in metres
TRUE_LIGHT = torch.tensor([0.45, 0.30, 0.80], dtype=DTYPE)


def _consistent_view(surface_point, light_position, standoff=0.9, center=None):
    """A camera whose ray to ``surface_point`` reflects exactly at the light.

    Reflection is symmetric, so reflecting the light direction about the normal
    gives the direction the camera must lie along.
    """
    if center is None:
        center = SPHERE_CENTER
    normal = (surface_point - center) / SPHERE_RADIUS
    to_light = torch.nn.functional.normalize(
        light_position - surface_point, dim=-1, eps=1e-15
    )
    to_camera = reflect(to_light, normal)
    camera = surface_point + standoff * to_camera
    ray = torch.nn.functional.normalize(surface_point - camera, dim=-1, eps=1e-15)
    return camera, ray


def _highlight_rays(
    num_views, light_position=TRUE_LIGHT, jitter=0.0, seed=0, center=SPHERE_CENTER
):
    """Build ``num_views`` consistent observations of the same light."""
    gen = torch.Generator().manual_seed(seed)
    candidates = fibonacci_sphere(256, dtype=DTYPE)
    # Keep points on the lit side of the ball, where a highlight can exist.
    towards_light = torch.nn.functional.normalize(
        light_position - center, dim=-1, eps=1e-15
    )
    facing = candidates @ towards_light > 0.35
    candidates = candidates[facing]
    step = max(1, candidates.shape[0] // num_views)
    origins, directions = [], []
    for k in range(num_views):
        point = center + SPHERE_RADIUS * candidates[k * step]
        camera, ray = _consistent_view(point, light_position, center=center)
        if jitter > 0.0:
            ray = torch.nn.functional.normalize(
                ray + jitter * torch.randn(3, generator=gen, dtype=DTYPE),
                dim=-1,
                eps=1e-15,
            )
        origins.append(camera)
        directions.append(ray)
    return torch.stack(origins), torch.stack(directions)


def _highlight_for_camera(
    camera, light_position, center=SPHERE_CENTER, radius=SPHERE_RADIUS
):
    """Where the highlight of ``light_position`` lands for a camera at ``camera``.

    This is Alhazen's problem and it has no convenient closed form, so the
    fixture solves it numerically: a global sample over the ball, then cone
    refinements that shrink by 4x each round. Accuracy matters -- a sloppy
    highlight looks exactly like ray noise and would quietly become the floor of
    every measurement below -- so the search runs until the reflected ray agrees
    with the true light direction to about 1e-12 in cosine.

    Used only to build observations. The code under test never does this.
    """

    def score(points):
        normals = (points - center) / radius
        to_camera = torch.nn.functional.normalize(camera - points, dim=-1, eps=1e-15)
        to_light = torch.nn.functional.normalize(
            light_position - points, dim=-1, eps=1e-15
        )
        cos = (reflect(to_camera, normals) * to_light).sum(dim=-1)
        visible = (normals * to_camera).sum(dim=-1) > 0.0
        return torch.where(visible, cos, torch.full_like(cos, -2.0))

    directions = fibonacci_sphere(4096, dtype=DTYPE)
    values = score(center + radius * directions)
    best_axis = directions[int(torch.argmax(values))].clone()
    best_cos = float(values.max())

    half_angle = 0.1
    probe = fibonacci_sphere(256, dtype=DTYPE)
    for _ in range(10):
        candidates = torch.nn.functional.normalize(
            best_axis + half_angle * probe, dim=-1, eps=1e-15
        )
        values = score(center + radius * candidates)
        index = int(torch.argmax(values))
        if float(values[index]) > best_cos:
            best_cos = float(values[index])
            best_axis = candidates[index].clone()
        half_angle *= 0.25

    point = center + radius * best_axis
    ray = torch.nn.functional.normalize(point - camera, dim=-1, eps=1e-15)
    return ray, best_cos


def test_reflection_is_its_own_inverse():
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    vector = torch.nn.functional.normalize(
        torch.tensor([0.3, -0.7, 0.6], dtype=DTYPE), dim=-1
    )
    assert (
        torch.max(torch.abs(reflect(reflect(vector, normal), normal) - vector)) < 1e-14
    )


def test_reflection_preserves_length_and_the_angle_to_the_normal():
    normal = torch.nn.functional.normalize(
        torch.tensor([0.2, 0.4, 0.9], dtype=DTYPE), dim=-1
    )
    vector = torch.nn.functional.normalize(
        torch.tensor([-0.5, 0.1, 0.8], dtype=DTYPE), dim=-1
    )
    mirrored = reflect(vector, normal)
    assert abs(float(torch.linalg.norm(mirrored)) - 1.0) < 1e-14
    assert abs(float(normal @ mirrored) - float(normal @ vector)) < 1e-14


def test_ray_sphere_intersection_hits_the_near_surface():
    origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE)
    point, hit = ray_sphere_intersection(
        origin, direction, SPHERE_CENTER, SPHERE_RADIUS
    )
    assert bool(hit[0])
    assert torch.allclose(
        point[0], torch.tensor([0.0, 0.0, SPHERE_RADIUS], dtype=DTYPE), atol=1e-14
    )


def test_ray_sphere_intersection_reports_a_miss():
    origin = torch.tensor([[0.0, 1.0, 1.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE)
    _, hit = ray_sphere_intersection(origin, direction, SPHERE_CENTER, SPHERE_RADIUS)
    assert not bool(hit[0])


def test_a_ray_pointing_away_from_the_sphere_is_a_miss():
    """Both roots behind the origin must not be reported as a hit."""
    origin = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    _, hit = ray_sphere_intersection(origin, direction, SPHERE_CENTER, SPHERE_RADIUS)
    assert not bool(hit[0])


def test_a_single_highlight_gives_a_ray_through_the_light():
    origins, directions = _highlight_rays(1)
    point, direction, hit = light_ray_from_highlight(
        origins, directions, SPHERE_CENTER, SPHERE_RADIUS
    )
    assert bool(hit[0])
    to_light = torch.nn.functional.normalize(TRUE_LIGHT - point[0], dim=-1, eps=1e-15)
    assert float(torch.linalg.norm(direction[0] - to_light)) < 1e-11


@pytest.mark.parametrize("num_views", [2, 3, 8, 24])
def test_noiseless_recovery_is_exact(num_views):
    origins, directions = _highlight_rays(num_views)
    point, direction, hit = light_ray_from_highlight(
        origins, directions, SPHERE_CENTER, SPHERE_RADIUS
    )
    assert bool(hit.all())
    recovered, rms = closest_point_to_rays(point, direction)
    assert float(torch.linalg.norm(recovered - TRUE_LIGHT)) < 1e-9
    assert float(rms) < 1e-9


def test_direction_from_a_single_ball_is_good_to_about_a_degree():
    """What the chrome ball is actually good for, measured.

    Worst case over 30 seeds at a half-pixel ray error is 0.92 degrees; the gate
    below is set a little above that so a real regression fails it.
    """
    worst = 0.0
    for seed in range(30):
        origins, directions = _highlight_rays(16, jitter=1e-4, seed=seed)
        point, direction, hit = light_ray_from_highlight(
            origins, directions, SPHERE_CENTER, SPHERE_RADIUS
        )
        assert bool(hit.all())
        estimated = torch.nn.functional.normalize(direction[hit].mean(0), dim=-1)
        truth = torch.nn.functional.normalize(TRUE_LIGHT - SPHERE_CENTER, dim=-1)
        angle = torch.rad2deg(torch.arccos(torch.clamp(estimated @ truth, -1.0, 1.0)))
        worst = max(worst, float(angle))
    assert worst < 1.5, f"worst direction error {worst:.3f} deg"


def test_position_from_one_small_ball_is_ill_conditioned_and_stays_that_way():
    """The measurement that replaced the plan's 2 mm gate.

    Pinned as a test rather than written down as a caveat, because the failure
    is silent: the solver returns a confident-looking point that is a metre out.
    Anyone tempted to reinstate per-shot triangulation will fail this and read
    why.
    """
    worst = 0.0
    for seed in range(30):
        origins, directions = _highlight_rays(16, jitter=1e-4, seed=seed)
        point, direction, hit = light_ray_from_highlight(
            origins, directions, SPHERE_CENTER, SPHERE_RADIUS
        )
        recovered, _ = closest_point_to_rays(point[hit], direction[hit])
        worst = max(worst, float(torch.linalg.norm(recovered - TRUE_LIGHT)))
    # Hundreds of millimetres, not fractions of one. The band is wide because
    # the point of the test is the order of magnitude, not the digit.
    assert 0.1 < worst < 5.0, f"worst position error {worst * 1000:.1f} mm"


def test_a_wider_baseline_is_what_fixes_position_not_more_views():
    """Two balls far apart beat sixteen views of one ball, by a lot."""

    def worst_error(centers, per_ball, jitter):
        out = 0.0
        for seed in range(15):
            points, dirs = [], []
            for k, center in enumerate(centers):
                o, d = _highlight_rays(
                    per_ball, jitter=jitter, seed=seed + 1000 * k, center=center
                )
                p, dr, hit = light_ray_from_highlight(o, d, center, SPHERE_RADIUS)
                points.append(p[hit])
                dirs.append(dr[hit])
            recovered, _ = closest_point_to_rays(torch.cat(points), torch.cat(dirs))
            out = max(out, float(torch.linalg.norm(recovered - TRUE_LIGHT)))
        return out

    one_ball = worst_error([SPHERE_CENTER], 16, 1e-4)
    two_balls = worst_error(
        [
            torch.tensor([-0.3, 0.0, 0.0], dtype=DTYPE),
            torch.tensor([0.3, 0.0, 0.0], dtype=DTYPE),
        ],
        8,
        1e-4,
    )
    assert two_balls < 0.25 * one_ball


def _bracket_capture(
    true_offset, num_shots=60, noise=1e-4, vary=True, seed=3, radius=SPHERE_RADIUS
):
    """Observations of a bracket-mounted flash over a capture path.

    ``vary`` switches between a perfectly regular orbit -- constant standoff,
    elevation and roll -- and a realistic one that changes all three.
    """
    gen = torch.Generator().manual_seed(seed)
    positions, rotations, ray_pts, ray_dirs = [], [], [], []
    for k in range(num_shots):
        angle = 2.0 * math.pi * k / num_shots
        elevation = 0.35 + (0.45 * math.sin(3.0 * angle) if vary else 0.0)
        standoff = 1.1 + (0.35 * math.cos(2.0 * angle) if vary else 0.0)
        roll = 0.6 * math.sin(5.0 * angle) if vary else 0.0

        camera = torch.tensor(
            [
                standoff * math.cos(angle),
                standoff * math.sin(angle),
                elevation,
            ],
            dtype=DTYPE,
        )
        forward = torch.nn.functional.normalize(SPHERE_CENTER - camera, dim=-1)
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        right = torch.nn.functional.normalize(
            torch.linalg.cross(forward, world_up), dim=-1
        )
        up = torch.linalg.cross(right, forward)
        # Roll about the optical axis, which a handheld capture always has.
        right, up = (
            math.cos(roll) * right + math.sin(roll) * up,
            -math.sin(roll) * right + math.cos(roll) * up,
        )
        rotation = torch.stack([right, up, forward], dim=-1)
        light = camera + rotation @ true_offset

        ray, quality = _highlight_for_camera(camera, light, radius=radius)
        if quality < 1.0 - 1e-10:
            continue
        if noise > 0.0:
            ray = torch.nn.functional.normalize(
                ray + noise * torch.randn(3, generator=gen, dtype=DTYPE), dim=-1
            )
        point, direction, hit = light_ray_from_highlight(
            camera.unsqueeze(0), ray.unsqueeze(0), SPHERE_CENTER, radius
        )
        if not bool(hit[0]):
            continue
        positions.append(camera)
        rotations.append(rotation)
        ray_pts.append(point[0])
        ray_dirs.append(direction[0])

    return (
        torch.stack(positions),
        torch.stack(rotations),
        torch.stack(ray_pts),
        torch.stack(ray_dirs),
    )


TRUE_BRACKET_OFFSET = torch.tensor([0.28, -0.12, 0.05], dtype=DTYPE)


def test_a_regular_orbit_cannot_determine_the_bracket_offset():
    """The trap this solver has to warn about, pinned as a test.

    On a constant-standoff, constant-elevation, zero-roll orbit the sphere is
    stationary in camera coordinates, so every shot contributes the *same*
    rank-2 constraint. The fit then returns a wrong offset with a residual of
    zero -- there is nothing in the data to object to it. Only the condition
    number gives it away.
    """
    solution = solve_flash_offset(
        *_bracket_capture(TRUE_BRACKET_OFFSET, vary=False, noise=0.0)
    )
    assert float(solution.rms) < 1e-6  # the residual is clean, and it lies
    assert float(solution.condition) > 1e6  # only this says not to trust it
    assert float(torch.linalg.norm(solution.offset - TRUE_BRACKET_OFFSET)) > 0.5


BIG_BALL = 0.05  # a 100 mm chrome sphere, in metres


def test_a_varied_capture_recovers_the_bracket_offset():
    """The gate that replaced the plan's 2 mm chrome-sphere figure.

    Varying standoff, elevation and roll -- which a handheld capture does
    anyway -- makes the three unknowns observable. Measured worst case over
    five seeds with a 100 mm sphere at a half-pixel ray error is 18 mm, so the
    gate is 25 mm.
    """
    worst = 0.0
    for seed in (1, 2, 3, 4, 5):
        solution = solve_flash_offset(
            *_bracket_capture(TRUE_BRACKET_OFFSET, seed=seed, radius=BIG_BALL)
        )
        assert float(solution.condition) < 1e4
        worst = max(
            worst, float(torch.linalg.norm(solution.offset - TRUE_BRACKET_OFFSET))
        )
    assert worst < 0.025, f"offset error {worst * 1000:.2f} mm"


def test_the_bracket_fit_is_exact_without_noise():
    solution = solve_flash_offset(
        *_bracket_capture(TRUE_BRACKET_OFFSET, vary=True, noise=0.0, radius=BIG_BALL)
    )
    assert float(torch.linalg.norm(solution.offset - TRUE_BRACKET_OFFSET)) < 1e-6
    # The floor here is the fixture's own numerical Alhazen solve, not the
    # solver: about 1e-7 m of residual is the highlight search, not the fit.
    assert float(solution.rms) < 1e-6


def test_sphere_size_is_the_lever_that_matters_not_the_shot_count():
    """The measurement that decides what to buy, and it is not more storage.

    The sphere's curvature amplifies ray noise by ``standoff / radius``, so the
    error falls roughly as ``1 / radius``: measured worst case over five seeds
    at a half-pixel ray error is 101 mm with a 40 mm ball, 18 mm with a 100 mm
    ball and 7.5 mm with a 200 mm one.

    Shooting more frames does far less. Going from 60 shots to 200 -- more than
    triples the capture -- barely moves the 40 mm ball's number at all, because
    at that radius the error is dominated by the geometry rather than by
    averageable noise. Buy a bigger sphere.
    """

    def worst_error(radius, shots):
        return max(
            float(
                torch.linalg.norm(
                    solve_flash_offset(
                        *_bracket_capture(
                            TRUE_BRACKET_OFFSET,
                            num_shots=shots,
                            seed=seed,
                            radius=radius,
                        )
                    ).offset
                    - TRUE_BRACKET_OFFSET
                )
            )
            for seed in (1, 2, 3)
        )

    small_ball = worst_error(SPHERE_RADIUS, 60)
    big_ball = worst_error(BIG_BALL, 60)
    small_ball_more_shots = worst_error(SPHERE_RADIUS, 200)

    assert big_ball < 0.4 * small_ball
    # Tripling the capture buys less than doubling the sphere's radius does.
    assert small_ball_more_shots > 0.7 * small_ball


def test_more_shots_help_once_the_sphere_is_large_enough():
    """Averaging does work -- but only when the geometry is not the bottleneck."""

    def worst_error(shots):
        return max(
            float(
                torch.linalg.norm(
                    solve_flash_offset(
                        *_bracket_capture(
                            TRUE_BRACKET_OFFSET,
                            num_shots=shots,
                            seed=seed,
                            radius=0.10,
                        )
                    ).offset
                    - TRUE_BRACKET_OFFSET
                )
            )
            for seed in (1, 2, 3)
        )

    assert worst_error(200) < worst_error(60)


def test_residual_reports_inconsistent_rays():
    """The residual is the diagnostic that says the solution is not trustworthy."""
    clean_origins, clean_directions = _highlight_rays(8)
    point, direction, _ = light_ray_from_highlight(
        clean_origins, clean_directions, SPHERE_CENTER, SPHERE_RADIUS
    )
    _, clean_rms = closest_point_to_rays(point, direction)

    corrupted = direction.clone()
    corrupted[0] = torch.nn.functional.normalize(
        corrupted[0] + torch.tensor([0.4, -0.3, 0.1], dtype=DTYPE), dim=-1
    )
    _, dirty_rms = closest_point_to_rays(point, corrupted)
    assert float(dirty_rms) > 100.0 * float(clean_rms) + 1e-6


def test_ambient_subtraction_isolates_the_flash():
    ambient = torch.full((4, 4, 3), 0.2, dtype=DTYPE)
    flash_only = torch.rand(
        4, 4, 3, generator=torch.Generator().manual_seed(1), dtype=DTYPE
    )
    measured = ambient + flash_only
    recovered = subtract_ambient(measured, ambient)
    assert torch.max(torch.abs(recovered - flash_only)) < 1e-14


def test_ambient_subtraction_normalises_differing_exposures():
    ambient = torch.full((2, 2, 3), 0.2, dtype=DTYPE)
    flash_only = torch.full((2, 2, 3), 0.5, dtype=DTYPE)
    # Flash frame shot at 2x exposure, ambient frame at 0.5x.
    measured = 2.0 * (ambient + flash_only)
    recovered = subtract_ambient(
        measured, 0.5 * ambient, flash_exposure=2.0, ambient_exposure=0.5
    )
    assert torch.max(torch.abs(recovered - flash_only)) < 1e-14


def test_ambient_subtraction_can_expose_the_noise_floor():
    """Clamping hides the statistic that says how noisy the capture was."""
    ambient = torch.full((8, 8, 3), 0.3, dtype=DTYPE)
    measured = ambient - 0.01  # flash contributed nothing; this is read noise
    clamped = subtract_ambient(measured, ambient)
    signed = subtract_ambient(measured, ambient, clamp_negative=False)
    assert float(clamped.min()) == 0.0
    assert float(signed.min()) == pytest.approx(-0.01)


# --- guards -----------------------------------------------------------------


def test_reflect_rejects_vectors_that_are_not_three_dimensional():
    with pytest.raises(ValueError, match="must end in 3"):
        reflect(torch.zeros(4, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))


def test_ray_sphere_rejects_a_non_positive_radius():
    with pytest.raises(ValueError, match="radius must be > 0"):
        ray_sphere_intersection(
            torch.zeros(1, 3, dtype=DTYPE),
            torch.zeros(1, 3, dtype=DTYPE),
            SPHERE_CENTER,
            0.0,
        )


def test_ray_sphere_rejects_a_centre_that_is_not_a_three_vector():
    with pytest.raises(ValueError, match=r"center must be \[3\]"):
        ray_sphere_intersection(
            torch.zeros(1, 3, dtype=DTYPE),
            torch.zeros(1, 3, dtype=DTYPE),
            torch.zeros(2, dtype=DTYPE),
            1.0,
        )


def test_closest_point_needs_at_least_two_rays():
    with pytest.raises(ValueError, match="at least two rays"):
        closest_point_to_rays(
            torch.zeros(1, 3, dtype=DTYPE), torch.ones(1, 3, dtype=DTYPE)
        )


def test_closest_point_rejects_origins_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"origins must be \[R, 3\]"):
        closest_point_to_rays(torch.zeros(4, dtype=DTYPE), torch.zeros(4, dtype=DTYPE))


def test_closest_point_rejects_negative_weights():
    with pytest.raises(ValueError, match="weights must be non-negative"):
        closest_point_to_rays(
            torch.zeros(3, 3, dtype=DTYPE),
            torch.ones(3, 3, dtype=DTYPE),
            weights=-torch.ones(3, dtype=DTYPE),
        )


def test_ambient_subtraction_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        subtract_ambient(torch.zeros(2, 2, 3), torch.zeros(2, 3, 3))


def test_ambient_subtraction_rejects_a_non_positive_exposure():
    with pytest.raises(ValueError, match="exposures must be > 0"):
        subtract_ambient(torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), flash_exposure=0.0)
