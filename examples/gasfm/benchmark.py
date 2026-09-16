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
"""Reproduce the geometric-algebra structure-from-motion gates.

Runs the two bundle-adjustment arms on the same synthetic problem and reports
whether the project's two pre-registered gates still hold:

* **Gate 1** -- the motor arm and the quaternion/se(3) control reach the same
  optimum. They are the same estimator in different coordinates, so agreement is
  the passing condition; divergence beyond the gauge means the GA code is wrong.
* **Gate 2** -- the motor arm runs within a fixed multiple of the control.

Exits non-zero if either gate fails, so it doubles as a regression guard rather
than a number that only ever appears in a commit message.

    PYTHONPATH=. python examples/gasfm/benchmark.py
    PYTHONPATH=. python examples/gasfm/benchmark.py --views 12 --points 600

(``PYTHONPATH=.`` is only needed when gsplat is not pip-installed, as when
working from a source checkout.)

CPU-only and self-contained: no dataset, no GPU, no network, and no dependency
on the test suite -- every scene it needs is built below.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from gsplat.contrib.ga import camera as ga_camera
from gsplat.contrib.ga import motor as ga_motor
from gsplat.contrib.ga import primitives as ga_prim
from gsplat.contrib.ga.baseline import ba as qt_ba
from gsplat.contrib.ga.sfm import ba as ga_ba
from gsplat.contrib.ga.sfm import triangulate as ga_tri
from gsplat.contrib.ga.sfm import twoview as ga_tv

DTYPE = torch.float64


# --------------------------------------------------------------------------
# Scene construction (kept here rather than imported from tests so the script
# stands alone for anyone reproducing the numbers).
# --------------------------------------------------------------------------
def make_scene(views: int, points: int, seed: int, distance: float = 8.0):
    gen = torch.Generator().manual_seed(seed)
    intrinsics = (
        torch.tensor(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        .expand(views, 3, 3)
        .contiguous()
    )
    stand_off = torch.zeros(views, 6, dtype=DTYPE)
    stand_off[:, 5] = -distance / 2.0
    motors = ga_motor.motor_compose(
        ga_motor.motor_exp(stand_off),
        ga_motor.motor_exp(torch.randn(views, 6, generator=gen, dtype=DTYPE) * 0.25),
    )
    world = torch.randn(points, 3, generator=gen, dtype=DTYPE) * 0.8
    pixels, _ = ga_camera.project(
        motors[:, None, :].expand(views, points, 8),
        intrinsics[:, None, :, :].expand(views, points, 3, 3),
        world.expand(views, points, 3),
    )
    return motors, intrinsics, world, pixels, gen


def perturb(motors, world, gen, pose_sigma=0.02, point_sigma=0.05):
    views = motors.shape[0]
    delta = torch.randn(views, 6, generator=gen, dtype=DTYPE) * pose_sigma
    delta[0] = 0.0  # camera 0 anchors the gauge
    return (
        ga_motor.motor_compose(ga_motor.motor_exp(delta), motors),
        world + torch.randn(world.shape, generator=gen, dtype=DTYPE) * point_sigma,
    )


# --------------------------------------------------------------------------
def run_bundle_gates(args) -> list[tuple[str, bool, str]]:
    motors, intrinsics, world, pixels, gen = make_scene(args.views, args.points, args.seed)
    start_motors, start_points = perturb(motors, world, gen)

    views, points = args.views, args.points
    camera_idx = torch.arange(views).repeat_interleave(points)
    point_idx = torch.arange(points).repeat(views)
    observations = pixels.reshape(-1, 2)

    ga_problem = ga_ba.BundleProblem(
        start_motors, intrinsics, start_points.clone(), observations, camera_idx, point_idx
    )
    qt_problem = qt_ba.QuaternionBundleProblem(
        qt_ba.poses_from_matrices(ga_motor.motor_to_matrix(start_motors)),
        intrinsics,
        start_points.clone(),
        observations,
        camera_idx,
        point_idx,
    )

    initial = float(
        ga_ba.reprojection_residuals(ga_problem).pow(2).sum(-1).mean().sqrt()
    )

    # Warm up both arms on a throwaway problem before timing. kingdon compiles
    # its operators symbolically on first use, and that cost is once per
    # process, not per iteration -- charging it to whichever arm happens to run
    # first turns a ~2x ratio into a ~60x one. The cold number is reported too,
    # since the compile is real, just amortized.
    warm_motors, warm_intrinsics, warm_world, warm_pixels, warm_gen = make_scene(3, 20, 999)
    warm_cam = torch.arange(3).repeat_interleave(20)
    warm_pt = torch.arange(20).repeat(3)
    warm_obs = warm_pixels.reshape(-1, 2)
    cold_began = time.perf_counter()
    ga_ba.bundle_adjust(
        ga_ba.BundleProblem(
            warm_motors, warm_intrinsics, warm_world.clone(), warm_obs, warm_cam, warm_pt
        ),
        iterations=2,
    )
    cold_ga = time.perf_counter() - cold_began
    qt_ba.bundle_adjust(
        qt_ba.QuaternionBundleProblem(
            qt_ba.poses_from_matrices(ga_motor.motor_to_matrix(warm_motors)),
            warm_intrinsics, warm_world.clone(), warm_obs, warm_cam, warm_pt,
        ),
        iterations=2,
    )

    began = time.perf_counter()
    ga_out, ga_stats = ga_ba.bundle_adjust(ga_problem, iterations=args.iterations)
    ga_time = time.perf_counter() - began

    began = time.perf_counter()
    qt_out, qt_stats = qt_ba.bundle_adjust(qt_problem, iterations=args.iterations)
    qt_time = time.perf_counter() - began

    print(f"\nPoint bundle adjustment  ({views} cameras, {points} points, float64)")
    print(f"  initial reprojection RMSE: {initial:.4f} px")
    print(f"  {'arm':<22}{'final RMSE':>14}{'iters':>8}{'seconds':>10}")
    print(f"  {'GA motor':<22}{ga_stats['rmse']:>14.3e}{ga_stats['iterations']:>8}{ga_time:>10.3f}")
    print(f"  {'quaternion control':<22}{qt_stats['rmse']:>14.3e}{qt_stats['iterations']:>8}{qt_time:>10.3f}")

    # Gate 1, checked three ways. Rotations are the sharpest: a similarity gauge
    # leaves them untouched, so they compare directly with no alignment.
    cost_gap = abs(ga_stats["final_cost"] - qt_stats["final_cost"])
    rotation_gap = float(
        (
            ga_motor.motor_to_matrix(ga_out.motors)[:, :3, :3]
            - qt_ba.poses_to_matrices(qt_out.poses)[:, :3, :3]
        )
        .abs()
        .max()
    )
    _, structure_gap = ga_ba.align_similarity(ga_out.points, qt_out.points)

    print("\n  Gate 1 -- the arms reach the same optimum")
    print(f"    |cost_GA - cost_control|            {cost_gap:.3e}   (want < {args.cost_tol:.0e})")
    print(f"    rotation disagreement, unaligned    {rotation_gap:.3e}   (want < {args.pose_tol:.0e})")
    print(f"    structure, after similarity align   {structure_gap:.3e}   (want < {args.pose_tol:.0e})")

    ratio = ga_time / max(qt_time, 1e-9)
    print(f"\n  Gate 2 -- runtime ratio GA/control    {ratio:.2f}x   (want < {args.runtime_budget:.1f}x)")
    print(f"    (kingdon's one-time operator compile, excluded above: {cold_ga:.2f}s)")

    # Accuracy against ground truth, for context rather than as a gate.
    _, truth_gap = ga_ba.align_similarity(ga_out.points, world)
    print(f"\n  (context) GA structure vs ground truth, aligned: {truth_gap:.3e}")

    return [
        ("Gate 1: cost agreement", cost_gap < args.cost_tol, f"{cost_gap:.3e}"),
        ("Gate 1: rotation agreement", rotation_gap < args.pose_tol, f"{rotation_gap:.3e}"),
        ("Gate 1: structure agreement", structure_gap < args.pose_tol, f"{structure_gap:.3e}"),
        ("Gate 2: runtime ratio", ratio < args.runtime_budget, f"{ratio:.2f}x"),
    ]


def run_triangulation(args) -> list[tuple[str, bool, str]]:
    motors, intrinsics, world, pixels, gen = make_scene(6, 2000, args.seed + 1)
    noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * 2.0

    print("\nTriangulation  (6 cameras, 2000 points, 2.0 px noise)")
    print(f"  {'solver':<16}{'3D RMSE':>12}{'pixel RMSE':>14}")
    errors = {}
    for name, solver in (
        ("linear", ga_tri.triangulate_linear),
        ("midpoint", ga_tri.triangulate_midpoint),
        ("reprojection", ga_tri.triangulate_reprojection),
    ):
        estimate, _ = solver(motors, intrinsics, noisy)
        spatial = float((estimate - world).norm(dim=-1).pow(2).mean().sqrt())
        residual, valid = ga_tri.reprojection_residuals(estimate, motors, intrinsics, noisy)
        pixel = float(residual[valid].pow(2).sum(-1).mean().sqrt())
        errors[name] = (spatial, pixel)
        print(f"  {name:<16}{spatial:>12.5f}{pixel:>14.5f}")

    # The reprojection solver is the maximum-likelihood one under Gaussian pixel
    # noise, so it must win on pixel residual. If it does not, something regressed.
    best = errors["reprojection"][1] < min(errors["linear"][1], errors["midpoint"][1])
    print(f"    reprojection solver minimizes pixel residual: {best}")
    return [("Triangulation: reprojection solver is best", best, f"{errors['reprojection'][1]:.5f}")]


def run_lines(args) -> list[tuple[str, bool, str]]:
    views, lines = 6, 40
    gen = torch.Generator().manual_seed(args.seed + 2)
    intrinsics = (
        torch.tensor(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        .expand(views, 3, 3)
        .contiguous()
    )
    stand_off = torch.zeros(views, 6, dtype=DTYPE)
    stand_off[:, 5] = -4.0
    motors = ga_motor.motor_compose(
        ga_motor.motor_exp(stand_off),
        ga_motor.motor_exp(torch.randn(views, 6, generator=gen, dtype=DTYPE) * 0.2),
    )
    line_motors = ga_motor.motor_exp(torch.randn(lines, 6, generator=gen, dtype=DTYPE) * 0.5)
    base = ga_ba.canonical_line()
    world_lines = ga_motor.motor_apply_line(line_motors, base.expand(lines, 6))

    camera_idx = torch.arange(views).repeat_interleave(lines)
    line_idx = torch.arange(lines).repeat(views)
    image_lines = ga_camera.project_line(
        motors[camera_idx], intrinsics[camera_idx], world_lines[line_idx]
    )

    delta_cam = torch.randn(views, 6, generator=gen, dtype=DTYPE) * 0.015
    delta_cam[0] = 0.0
    start = ga_ba.LineBundleProblem(
        ga_motor.motor_compose(ga_motor.motor_exp(delta_cam), motors),
        intrinsics,
        ga_motor.motor_compose(
            ga_motor.motor_exp(torch.randn(lines, 6, generator=gen, dtype=DTYPE) * 0.03),
            line_motors,
        ),
        image_lines,
        camera_idx,
        line_idx,
    )
    initial = float(ga_ba.line_residuals(start).pow(2).sum(-1).mean().sqrt())
    began = time.perf_counter()
    refined, stats = ga_ba.bundle_adjust_lines(start, iterations=args.iterations)
    elapsed = time.perf_counter() - began

    # Directions are scale-invariant, so they compare with no alignment at all.
    got = ga_prim.line_direction(ga_prim.normalize_line(refined.world_lines()))
    want = ga_prim.line_direction(ga_prim.normalize_line(world_lines))
    direction_gap = float(
        torch.minimum((got - want).abs().amax(-1), (got + want).abs().amax(-1)).max()
    )

    print(f"\nLine bundle adjustment  ({views} cameras, {lines} lines)  -- claim C1")
    print(f"  initial residual RMSE: {initial:.5f}")
    print(f"  final residual RMSE:   {stats['rmse']:.3e}  ({stats['iterations']} iters, {elapsed:.2f}s)")
    print(f"  line directions vs truth, unaligned: {direction_gap:.3e}")
    return [
        ("Claim C1: line BA converges", stats["rmse"] < 1e-9, f"{stats['rmse']:.3e}"),
        ("Claim C1: line directions exact", direction_gap < 1e-9, f"{direction_gap:.3e}"),
    ]


def make_two_view_scene(points, seed, pixel_noise=0.0, outlier_fraction=0.0):
    """A calibrated pair with camera A at the identity, so world == frame A."""
    gen = torch.Generator().manual_seed(seed)
    intrinsics = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=DTYPE
    )
    true_relative = ga_motor.motor_exp(
        torch.tensor([0.06, -0.09, 0.03, 0.35, -0.15, 0.05], dtype=DTYPE)
    )
    pose_b = ga_motor.motor_inverse(true_relative)
    world = torch.randn(points, 3, generator=gen, dtype=DTYPE)
    world[:, 2] += 6.0
    identity = ga_motor.motor_identity(dtype=DTYPE)
    points_a, _ = ga_camera.project(
        identity.expand(points, 8), intrinsics.expand(points, 3, 3), world
    )
    points_b, _ = ga_camera.project(
        pose_b.expand(points, 8), intrinsics.expand(points, 3, 3), world
    )
    if pixel_noise:
        points_a = points_a + torch.randn(points_a.shape, generator=gen, dtype=DTYPE) * pixel_noise
        points_b = points_b + torch.randn(points_b.shape, generator=gen, dtype=DTYPE) * pixel_noise
    count = int(points * outlier_fraction)
    if count:
        index = torch.randperm(points, generator=gen)[:count]
        points_b[index] = torch.rand(count, 2, generator=gen, dtype=DTYPE) * torch.tensor(
            [640.0, 480.0], dtype=DTYPE
        )
    return points_a, points_b, intrinsics, true_relative


def pose_errors(estimate, reference):
    """``(rotation_deg, translation_direction_deg)``; translation is up to scale and sign."""
    est = ga_motor.motor_to_matrix(estimate)
    ref = ga_motor.motor_to_matrix(reference)
    cosine = ((torch.trace(est[:3, :3].T @ ref[:3, :3]) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = float(torch.arccos(cosine) * 180.0 / torch.pi)
    aligned = (
        torch.nn.functional.cosine_similarity(
            est[:3, 3].unsqueeze(0), ref[:3, 3].unsqueeze(0)
        )
        .abs()
        .clamp(max=1.0)
    )
    return rotation, float(torch.arccos(aligned) * 180.0 / torch.pi)


def run_two_view(args) -> list[tuple[str, bool, str]]:
    """Sweep seeds rather than reporting one.

    Single-seed numbers here are actively misleading: at 10% outliers two seeds
    in six land near 1.5 deg / 10 deg, and at 20% one in six collapses to
    7 deg / 84 deg. Only the median and the worst case together describe the
    estimator.
    """
    seeds = range(6)
    print(f"\nTwo-view relative pose  (300 correspondences, 1.5 px threshold, {len(list(seeds))} seeds)")
    print(f"  {'noise':>7}{'outliers':>10}{'rot med':>10}{'rot max':>10}{'t-dir med':>12}{'t-dir max':>12}")

    checks = []
    for sigma, fraction in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.1), (0.5, 0.2)):
        rotations, directions = [], []
        for seed in seeds:
            points_a, points_b, intrinsics, truth = make_two_view_scene(
                300, seed, pixel_noise=sigma, outlier_fraction=fraction
            )
            motor, _ = ga_tv.ransac_relative_motor(
                points_a, points_b, intrinsics, threshold_px=1.5, iterations=200
            )
            rotation, direction = pose_errors(motor, truth)
            rotations.append(rotation)
            directions.append(direction)
        rotations.sort()
        directions.sort()
        median_rotation = rotations[len(rotations) // 2]
        median_direction = directions[len(directions) // 2]
        print(
            f"  {sigma:>7.1f}{fraction * 100:>9.0f}%{median_rotation:>10.3f}"
            f"{max(rotations):>10.3f}{median_direction:>12.3f}{max(directions):>12.3f}"
        )
        # Only the clean regime is gated. The outlier regimes are reported
        # because they are informative, not asserted because they are not
        # dependable -- see the note below.
        if fraction == 0.0:
            checks.append(
                (
                    f"Two-view: clean, {sigma}px noise",
                    max(rotations) < 0.5 and max(directions) < 1.5,
                    f"max {max(rotations):.3f} deg / {max(directions):.3f} deg",
                )
            )

    print(
        "  note: outlier regimes are reported, not gated. At 10% outliers two seeds\n"
        "        in six reach ~1.5 deg / ~10 deg; at 20% one in six collapses to\n"
        "        ~7 deg / ~84 deg. Raising the RANSAC budget does not help (200 and\n"
        "        800 iterations agree bit for bit), so it is one wrong attractor, not\n"
        "        a sampling shortage. A five-point solver and MSAC are the fix; see\n"
        "        docs/ga-sfm.md. This is the honest boundary of the two-view stage."
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-tol", type=float, default=1e-12)
    parser.add_argument("--pose-tol", type=float, default=1e-9)
    parser.add_argument("--runtime-budget", type=float, default=3.0)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["bundle", "triangulation", "lines", "twoview"]
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("=" * 74)
    print("Geometric-algebra structure from motion -- gate report")
    print("=" * 74)
    print(
        "\nMotors are isomorphic to unit dual quaternions and PGA bivectors are\n"
        "exactly se(3), so the GA arm and the control are the same estimator in\n"
        "different coordinates. Agreement is the passing condition; no accuracy\n"
        "advantage is claimed, because none is available."
    )

    checks: list[tuple[str, bool, str]] = []
    if "bundle" not in args.skip:
        checks += run_bundle_gates(args)
    if "triangulation" not in args.skip:
        checks += run_triangulation(args)
    if "lines" not in args.skip:
        checks += run_lines(args)
    if "twoview" not in args.skip:
        checks += run_two_view(args)

    print("\n" + "=" * 74)
    failures = 0
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"  [{status}] {name:<44} {detail}")
    print("=" * 74)
    if failures:
        print(f"\n{failures} of {len(checks)} checks FAILED")
        return 1
    print(f"\nall {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
