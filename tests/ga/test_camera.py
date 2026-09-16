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
"""Pinhole cameras with motor extrinsics.

The properties checked here are the ones the SfM stages rely on: a point lies
on the ray it projected to, the ray is the meet of its two constraint planes,
and every ray passes through the camera centre.
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat.contrib.ga import camera as cam
from gsplat.contrib.ga import motor as mot
from gsplat.contrib.ga import primitives as prim
from tests.ga._helpers import synthetic_scene

DTYPE = torch.float64


class TestProjection:
    def test_projection_matches_the_matrix_pipeline(self):
        motors, intrinsics, world, pixels = synthetic_scene()
        views, num = pixels.shape[0], pixels.shape[1]
        matrices = mot.motor_to_matrix(motors).numpy()
        k = intrinsics.numpy()

        for v in range(views):
            cam_pts = world.numpy() @ matrices[v][:3, :3].T + matrices[v][:3, 3]
            want = (cam_pts / cam_pts[:, 2:3]) @ k[v].T
            np.testing.assert_allclose(pixels[v].numpy(), want[:, :2], atol=1e-9, rtol=0)

    def test_points_behind_the_camera_are_marked_invalid(self):
        motors, intrinsics, _, _ = synthetic_scene(views=1)
        behind = cam.camera_to_world(
            motors[0], torch.tensor([[0.1, 0.1, -2.0]], dtype=DTYPE)
        )
        _, valid = cam.project(motors[0], intrinsics[0], behind)
        assert not bool(valid.any())

    def test_projection_stays_finite_behind_the_camera(self):
        """Invalid projections must not poison a batch with inf/nan."""
        motors, intrinsics, _, _ = synthetic_scene(views=1)
        pts = cam.camera_to_world(
            motors[0],
            torch.tensor([[0.1, 0.1, -2.0], [0.0, 0.0, 0.0], [0.1, 0.1, 5.0]], dtype=DTYPE),
        )
        pixels, valid = cam.project(motors[0], intrinsics[0], pts)
        assert torch.isfinite(pixels).all()
        assert valid.tolist() == [False, False, True]

    def test_pixel_direction_inverts_projection(self):
        motors, intrinsics, world, pixels = synthetic_scene(views=2, points=64)
        for v in range(2):
            cam_pts = cam.world_to_camera(motors[v], world)
            direction = cam.pixel_direction(intrinsics[v], pixels[v])
            torch.testing.assert_close(
                direction * cam_pts[:, 2:3], cam_pts, atol=1e-9, rtol=0
            )


class TestRays:
    def test_point_lies_on_its_own_ray(self):
        motors, intrinsics, world, pixels = synthetic_scene()
        views, num = pixels.shape[0], pixels.shape[1]
        rays = cam.pixel_ray(
            motors[:, None, :].expand(views, num, 8),
            intrinsics[:, None, :, :].expand(views, num, 3, 3),
            pixels,
        )
        distance = prim.point_line_distance(world.expand(views, num, 3), rays)
        assert float(distance.abs().max()) < 1e-9

    def test_camera_centre_lies_on_every_ray(self):
        motors, intrinsics, _, pixels = synthetic_scene()
        views, num = pixels.shape[0], pixels.shape[1]
        rays = cam.pixel_ray(
            motors[:, None, :].expand(views, num, 8),
            intrinsics[:, None, :, :].expand(views, num, 3, 3),
            pixels,
        )
        centres = cam.camera_center(motors)[:, None, :].expand(views, num, 3)
        assert float(prim.point_line_distance(centres, rays).abs().max()) < 1e-9

    def test_ray_is_the_meet_of_its_constraint_planes(self):
        motors, intrinsics, _, pixels = synthetic_scene()
        views, num = pixels.shape[0], pixels.shape[1]
        motors_e = motors[:, None, :].expand(views, num, 8)
        intr_e = intrinsics[:, None, :, :].expand(views, num, 3, 3)

        ray = prim.normalize_line(cam.pixel_ray(motors_e, intr_e, pixels))
        plane_u, plane_v = cam.pixel_ray_planes(motors_e, intr_e, pixels)
        meet = prim.normalize_line(prim.meet_planes(plane_u, plane_v))
        # Lines carry no orientation, so compare up to sign.
        agree = torch.minimum((meet - ray).abs().amax(-1), (meet + ray).abs().amax(-1))
        assert float(agree.max()) < 1e-9

    def test_point_lies_on_both_constraint_planes(self):
        motors, intrinsics, world, pixels = synthetic_scene()
        views, num = pixels.shape[0], pixels.shape[1]
        plane_u, plane_v = cam.pixel_ray_planes(
            motors[:, None, :].expand(views, num, 8),
            intrinsics[:, None, :, :].expand(views, num, 3, 3),
            pixels,
        )
        expanded = world.expand(views, num, 3)
        for plane in (plane_u, plane_v):
            assert float(prim.point_plane_distance(expanded, plane).abs().max()) < 1e-9

    def test_camera_centre_matches_the_matrix_form(self):
        motors, _, _, _ = synthetic_scene()
        matrices = mot.motor_to_matrix(motors)
        want = -torch.einsum(
            "vij,vj->vi", matrices[:, :3, :3].transpose(-2, -1), matrices[:, :3, 3]
        )
        torch.testing.assert_close(cam.camera_center(motors), want, atol=1e-10, rtol=0)


class TestRoundTrips:
    def test_world_camera_round_trip(self):
        motors, _, world, _ = synthetic_scene()
        for v in range(motors.shape[0]):
            back = cam.camera_to_world(motors[v], cam.world_to_camera(motors[v], world))
            torch.testing.assert_close(back, world, atol=1e-10, rtol=0)

    def test_projection_is_differentiable_in_the_motor(self):
        motors, intrinsics, world, pixels = synthetic_scene(views=1, points=16)
        biv = mot.motor_log(motors[0]).detach().requires_grad_(True)

        def fn(b):
            return cam.project(mot.motor_exp(b), intrinsics[0], world)[0]

        assert torch.autograd.gradcheck(fn, (biv,), eps=1e-6, atol=1e-5)
