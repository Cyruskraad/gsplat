# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""The reference renderer, checked against the algebra rather than against a
picture that looks about right.

This renderer's whole value is being independent of ``gsplat.rasterization``,
so every expectation here is derived from the projection maths by hand. Where a
number appears it is worked out in the docstring: a peak alpha, a pixel
coordinate, an occlusion ordering.

The most important test in the file is the linearity one. The GPU parity test
shows Path A and Path B agree *with each other*, which they would even if the
kernel were not linear in the per-primitive feature -- both paths would simply
be wrong together. Linearity of this renderer, derived from compositing rather
than from the kernel, is what makes that parity mean something.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from atlas.functional.transport import composite, compositing_weights  # noqa: E402
from atlas.reference import (  # noqa: E402
    MAX_ALPHA,
    covariance_3d,
    look_at,
    pinhole_intrinsics,
    project_gaussians,
    quaternion_to_rotation,
    render_reference,
)

IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)


def _scene(positions, scale=0.15, opacity_logit=6.0, colors=None):
    num = len(positions)
    means = torch.tensor(positions, dtype=torch.float64)
    quats = IDENTITY_QUAT.expand(num, 4).contiguous()
    log_scales = torch.full((num, 3), math.log(scale), dtype=torch.float64)
    opacities = torch.full((num,), float(opacity_logit), dtype=torch.float64)
    if colors is None:
        colors = torch.ones(num, 3, dtype=torch.float64)
    return (
        means,
        quats,
        log_scales,
        opacities,
        torch.as_tensor(colors, dtype=torch.float64),
    )


def _camera(width=64, height=64, distance=4.0, fov=45.0):
    K = pinhole_intrinsics(width, height, fov)
    view = look_at(torch.tensor([0.0, -distance, 0.0]), torch.zeros(3))
    return view, K, width, height


# --- rotations and covariances ---------------------------------------------


def test_the_identity_quaternion_is_the_identity_rotation():
    assert torch.allclose(
        quaternion_to_rotation(IDENTITY_QUAT)[0], torch.eye(3, dtype=torch.float64)
    )


def test_a_quarter_turn_about_z_sends_x_to_y():
    root = math.sqrt(0.5)
    quats = torch.tensor([[root, 0.0, 0.0, root]], dtype=torch.float64)
    rotated = quaternion_to_rotation(quats)[0] @ torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float64
    )
    assert torch.allclose(
        rotated, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12
    )


def test_an_unnormalised_quaternion_is_normalised_rather_than_scaling_the_scene():
    doubled = quaternion_to_rotation(IDENTITY_QUAT * 2.0)[0]
    assert torch.allclose(doubled, torch.eye(3, dtype=torch.float64))


def test_an_isotropic_covariance_is_the_scale_squared():
    covariance = covariance_3d(IDENTITY_QUAT, torch.full((1, 3), math.log(0.2)))
    assert torch.allclose(
        covariance[0], 0.04 * torch.eye(3, dtype=torch.float64), atol=1e-14
    )


def test_an_anisotropic_covariance_keeps_its_axes_under_no_rotation():
    log_scales = torch.log(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64))
    covariance = covariance_3d(IDENTITY_QUAT, log_scales)
    assert torch.allclose(
        covariance[0], torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))
    )


def test_a_rotated_covariance_keeps_its_determinant():
    """Rotation cannot change the volume of the ellipsoid; if it does, the
    rotation is being applied on the wrong side."""
    quats = torch.tensor([[0.3, -0.5, 0.7, 0.2]], dtype=torch.float64)
    log_scales = torch.log(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64))
    determinant = torch.linalg.det(covariance_3d(quats, log_scales)[0])
    assert float(determinant) == pytest.approx(36.0, rel=1e-12)


# --- projection -------------------------------------------------------------


def test_a_gaussian_on_the_optical_axis_lands_on_the_principal_point():
    means, quats, log_scales, opacities, _ = _scene([[0.0, 0.0, 0.0]])
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height
    )
    assert projection.mean2d[0].tolist() == pytest.approx([31.5, 31.5], abs=1e-12)
    assert float(projection.depth[0]) == pytest.approx(4.0, abs=1e-12)


def test_an_offset_gaussian_lands_where_the_pinhole_says():
    """``u = fx * x / z + cx``. At a 45 degree vertical field over 64 pixels,
    ``fx = 32 / tan(22.5 deg) = 77.2548``; a primitive 0.5 m to camera-right at
    4 m gives ``77.2548 * 0.5 / 4 + 31.5 = 41.1569``."""
    means, quats, log_scales, opacities, _ = _scene([[0.5, 0.0, 0.0]])
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height
    )
    focal = 32.0 / math.tan(math.radians(22.5))
    assert float(projection.mean2d[0, 0]) == pytest.approx(focal * 0.5 / 4.0 + 31.5)
    assert float(projection.mean2d[0, 1]) == pytest.approx(31.5)


def test_the_peak_alpha_is_the_value_the_gaussian_formula_gives():
    """The whole projection chain in one number.

    Focal 77.2548, world scale 0.15 at 4 m gives a screen sigma of
    ``77.2548 * 0.15 / 4 = 2.8971`` px, so the variance is 8.3931, plus the
    0.3 dilation gives 8.6931. The nearest pixel centre sits half a pixel from
    the primitive's centre in each axis, so ``d^2 = 0.5``, and
    ``alpha = sigmoid(6) * exp(-0.5 * 0.5 / 8.6931) = 0.99753 * 0.97164``.
    """
    means, quats, log_scales, opacities, colors = _scene([[0.0, 0.0, 0.0]])
    view, K, width, height = _camera()
    _, alpha = render_reference(
        means, quats, log_scales, opacities, colors, view, K, width, height
    )
    variance = (32.0 / math.tan(math.radians(22.5)) * 0.15 / 4.0) ** 2 + 0.3
    expected = (1.0 / (1.0 + math.exp(-6.0))) * math.exp(-0.5 * 0.5 / variance)
    assert float(alpha.max()) == pytest.approx(expected, rel=1e-9)


def test_the_off_axis_conic_matches_a_numerically_differentiated_jacobian():
    """The perspective term in the Jacobian, which only exists off-axis.

    ``J`` is differentiated by central differences on the world-to-pixel map
    rather than written out again, so this is an independent derivation and not
    the same algebra transcribed twice. On-axis primitives cannot see the
    difference -- the term carries a factor of ``x`` -- which is why every other
    projection test in this file would pass with it deleted.
    """
    means, quats, log_scales, opacities, _ = _scene([[0.6, 0.4, -0.5]], scale=0.2)
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height, dilation=0.0
    )

    def to_pixel(point):
        camera = view[:3, :3] @ point + view[:3, 3]
        return K[:2, :2] @ (camera[:2] / camera[2]) + K[:2, 2]

    step = 1e-6
    centre = means[0]
    columns = []
    for axis in range(3):
        offset = torch.zeros(3, dtype=torch.float64)
        offset[axis] = step
        columns.append(
            (to_pixel(centre + offset) - to_pixel(centre - offset)) / (2 * step)
        )
    jacobian = torch.stack(columns, dim=-1)  # [2, 3], world -> pixel

    covariance = covariance_3d(quats, log_scales)[0]
    expected = jacobian @ covariance @ jacobian.T
    a, b, c = expected[0, 0], expected[0, 1], expected[1, 1]
    determinant = a * c - b * b
    expected_conic = torch.stack([c, -b, a]) / determinant

    assert torch.allclose(projection.conic[0], expected_conic, rtol=1e-6), (
        projection.conic[0].tolist(),
        expected_conic.tolist(),
    )
    # The premise: the primitive really is off-axis, so the term being tested
    # is not multiplied by zero.
    assert abs(float(projection.mean2d[0, 0]) - 31.5) > 5.0


def test_a_gaussian_behind_the_camera_is_culled():
    means, quats, log_scales, opacities, _ = _scene([[0.0, -8.0, 0.0]])
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height
    )
    assert projection.index.numel() == 0


def test_a_gaussian_far_outside_the_frame_is_culled():
    means, quats, log_scales, opacities, _ = _scene([[40.0, 0.0, 0.0]], scale=0.05)
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height
    )
    assert projection.index.numel() == 0


def test_projection_returns_primitives_sorted_front_to_back():
    means, quats, log_scales, opacities, _ = _scene(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    view, K, width, height = _camera()
    projection = project_gaussians(
        means, quats, log_scales, opacities, view, K, width, height
    )
    assert projection.index.tolist() == [1, 2, 0]
    assert projection.depth.tolist() == sorted(projection.depth.tolist())


# --- compositing ------------------------------------------------------------


def test_the_near_primitive_occludes_the_far_one():
    """Two Gaussians on the optical axis, red in front of green. An opacity
    logit of 6 is 0.9975 opaque, so the front one dominates."""
    means, quats, log_scales, opacities, colors = _scene(
        # Far first, so this fails if the depth sort is removed.
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        colors=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    view, K, width, height = _camera()
    image, _ = render_reference(
        means, quats, log_scales, opacities, colors, view, K, width, height
    )
    centre = image[31, 31]
    assert float(centre[0]) > 0.9
    assert float(centre[1]) < 0.05


def test_swapping_the_depths_swaps_which_colour_wins():
    view, K, width, height = _camera()
    front_red = _scene(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    front_green = _scene(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], colors=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    first, _ = render_reference(*front_red, view, K, width, height)
    second, _ = render_reference(*front_green, view, K, width, height)
    # Exchanging the two colours exchanges channels 0 and 1 and leaves channel 2
    # alone. It is not a reversal: the two primitives sit at different depths,
    # so their alphas differ and the pixel is not symmetric in any other sense.
    assert torch.allclose(first[31, 31], second[31, 31][[1, 0, 2]], atol=1e-12)


def test_an_opaque_primitive_hides_what_is_behind_it_entirely():
    """The alpha clamp at 0.999 is the only thing that gets through, and it is
    deliberate: a true 1.0 would zero the gradient of everything behind it.

    An odd frame size puts the principal point at pixel 32 exactly, so the
    primitive's centre lands on a pixel centre and its Gaussian falloff is
    exactly 1. Without that the falloff alone drops the front alpha to 0.97 and
    the test measures the footprint rather than the occlusion.
    """
    means, quats, log_scales, opacities, colors = _scene(
        # The far primitive first: array order and depth order must disagree,
        # or the depth sort could be deleted and this would still pass.
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        opacity_logit=40.0,
        colors=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    view, K, width, height = _camera(width=65, height=65)
    image, alpha = render_reference(
        means, quats, log_scales, opacities, colors, view, K, width, height
    )
    assert float(alpha[32, 32]) > 0.999  # the premise: the pixel is covered
    assert float(image[32, 32, 0]) > 0.998
    # Exactly the clamp, not merely under it. `sigmoid(40)` is 1.0 to float64,
    # so without the clamp the transmittance would be 0 and the green would be
    # 0 too -- which "less than 0.001" would happily accept.
    expected_leak = (1.0 - MAX_ALPHA) * MAX_ALPHA
    assert float(image[32, 32, 1]) == pytest.approx(expected_leak, rel=1e-9)


def test_an_empty_scene_renders_black_with_zero_alpha():
    """Behind the camera, not merely far away: at 900 m a primitive still lands
    on the principal point, and the 0.3 dilation gives it a sub-pixel footprint
    with an alpha of 0.43. Distance does not cull; the near plane does."""
    means, quats, log_scales, opacities, colors = _scene([[0.0, -900.0, 0.0]])
    view, K, width, height = _camera()
    image, alpha = render_reference(
        means, quats, log_scales, opacities, colors, view, K, width, height
    )
    assert float(image.abs().max()) == 0.0 and float(alpha.max()) == 0.0


def test_the_background_shows_through_where_alpha_is_short():
    means, quats, log_scales, opacities, colors = _scene([[0.0, 0.0, 0.0]])
    view, K, width, height = _camera()
    image, alpha = render_reference(
        means,
        quats,
        log_scales,
        opacities,
        colors,
        view,
        K,
        width,
        height,
        background=torch.tensor([0.25, 0.5, 0.75]),
    )
    assert image[0, 0].tolist() == pytest.approx([0.25, 0.5, 0.75])
    covered = image[31, 31]
    assert float(covered[0]) == pytest.approx(
        float(alpha[31, 31]) + (1 - float(alpha[31, 31])) * 0.25
    )


# --- the property that makes this an oracle --------------------------------


def test_the_render_is_linear_in_the_per_primitive_colour():
    """The exactness claim, checked on a renderer derived from compositing
    rather than from the CUDA kernel.

    If this fails, the whole method fails: compositing weights would depend on
    the feature being composited, and contract-then-splat could not equal
    splat-then-contract.
    """
    means, quats, log_scales, opacities, _ = _scene(
        [[0.0, -0.5, 0.0], [0.2, 0.3, 0.1], [-0.3, 0.6, -0.2]]
    )
    view, K, width, height = _camera()
    generator = torch.Generator().manual_seed(0)
    first = torch.rand(3, 3, generator=generator, dtype=torch.float64)
    second = torch.rand(3, 3, generator=generator, dtype=torch.float64)

    def render(colors):
        return render_reference(
            means, quats, log_scales, opacities, colors, view, K, width, height
        )[0]

    additive = render(first + second) - (render(first) + render(second))
    assert float(additive.abs().max()) < 1e-14, float(additive.abs().max())

    homogeneous = render(2.5 * first) - 2.5 * render(first)
    assert float(homogeneous.abs().max()) < 1e-14
    # The premise: the scene is not transparent, so this is not 0 == 0.
    assert float(render(first).abs().max()) > 0.1


def test_the_matmul_shortcut_equals_the_reference_compositor():
    """``render_reference`` reduces with a matmul instead of calling
    ``composite``, because the features are the same at every pixel. The
    subtle part -- the exclusive cumulative product -- is still
    ``compositing_weights``; this pins the shortcut around it."""
    generator = torch.Generator().manual_seed(1)
    alphas = torch.rand(7, 5, generator=generator, dtype=torch.float64)
    features = torch.rand(5, 3, generator=generator, dtype=torch.float64)
    weights = compositing_weights(alphas)
    # `composite` wants one feature vector per (pixel, primitive); the shortcut
    # exists precisely because they are all the same vector.
    expanded = features.unsqueeze(0).expand(7, 5, 3)
    assert torch.allclose(weights @ features, composite(alphas, expanded), atol=1e-15)


def test_the_channel_count_is_whatever_it_is_given():
    """Three for radiance, 3B for a packed transport splat. Path B's oracle
    depends on this working for an arbitrary channel count."""
    means, quats, log_scales, opacities, _ = _scene([[0.0, 0.0, 0.0]])
    view, K, width, height = _camera(width=16, height=16)
    for channels in (1, 3, 24, 96):
        image, _ = render_reference(
            means,
            quats,
            log_scales,
            opacities,
            torch.ones(1, channels, dtype=torch.float64),
            view,
            K,
            width,
            height,
        )
        assert image.shape == (16, 16, channels)


# --- chunking and guards ----------------------------------------------------


@pytest.mark.parametrize("rows", [1, 3, 16, 64, 200])
def test_chunking_the_pixel_rows_does_not_change_the_image(rows):
    means, quats, log_scales, opacities, colors = _scene(
        [[0.0, -0.5, 0.0], [0.2, 0.3, 0.1]]
    )
    view, K, width, height = _camera(width=21, height=17)
    reference, _ = render_reference(
        means,
        quats,
        log_scales,
        opacities,
        colors,
        view,
        K,
        width,
        height,
        rows_per_chunk=0,
    )
    chunked, _ = render_reference(
        means,
        quats,
        log_scales,
        opacities,
        colors,
        view,
        K,
        width,
        height,
        rows_per_chunk=rows,
    )
    assert torch.equal(reference, chunked)


def test_a_colour_array_of_the_wrong_length_names_the_length_it_wanted():
    means, quats, log_scales, opacities, _ = _scene([[0.0, 0.0, 0.0]])
    view, K, width, height = _camera()
    with pytest.raises(ValueError, match=r"colors must be \[1, C\]"):
        render_reference(
            means,
            quats,
            log_scales,
            opacities,
            torch.ones(2, 3, dtype=torch.float64),
            view,
            K,
            width,
            height,
        )


def test_a_degenerate_frame_is_refused():
    means, quats, log_scales, opacities, colors = _scene([[0.0, 0.0, 0.0]])
    view, K, _, _ = _camera()
    with pytest.raises(ValueError, match="at least 1x1"):
        render_reference(means, quats, log_scales, opacities, colors, view, K, 0, 8)


def test_look_at_puts_the_target_on_the_principal_point():
    target = torch.tensor([1.0, 2.0, 3.0])
    view = look_at(torch.tensor([4.0, -3.0, 5.0]), target)
    K = pinhole_intrinsics(64, 64)
    camera = view[:3, :3] @ target.to(torch.float64) + view[:3, 3]
    pixel = K[:2, :2] @ (camera[:2] / camera[2]) + K[:2, 2]
    assert pixel.tolist() == pytest.approx([31.5, 31.5], abs=1e-9)
    assert float(camera[2]) > 0, "the target must be in front of the camera"


def test_look_at_refuses_an_up_vector_along_the_view_direction():
    with pytest.raises(ValueError, match="parallel"):
        look_at(
            torch.tensor([0.0, 0.0, 1.0]),
            torch.zeros(3),
            up=torch.tensor([0.0, 0.0, 1.0]),
        )


def test_quaternions_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match=r"quats must be \[N, 4\]"):
        quaternion_to_rotation(torch.ones(3, 3))
