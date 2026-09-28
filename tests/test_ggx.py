# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GGX ground truth is only worth sweeping against if it is a real GGX."""

import math

import torch

from atlas.data.synthetic import SyntheticConfig, generate_ggx_capture
from atlas.functional.ggx import (
    alpha_from_roughness,
    ggx_distribution,
    ggx_specular,
    microfacet_response,
)


def _hemisphere(num_theta: int = 2000, num_phi: int = 64):
    theta = (torch.arange(num_theta, dtype=torch.float64) + 0.5) * (
        math.pi / 2 / num_theta
    )
    return theta, (math.pi / 2 / num_theta) * (2 * math.pi)


def test_the_distribution_projects_to_unit_area():
    # int D(h) (n.h) dh = 1 over the hemisphere, for every roughness.
    theta, d_step = _hemisphere()
    for roughness in (0.3, 0.5, 0.8, 1.0):
        alpha = alpha_from_roughness(roughness)
        d = ggx_distribution(torch.cos(theta), alpha)
        total = float((d * torch.cos(theta) * torch.sin(theta)).sum() * d_step)
        assert abs(total - 1.0) < 2e-3, (roughness, total)


def test_the_specular_brdf_is_reciprocal():
    generator = torch.Generator().manual_seed(0)
    normals = torch.nn.functional.normalize(
        torch.randn(64, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    a = torch.nn.functional.normalize(
        normals + 0.5 * torch.randn(64, 3, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    b = torch.nn.functional.normalize(
        normals + 0.5 * torch.randn(64, 3, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    rough = torch.full((64,), 0.4, dtype=torch.float64)
    f0 = torch.full((64,), 0.5, dtype=torch.float64)
    forward = ggx_specular(normals, a, b, rough, f0)
    backward = ggx_specular(normals, b, a, rough, f0)
    assert torch.allclose(forward, backward, atol=1e-12)


def test_the_response_is_linear_in_the_incident_radiance():
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    view = torch.nn.functional.normalize(
        torch.tensor([[0.2, 0.0, 1.0]], dtype=torch.float64), dim=-1
    )
    light = torch.nn.functional.normalize(
        torch.tensor([[-0.1, 0.1, 1.0]], dtype=torch.float64), dim=-1
    )
    kwargs = dict(diffuse_albedo=torch.full((1, 3), 0.5, dtype=torch.float64))
    kwargs.update(
        specular_f0=torch.full((1,), 0.04, dtype=torch.float64),
        roughness=torch.full((1,), 0.5, dtype=torch.float64),
    )
    one = microfacet_response(normals, view, light, torch.ones(1, 3).double(), **kwargs)
    three = microfacet_response(
        normals, view, light, 3 * torch.ones(1, 3).double(), **kwargs
    )
    for single, triple in zip(one, three):
        assert torch.allclose(3 * single, triple, atol=1e-14)


def test_a_surface_facing_away_reflects_nothing():
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    below = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    above = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert (
        float(
            ggx_specular(
                normals,
                above,
                below,
                torch.full((1,), 0.5).double(),
                torch.full((1,), 0.5).double(),
            )
            .abs()
            .max()
        )
        == 0.0
    )


def test_the_ggx_capture_records_its_ground_truth(tmp_path):
    import json

    config = SyntheticConfig(num_views=4, num_lights=4, width=16, height=12)
    capture = generate_ggx_capture(tmp_path / "cap", config, roughness=0.45)
    manifest = json.loads((tmp_path / "cap" / "transforms.json").read_text())
    assert manifest["atlas"]["ground_truth"] == "ggx"
    assert manifest["atlas"]["ggx"]["roughness"] == 0.45
    assert abs(float(torch.as_tensor(capture.roughness).flatten()[0]) - 0.45) < 1e-6
    assert len(capture.frames) == len(manifest["frames"])
