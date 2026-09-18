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

"""A capture with known ground truth, so the pipeline can be wrong out loud.

Everything downstream of here -- the loader, the trainer, and the gate that
decides whether the project is worth continuing -- has so far had nothing to
run on. Waiting for the real capture would leave all three unwritten and,
worse, would make the first real run the first run of *anything*, with no way
to tell a bug in the loader from a failure of the method.

This generates the whole thing: geometry, transport fitted to an analytic BRDF,
cameras, lights, images rendered by :mod:`atlas.reference`, and a manifest in
the format the real loader will read. Because the answer is known, three
questions become checkable that a real capture can never settle:

* Does the loader reproduce what the generator wrote?
* Does ``solve_flash_offset`` recover an offset that was *planted*?
* Does the held-out-light gate **fail** when it should, not merely pass?

## The flash-placement problem, which this module exists to make visible

``docs/relighting-atlas.md`` specifies an **off-camera** flash whose position is
recovered per shot from chrome spheres. When the chrome-sphere *position* gate
turned out to be geometrically unreachable it was replaced by a bracket-mounted
flash at one fitted camera-frame offset. Those two are not interchangeable, and
the difference is not a detail of calibration:

    If the light is a fixed function of the camera, then holding out a light
    holds out its view as well. ``split_lights`` and ``split_views`` select the
    same shots, the two reported numbers are the same number, and the gate
    passes no matter what the model learned.

So this module supports both, and names them:

``flash_mode="free"``
    Light positions chosen independently of the cameras -- a second operator,
    or a flash on a stand moved between shots. Views and lights form a grid and
    the two splits are genuinely different measurements.

``flash_mode="bracket"``
    Light rigidly attached to the camera at a constant offset. This is what the
    calibration code targets, and it is a perfectly good capture for geometry,
    for the loader and for the trainer. It cannot support the gate, and
    :func:`generate_capture` says so in the manifest rather than leaving it to
    be discovered later.

## Conventions

The manifest is a genuine NeRF ``transforms.json`` -- ``transform_matrix`` is
camera-to-world in OpenGL axes -- so that other tools can read it, with the
relighting-specific fields under an ``atlas`` key per frame. The loader converts
to the OpenCV world-to-camera matrices the renderer uses;
:func:`viewmat_to_nerf` and :func:`nerf_to_viewmat` are that conversion and are
tested as a round trip, because getting it wrong flips the object top to bottom
and looks almost plausible.

Images are written as **linear** 16-bit-equivalent PNG at 8 bits with a stated
scale factor, not as sRGB. Nothing in this project does arithmetic on
gamma-encoded values, and a synthetic capture that quietly did would teach the
loader the wrong habit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..functional.atoms import (
    default_sharpness,
    evaluate_atoms,
    fibonacci_sphere,
    make_sg_atoms,
    project_point_light,
)
from ..functional.nearfield import incident_radiance
from ..functional.transport import contract
from ..imageio import write_png
from ..reference import look_at, pinhole_intrinsics, render_reference

__all__ = [
    "SyntheticConfig",
    "fit_lobe_transport",
    "generate_capture",
    "viewmat_to_nerf",
    "nerf_to_viewmat",
    "GeneratedCapture",
]

#: Flip between OpenGL (y up, z backward) and OpenCV (y down, z forward). It is
#: its own inverse, which is why one constant serves both directions.
_AXIS_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))


def viewmat_to_nerf(viewmat: Tensor) -> Tensor:
    """OpenCV world-to-camera ``[4, 4]`` to a NeRF ``transform_matrix``."""
    return torch.linalg.inv(viewmat.to(torch.float64)) @ _AXIS_FLIP


def nerf_to_viewmat(transform_matrix: Tensor) -> Tensor:
    """A NeRF ``transform_matrix`` back to OpenCV world-to-camera ``[4, 4]``."""
    return torch.linalg.inv(transform_matrix.to(torch.float64) @ _AXIS_FLIP)


# --- fitting a BRDF onto the atom basis ------------------------------------


def fit_lobe_transport(
    normals: Tensor,
    albedo: Tensor,
    axes: Tensor,
    sharpnesses: Tensor,
    *,
    specular: float = 0.0,
    shininess: float = 20.0,
    num_samples: int = 512,
    ridge: float = 1e-8,
) -> Tensor:
    """Least-squares transport reproducing a clamped-cosine plus backscatter lobe.

    The transport must satisfy ``sum_k M[c,k] A_k(w) = f_c(w)`` for every
    incident direction ``w``, where ``f`` is the surface's response. That is a
    function-fitting problem, not a quadrature: the atoms are a frame rather
    than an orthonormal basis, so the coefficients come from a least-squares
    solve against the atoms evaluated on a dense set of directions.

    The response is

    .. math::
        f_c(w) = \\frac{k_{d,c}}{\\pi} \\max(0, n \\cdot w)
                 + k_s \\max(0, n \\cdot w)^p

    Both terms are functions of the incident direction alone, which is what
    makes them expressible here at all. The second is a **backscatter** lobe
    rather than a mirror lobe, and that is physically the right choice for this
    capture: with the flash near the camera, the specular peak sits where the
    light is, so a view-independent lobe about ``n . w`` is not an
    approximation of convenience.

    The shininess is the knob that makes the basis inadequate on purpose. A
    ``p = 60`` lobe is far narrower than 16 spherical Gaussians can represent,
    so the fit residual is large and a model trained on it cannot generalise to
    an unseen light -- which is exactly the condition the gate must detect.

    Returns:
        ``[N, 3, B]`` transport.
    """
    if normals.ndim != 2 or normals.shape[-1] != 3:
        raise ValueError(f"normals must be [N, 3], got {tuple(normals.shape)}")
    if albedo.shape != normals.shape:
        raise ValueError(
            f"albedo must match normals {tuple(normals.shape)}, got "
            f"{tuple(albedo.shape)}"
        )
    if num_samples < axes.shape[0]:
        raise ValueError(
            f"num_samples ({num_samples}) must be at least the atom count "
            f"({axes.shape[0]}); an under-determined fit would be arbitrary"
        )

    directions = fibonacci_sphere(num_samples).to(torch.float64)  # [S, 3]
    design = evaluate_atoms(directions, axes, sharpnesses).to(torch.float64)  # [S, B]

    cosine = (normals.to(torch.float64) @ directions.T).clamp_min(0.0)  # [N, S]
    diffuse = cosine.unsqueeze(1) * (albedo.to(torch.float64) / math.pi).unsqueeze(-1)
    target = diffuse  # [N, 3, S]
    if specular > 0.0:
        lobe = specular * cosine.pow(shininess)
        target = target + lobe.unsqueeze(1)

    # One shared normal-equation solve: the design matrix is the same for every
    # primitive and channel, so it is factorised once rather than N*3 times.
    gram = design.T @ design
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    rhs = target @ design  # [N, 3, B]
    return torch.linalg.solve(gram, rhs.reshape(-1, gram.shape[0]).T).T.reshape(
        target.shape[0], 3, gram.shape[0]
    )


# --- the capture ------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticConfig:
    """Everything about a synthetic capture, so it is reproducible from a hash."""

    num_primitives: int = 96
    num_views: int = 10
    num_lights: int = 10
    num_atoms: int = 16
    width: int = 48
    height: int = 48
    fov_degrees: float = 45.0

    #: Object radius, and the shell the primitives sit on.
    object_radius: float = 0.35
    primitive_scale: float = 0.10
    opacity_logit: float = 4.0

    #: Camera orbit. The three ranges are not decoration: a perfectly regular
    #: orbit leaves the flash-offset fit rank-2, so it returns a metre-wrong
    #: answer with a residual of exactly zero. Varying all three is what makes
    #: the problem conditioned, and `tests/test_calibration.py` is where that
    #: was measured.
    camera_distance: float = 1.6
    distance_jitter: float = 0.25
    elevation_range: Tuple[float, float] = (-35.0, 55.0)
    roll_range: Tuple[float, float] = (-20.0, 20.0)

    #: "free" decouples lights from cameras and is the only mode in which the
    #: held-out-light split measures anything. "bracket" rigidly attaches the
    #: flash to the camera; see the module docstring.
    flash_mode: str = "free"
    flash_offset: Tuple[float, float, float] = (0.18, -0.06, 0.02)
    light_distance: float = 1.3
    flash_intensity: float = 3.0
    reference_distance: float = 1.0

    #: Surface response. A high `shininess` with few atoms is the deliberate
    #: way to make the basis inadequate and the gate fail.
    specular: float = 0.35
    shininess: float = 20.0

    #: Capture imperfections, all off by default.
    ambient: float = 0.0
    read_noise: float = 0.0
    exposure_jitter: float = 0.0

    scale: float = 4.0  #: linear radiance per unit of stored 8-bit code value
    seed: int = 0


@dataclass
class GeneratedCapture:
    """What :func:`generate_capture` wrote, and the answer it wrote it from."""

    root: Path
    config: SyntheticConfig
    means: Tensor
    quats: Tensor
    log_scales: Tensor
    opacity_logits: Tensor
    transport: Tensor
    atom_axes: Tensor
    atom_sharpness: Tensor
    viewmats: Tensor
    light_positions: Tensor
    intrinsics: Tensor
    frames: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def splits_are_independent(self) -> bool:
        """Whether a held-out-light split means anything on this capture."""
        return self.config.flash_mode == "free"


def _scene(
    config: SyntheticConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Primitives on a shell, with normals and albedo, deterministic in the seed."""
    generator = torch.Generator().manual_seed(config.seed)
    directions = fibonacci_sphere(config.num_primitives).to(torch.float64)
    # A little radial jitter so the surface is not a perfect analytic sphere,
    # which would let a model exploit the symmetry instead of the transport.
    wobble = 1.0 + 0.06 * (
        torch.rand(config.num_primitives, 1, generator=generator, dtype=torch.float64)
        - 0.5
    )
    means = directions * config.object_radius * wobble

    albedo = 0.25 + 0.6 * torch.rand(
        config.num_primitives, 3, generator=generator, dtype=torch.float64
    )
    quats = torch.zeros(config.num_primitives, 4, dtype=torch.float64)
    quats[:, 0] = 1.0
    log_scales = torch.full(
        (config.num_primitives, 3),
        math.log(config.primitive_scale),
        dtype=torch.float64,
    )
    opacities = torch.full(
        (config.num_primitives,), float(config.opacity_logit), dtype=torch.float64
    )
    return means, directions, albedo, quats, log_scales, opacities


def _cameras(config: SyntheticConfig) -> Tensor:
    """``[V, 4, 4]`` world-to-camera matrices on a deliberately irregular orbit."""
    generator = torch.Generator().manual_seed(config.seed + 1)
    matrices = []
    for index in range(config.num_views):
        azimuth = 2.0 * math.pi * index / config.num_views
        spread = torch.rand(3, generator=generator, dtype=torch.float64)
        low, high = config.elevation_range
        elevation = math.radians(low + (high - low) * float(spread[0]))
        distance = config.camera_distance * (
            1.0 + config.distance_jitter * (float(spread[1]) - 0.5)
        )
        roll_low, roll_high = config.roll_range
        roll = math.radians(roll_low + (roll_high - roll_low) * float(spread[2]))

        eye = torch.tensor(
            [
                distance * math.cos(elevation) * math.cos(azimuth),
                distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
            ],
            dtype=torch.float64,
        )
        up = torch.tensor(
            [
                math.sin(roll) * math.cos(azimuth),
                math.sin(roll) * math.sin(azimuth),
                math.cos(roll),
            ],
            dtype=torch.float64,
        )
        matrices.append(look_at(eye, torch.zeros(3, dtype=torch.float64), up=up))
    return torch.stack(matrices)


def _lights(config: SyntheticConfig, viewmats: Tensor) -> Tensor:
    """``[L, 3]`` light positions, and how they relate to the cameras."""
    if config.flash_mode == "bracket":
        # Rigidly attached: one light per view, at a constant camera-frame
        # offset. The two splits collapse onto each other; see the docstring.
        offset = torch.tensor(config.flash_offset, dtype=torch.float64)
        positions = []
        for viewmat in viewmats:
            rotation = viewmat[:3, :3]
            centre = -rotation.T @ viewmat[:3, 3]
            positions.append(centre + rotation.T @ offset)
        return torch.stack(positions)

    if config.flash_mode != "free":
        raise ValueError(
            f"flash_mode must be 'free' or 'bracket', got {config.flash_mode!r}"
        )

    # Independent of the cameras: a well-separated set of directions on a
    # sphere, so that farthest-point light splits have something to separate.
    generator = torch.Generator().manual_seed(config.seed + 2)
    directions = fibonacci_sphere(config.num_lights).to(torch.float64)
    jitter = 1.0 + 0.15 * (
        torch.rand(config.num_lights, 1, generator=generator, dtype=torch.float64) - 0.5
    )
    return directions * config.light_distance * jitter


def generate_capture(
    root: Path | str,
    config: Optional[SyntheticConfig] = None,
    *,
    write_images: bool = True,
    write_masks: bool = True,
) -> GeneratedCapture:
    """Render and write a whole capture. Returns it, and the answer.

    The directory is what the loader reads::

        <root>/
            transforms.json      NeRF manifest, plus an `atlas` block per frame
            images/0000.png      linear radiance, divided by `config.scale`
            masks/0000.png       object coverage, from the rendered alpha
            ground_truth.pt      the model these came from

    Args:
        root: Destination directory. Created; must not already hold a capture.
        config: Defaults to :class:`SyntheticConfig`.
        write_images: Off for tests that only want the geometry and manifest.
        write_masks: Whether to write the alpha channel as a mask.

    Returns:
        A :class:`GeneratedCapture` holding the ground-truth model, the poses,
        the light positions and the frame records.
    """
    config = config or SyntheticConfig()
    root = Path(root)
    manifest_path = root / "transforms.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; generating over a capture would "
            f"leave a mixture of two and no way to tell which is which"
        )
    (root / "images").mkdir(parents=True, exist_ok=True)
    if write_masks:
        (root / "masks").mkdir(parents=True, exist_ok=True)

    means, normals, albedo, quats, log_scales, opacities = _scene(config)
    axes, sharpnesses = make_sg_atoms(config.num_atoms)
    axes, sharpnesses = axes.to(torch.float64), sharpnesses.to(torch.float64)
    transport = fit_lobe_transport(
        normals,
        albedo,
        axes,
        sharpnesses,
        specular=config.specular,
        shininess=config.shininess,
    )

    viewmats = _cameras(config)
    lights = _lights(config, viewmats)
    intrinsics = pinhole_intrinsics(config.width, config.height, config.fov_degrees)
    intensity = torch.full((3,), float(config.flash_intensity), dtype=torch.float64)

    # In bracket mode the light is a function of the camera, so there is one
    # shot per view rather than a grid. Saying so here is what keeps the frame
    # count honest instead of silently rendering the same pair many times.
    if config.flash_mode == "bracket":
        pairs = [(v, v) for v in range(config.num_views)]
    else:
        pairs = [
            (v, l) for v in range(config.num_views) for l in range(config.num_lights)
        ]

    generator = torch.Generator().manual_seed(config.seed + 3)
    frames: List[Dict[str, Any]] = []
    for shot, (view_index, light_index) in enumerate(pairs):
        viewmat = viewmats[view_index]
        light_position = lights[light_index]

        directions, radiance = incident_radiance(
            means,
            light_position,
            intensity,
            reference_distance=config.reference_distance,
        )
        ell = project_point_light(directions, radiance, axes, sharpnesses)  # [N,3,B]
        colors = contract(transport, ell).clamp_min(0.0)  # [N, 3]

        exposure = 1.0
        if config.exposure_jitter > 0.0:
            exposure = 1.0 + config.exposure_jitter * (
                float(torch.rand(1, generator=generator, dtype=torch.float64)) - 0.5
            )

        record: Dict[str, Any] = {
            "file_path": f"images/{shot:04d}.png",
            "transform_matrix": viewmat_to_nerf(viewmat).tolist(),
            "atlas": {
                "view_index": view_index,
                "light_index": light_index,
                "light_position": light_position.tolist(),
                "light_intensity": intensity.tolist(),
                "exposure": exposure,
                "reference_distance": config.reference_distance,
            },
        }

        if write_images:
            image, alpha = render_reference(
                means,
                quats,
                log_scales,
                opacities,
                colors,
                viewmat,
                intrinsics,
                config.width,
                config.height,
            )
            image = image * exposure + config.ambient
            if config.read_noise > 0.0:
                image = image + config.read_noise * torch.randn(
                    image.shape, generator=generator, dtype=torch.float64
                )
            write_png(
                root / record["file_path"], (image / config.scale).clamp(0.0, 1.0)
            )
            if write_masks:
                record["atlas"]["mask_path"] = f"masks/{shot:04d}.png"
                write_png(root / record["atlas"]["mask_path"], alpha.clamp(0.0, 1.0))

        frames.append(record)

    manifest = {
        "camera_model": "PINHOLE",
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "w": config.width,
        "h": config.height,
        "atlas": {
            "format_version": 1,
            "synthetic": True,
            "colour_space": "linear",
            "scale": config.scale,
            "ambient": config.ambient,
            "flash_mode": config.flash_mode,
            # Recorded, not inferred later: in bracket mode the light is a
            # function of the camera, so holding out a light holds out its view
            # and the two reported numbers are the same number.
            "splits_are_independent": config.flash_mode == "free",
            "num_views": config.num_views,
            "num_lights": config.num_lights
            if config.flash_mode == "free"
            else config.num_views,
            "config": asdict(config),
        },
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    torch.save(
        {
            "means": means,
            "quats": quats,
            "scales": log_scales,
            "opacities": opacities,
            "transport": transport,
            "atom_axes": axes,
            "atom_sharpness": sharpnesses,
            "normals": normals,
            "albedo": albedo,
            "viewmats": viewmats,
            "light_positions": lights,
            "intrinsics": intrinsics,
            "config": asdict(config),
        },
        root / "ground_truth.pt",
    )

    return GeneratedCapture(
        root=root,
        config=config,
        means=means,
        quats=quats,
        log_scales=log_scales,
        opacity_logits=opacities,
        transport=transport,
        atom_axes=axes,
        atom_sharpness=sharpnesses,
        viewmats=viewmats,
        light_positions=lights,
        intrinsics=intrinsics,
        frames=frames,
    )
