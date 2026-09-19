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

"""The one gap CPU cannot reach: does the real rasteriser respect the theorem?

Everything ATLAS claims rests on shading and alpha compositing commuting. On CPU
that is proven against ``atlas.functional.composite``, a compositor written to be
obviously correct. But the renderer does not use that compositor -- it uses
``gsplat.rasterization``, a tiled CUDA kernel with its own sort, its own
front-to-back accumulation and its own precision. If *it* does not commute, the
CPU proof is a proof about the wrong program.

The test does not need to know the rasteriser's compositing weights, which is
what makes it possible at all. It renders the same scene twice:

- **Path A** -- contract the transport to three channels, then splat.
- **Path B** -- splat the ``3B`` transport channels, then contract per pixel.

Whatever weights the kernel chose, it chose the same ones both times, so the two
images must agree. Any disagreement is the kernel failing to be linear in the
per-primitive feature, and that would invalidate the method rather than merely
this test.

These run only on the self-hosted GPU runner. See ``docs/runner-setup.md``.
"""

import pytest
import torch

pytest.importorskip("gsplat", reason="GPU tests need gsplat installed")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU tests need CUDA"
)

from gsplat import rasterization  # noqa: E402

from atlas.functional import (  # noqa: E402
    contract,
    contract_screen,
    make_sg_atoms,
    pack_transport,
    unpack_transport,
)

DEVICE = "cuda"
DTYPE = torch.float32
# 3 * 8 = 24 channels, which is in gsplat's compiled NUM_CHANNELS list and stays
# under the channel_chunk=32 boundary where the N-D path starts chunking.
NUM_ATOMS = 8
WIDTH, HEIGHT = 64, 48


def _scene(num_primitives=256, seed=0):
    """A small scene with real overlap, so compositing actually does something."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=DTYPE).to(DEVICE)

    means = (rand(num_primitives, 3) - 0.5) * 1.5
    means[:, 2] += 4.0  # in front of the camera
    quats = torch.nn.functional.normalize(rand(num_primitives, 4) - 0.5, dim=-1)
    scales = torch.full((num_primitives, 3), 0.08, device=DEVICE, dtype=DTYPE)
    # Mid-range opacities: fully opaque primitives would hide the compositing
    # this test exists to exercise.
    opacities = 0.2 + 0.5 * rand(num_primitives)
    transport = rand(num_primitives, 3, NUM_ATOMS)

    viewmats = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    focal = 0.8 * WIDTH
    Ks = torch.tensor(
        [[[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return means, quats, scales, opacities, transport, viewmats, Ks


def _render(colors, means, quats, scales, opacities, viewmats, Ks):
    rendered, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=WIDTH,
        height=HEIGHT,
        sh_degree=None,
    )
    return rendered


def _path_a(scene, ell):
    means, quats, scales, opacities, transport, viewmats, Ks = scene
    return _render(
        contract(transport, ell), means, quats, scales, opacities, viewmats, Ks
    )


def _path_b(scene, ell):
    means, quats, scales, opacities, transport, viewmats, Ks = scene
    composited = _render(
        pack_transport(transport), means, quats, scales, opacities, viewmats, Ks
    )
    return contract_screen(unpack_transport(composited, NUM_ATOMS), ell)


def _light(seed=1):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(3, NUM_ATOMS, generator=generator, dtype=DTYPE).to(DEVICE)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_two_paths_agree_through_the_real_rasteriser(seed):
    """The gate. If this fails, the method is wrong, not the test."""
    scene = _scene(seed=seed)
    ell = _light(seed)
    a, b = _path_a(scene, ell), _path_b(scene, ell)

    assert a.shape == b.shape
    scale = float(a.abs().max())
    assert scale > 1e-3, "the scene rendered black; the comparison would be vacuous"
    difference = float((a - b).abs().max())
    # float32 accumulation over a few hundred overlapping primitives, summed in
    # two different orders. The CPU gate is 1e-5 at unit scale; allow for the
    # kernel's wider accumulation here but keep it far below the signal.
    assert difference < 1e-4 * max(scale, 1.0), (
        f"Path A and Path B disagree by {difference:.3e} at scale {scale:.3f}. "
        f"The rasteriser is not linear in the per-primitive feature."
    )


def test_the_scene_is_not_trivially_opaque():
    """A scene with no overlap would pass the parity test for the wrong reason."""
    scene = _scene()
    _, _, _, _, _, viewmats, Ks = scene
    means, quats, scales, opacities = scene[0], scene[1], scene[2], scene[3]
    _, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=torch.ones(means.shape[0], 3, device=DEVICE, dtype=DTYPE),
        viewmats=viewmats,
        Ks=Ks,
        width=WIDTH,
        height=HEIGHT,
        sh_degree=None,
    )
    covered = float((alphas > 0.05).float().mean())
    partial = float(((alphas > 0.05) & (alphas < 0.95)).float().mean())
    assert covered > 0.15, f"only {covered:.1%} of the frame is covered"
    assert partial > 0.02, "no partially transparent pixels; compositing is untested"


def test_rendering_is_linear_in_the_light():
    """Homogeneity, through the kernel rather than through the reference."""
    scene = _scene(seed=3)
    ell = _light(3)
    once = _path_a(scene, ell)
    thrice = _path_a(scene, 3.0 * ell)
    assert float((thrice - 3.0 * once).abs().max()) < 1e-4 * max(
        float(once.abs().max()), 1.0
    )


def test_superposition_holds_through_the_kernel():
    """render(E1 + E2) == render(E1) + render(E2), on real CUDA.

    Structural, not penalised. A conditioned decoder has to be taught this with
    a loss term and keeps a measurable error afterwards; here the only residue
    is the kernel's accumulation order.
    """
    scene = _scene(seed=4)
    first, second = _light(10), _light(11)
    combined = _path_a(scene, first + second)
    separate = _path_a(scene, first) + _path_a(scene, second)
    scale = max(float(combined.abs().max()), 1.0)
    assert float((combined - separate).abs().max()) < 1e-4 * scale


def test_path_b_channel_count_is_supported_by_the_build():
    """3B channels must be a shape this gsplat build actually rasterises.

    gsplat compiles a fixed list of channel counts. If 3*B is not among them the
    N-D path either chunks (slow) or refuses, and that is a build decision worth
    surfacing as a named failure rather than as a confusing kernel error.
    """
    scene = _scene(seed=5)
    packed = pack_transport(scene[4])
    assert packed.shape[-1] == 3 * NUM_ATOMS
    rendered = _render(
        packed, scene[0], scene[1], scene[2], scene[3], scene[5], scene[6]
    )
    assert rendered.shape[-1] == 3 * NUM_ATOMS
    assert torch.isfinite(rendered).all()


def test_a_loaded_model_renders_something(tmp_path):
    """End to end through RelightSplats, the class the trainer will use."""
    from atlas.model import RelightSplats

    means, quats, scales, opacities, transport, viewmats, Ks = _scene(seed=6)
    axes, sharpness = make_sg_atoms(NUM_ATOMS, device=DEVICE, dtype=DTYPE)
    model = RelightSplats(
        means=means,
        quats=quats,
        # RelightSplats stores log-scales and logit-opacities; the rasteriser's
        # activations are applied inside render().
        scales=torch.log(scales),
        opacities=torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
        transport=transport,
        atom_axes=axes,
        atom_sharpness=sharpness,
    )
    rendered, alphas, _ = model.render(viewmats, Ks, WIDTH, HEIGHT, _light(6))
    assert rendered.shape[-3:] == (HEIGHT, WIDTH, 3)
    assert torch.isfinite(rendered).all()
    assert float(rendered.abs().max()) > 1e-3

    # And it survives a round trip, which is how the trainer will hand it on.
    path = tmp_path / "model.pt"
    model.save(path)
    reloaded = RelightSplats.load(path, device=DEVICE)
    again, _, _ = reloaded.render(viewmats, Ks, WIDTH, HEIGHT, _light(6))
    assert float((again - rendered).abs().max()) == 0.0


# --- densification carries the transport ------------------------------------


def test_gsplat_densification_carries_the_transport_through_a_split():
    """The reuse the design counts on, checked against the real gsplat code.

    ``tests/test_model.py`` checks our side of the contract -- that every entry
    in the parameter dict is indexed by primitive -- by applying the same rule
    by hand. This runs gsplat's actual ``ops.split`` and confirms it does what
    we assumed: no code of ours is involved in carrying ``[N, 3, B]`` through,
    and none should need to be.
    """
    import pytest
    import torch

    ops = pytest.importorskip("gsplat.strategy.ops")

    from atlas.functional.atoms import make_sg_atoms
    from atlas.model import PRIMITIVE_PARAMETERS, RelightSplats

    num, atoms = 12, 9
    axes, sharpnesses = make_sg_atoms(atoms)
    model = RelightSplats(
        means=torch.randn(num, 3),
        quats=torch.nn.functional.normalize(torch.randn(num, 4), dim=-1),
        scales=torch.full((num, 3), -2.0),
        opacities=torch.zeros(num),
        transport=torch.randn(num, 3, atoms),
        atom_axes=axes,
        atom_sharpness=sharpnesses,
    ).requires_grad_(True)

    params = model.as_parameter_dict()
    optimizers = {
        name: torch.optim.Adam([{"params": [params[name]], "lr": 1e-3, "name": name}])
        for name in PRIMITIVE_PARAMETERS
    }
    mask = torch.zeros(num, dtype=torch.bool)
    mask[torch.tensor([2, 5, 7])] = True
    kept = torch.tensor([i for i in range(num) if not mask[i]])
    before = params["transport"].detach().clone()

    ops.split(params=params, optimizers=optimizers, state={}, mask=mask)

    # Nine survivors plus two copies of each of the three split primitives.
    assert params["transport"].shape == (num - 3 + 6, 3, atoms)
    assert torch.allclose(params["transport"][: num - 3], before[kept])

    rebuilt = RelightSplats.from_parameter_dict(
        params, model.atom_axes, model.atom_sharpness
    )
    assert rebuilt.num_primitives == 15
    assert rebuilt.num_atoms == atoms  # the basis did not grow with the scene
