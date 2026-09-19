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

"""The model, everything about it that does not need a GPU.

The rasteriser call itself is CUDA-only and cannot be covered here; that gap is
closed by the ``--smoke`` run on the workstation. What *can* be covered is
everything that decides whether that run means anything: the PLY reader, the
transport initialiser, the round trip, and the shape contracts.
"""

import math
import struct

import pytest
import torch

from atlas.functional import make_sg_atoms, project_environment
from atlas.model import DEFAULT_NUM_ATOMS, RelightSplats
from atlas.ply import SH_C0, read_ply

DTYPE = torch.float64

GSPLAT_PROPERTIES = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def _write_gsplat_ply(path, rows, binary=True, extra_rest=0):
    """A PLY in exactly the layout gsplat.export_splats writes."""
    names = list(GSPLAT_PROPERTIES)
    if extra_rest:
        insert_at = names.index("opacity")
        names[insert_at:insert_at] = [f"f_rest_{i}" for i in range(extra_rest)]
    header = ["ply"]
    header.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    header.append(f"element vertex {len(rows)}")
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    blob = ("\n".join(header) + "\n").encode()
    if binary:
        packer = struct.Struct("<" + "f" * len(names))
        for row in rows:
            blob += packer.pack(*[row.get(name, 0.0) for name in names])
    else:
        for row in rows:
            blob += (
                " ".join(str(float(row.get(name, 0.0))) for name in names) + "\n"
            ).encode()
    path.write_bytes(blob)
    return path


def _row(x, y, z, rgb=(0.4, 0.6, 0.8), opacity=2.0, scale=-3.0):
    row = {"x": x, "y": y, "z": z, "opacity": opacity}
    for index, value in enumerate(rgb):
        row[f"f_dc_{index}"] = (value - 0.5) / SH_C0
    for index in range(3):
        row[f"scale_{index}"] = scale
    row["rot_0"] = 1.0
    return row


# --- the PLY reader ----------------------------------------------------------


def test_binary_and_ascii_ply_read_identically(tmp_path):
    rows = [_row(0.0, 0.0, 0.0), _row(1.0, 2.0, 3.0, rgb=(0.1, 0.2, 0.3))]
    binary = read_ply(_write_gsplat_ply(tmp_path / "b.ply", rows, binary=True))
    ascii_ = read_ply(_write_gsplat_ply(tmp_path / "a.ply", rows, binary=False))
    assert binary.count == ascii_.count == 2
    for name in GSPLAT_PROPERTIES:
        assert binary[name] == pytest.approx(ascii_[name], abs=1e-6)


def test_prefixed_columns_sort_numerically_not_lexicographically(tmp_path):
    """`f_rest_10` after `f_rest_9`, or the SH coefficients quietly permute."""
    rows = [_row(0.0, 0.0, 0.0)]
    data = read_ply(_write_gsplat_ply(tmp_path / "p.ply", rows, extra_rest=12))
    names = data.prefixed("f_rest_")
    assert names == [f"f_rest_{i}" for i in range(12)]


def test_a_truncated_ply_is_reported_not_silently_short(tmp_path):
    path = _write_gsplat_ply(tmp_path / "t.ply", [_row(0, 0, 0), _row(1, 1, 1)])
    blob = path.read_bytes()
    path.write_bytes(blob[:-20])
    with pytest.raises(ValueError, match="found"):
        read_ply(path)


def test_a_non_ply_file_is_rejected(tmp_path):
    path = tmp_path / "nope.ply"
    path.write_bytes(b"this is not a ply\n")
    with pytest.raises(ValueError, match="not a PLY"):
        read_ply(path)


def test_an_unsupported_format_names_what_it_found(tmp_path):
    path = tmp_path / "big.ply"
    path.write_bytes(
        b"ply\nformat binary_big_endian 1.0\nelement vertex 0\nend_header\n"
    )
    with pytest.raises(ValueError, match="binary_big_endian"):
        read_ply(path)


# --- initialisation from a fixed-light reconstruction ------------------------


def test_from_ply_carries_the_geometry_across(tmp_path):
    rows = [_row(0.0, 1.0, 2.0, scale=-2.5), _row(3.0, 4.0, 5.0, scale=-1.5)]
    path = _write_gsplat_ply(tmp_path / "g.ply", rows)
    model = RelightSplats.from_ply(path, num_atoms=8, dtype=DTYPE)

    assert model.num_primitives == 2
    assert model.num_atoms == 8
    assert torch.allclose(
        model.means, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=DTYPE)
    )
    assert torch.allclose(model.scales[0], torch.full((3,), -2.5, dtype=DTYPE))
    assert torch.allclose(
        model.quats[0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    )
    assert torch.allclose(model.opacities, torch.tensor([2.0, 2.0], dtype=DTYPE))


@pytest.mark.parametrize("num_atoms", [4, 16, DEFAULT_NUM_ATOMS])
def test_the_initial_transport_reproduces_the_fixed_light_albedo(tmp_path, num_atoms):
    """The claim the initialiser makes, checked against the real projection.

    Under a uniform white environment the model must render the colour the
    fixed-light reconstruction had. The initialiser uses a closed form for an
    atom's integral; this drives it through ``project_environment``'s numerical
    quadrature instead, so the two cannot drift apart unnoticed.
    """
    albedo = (0.4, 0.6, 0.8)
    path = _write_gsplat_ply(tmp_path / "g.ply", [_row(0.0, 0.0, 0.0, rgb=albedo)])
    model = RelightSplats.from_ply(path, num_atoms=num_atoms, dtype=DTYPE)

    envmap = torch.ones(64, 128, 3, dtype=DTYPE)
    ell = model.project_environment(envmap)
    rendered = (model.transport * ell.unsqueeze(0)).sum(dim=-1)[0]

    assert torch.allclose(rendered, torch.tensor(albedo, dtype=DTYPE), rtol=0.02)


def test_a_ply_without_colour_starts_grey_rather_than_refusing(tmp_path):
    names = [n for n in GSPLAT_PROPERTIES if not n.startswith("f_dc_")]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 1",
        *[f"property float {n}" for n in names],
        "end_header",
    ]
    row = {"x": 0.0, "y": 0.0, "z": 0.0, "opacity": 1.0, "rot_0": 1.0}
    for i in range(3):
        row[f"scale_{i}"] = -3.0
    path = tmp_path / "nocolour.ply"
    path.write_bytes(
        (
            "\n".join(header)
            + "\n"
            + " ".join(str(row.get(n, 0.0)) for n in names)
            + "\n"
        ).encode()
    )
    model = RelightSplats.from_ply(path, num_atoms=8, dtype=DTYPE)
    envmap = torch.ones(32, 64, 3, dtype=DTYPE)
    rendered = (model.transport * model.project_environment(envmap).unsqueeze(0)).sum(
        -1
    )
    assert torch.allclose(rendered[0], torch.full((3,), 0.5, dtype=DTYPE), rtol=0.05)


def test_a_ply_missing_the_gsplat_layout_says_so(tmp_path):
    path = tmp_path / "wrong.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n"
    )
    with pytest.raises(ValueError, match="missing"):
        RelightSplats.from_ply(path, num_atoms=4)


# --- contracts ---------------------------------------------------------------


def _model(num=5, num_atoms=6):
    axes, sharp = make_sg_atoms(num_atoms, dtype=DTYPE)
    return RelightSplats(
        means=torch.zeros(num, 3, dtype=DTYPE),
        quats=torch.zeros(num, 4, dtype=DTYPE),
        scales=torch.zeros(num, 3, dtype=DTYPE),
        opacities=torch.zeros(num, dtype=DTYPE),
        transport=torch.rand(num, 3, num_atoms, dtype=DTYPE),
        atom_axes=axes,
        atom_sharpness=sharp,
    )


def test_the_basis_travels_with_the_model(tmp_path):
    """Coefficients without their basis are meaningless, so they ship together."""
    model = _model()
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = RelightSplats.load(path)
    assert torch.equal(loaded.transport, model.transport)
    assert torch.equal(loaded.atom_axes, model.atom_axes)
    assert torch.equal(loaded.atom_sharpness, model.atom_sharpness)


def test_loading_something_that_is_not_an_atlas_checkpoint_is_refused(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="not an ATLAS checkpoint"):
        RelightSplats.load(path)


def test_parameter_bytes_surfaces_the_transport_as_the_dominant_cost():
    """The memory surprise this project is most likely to hit, reported early."""
    model = _model(num=1000, num_atoms=DEFAULT_NUM_ATOMS)
    sizes = model.parameter_bytes()
    assert sizes["transport"] > sum(v for k, v in sizes.items() if k != "transport")


def test_a_transport_whose_atom_count_disagrees_with_the_basis_is_refused():
    axes, sharp = make_sg_atoms(6, dtype=DTYPE)
    with pytest.raises(ValueError, match="atom_axes must be"):
        RelightSplats(
            means=torch.zeros(2, 3, dtype=DTYPE),
            quats=torch.zeros(2, 4, dtype=DTYPE),
            scales=torch.zeros(2, 3, dtype=DTYPE),
            opacities=torch.zeros(2, dtype=DTYPE),
            transport=torch.zeros(2, 3, 9, dtype=DTYPE),
            atom_axes=axes,
            atom_sharpness=sharp,
        )


@pytest.mark.parametrize(
    "field,shape,message",
    [
        ("quats", (5, 3), "quats must be"),
        ("scales", (5, 2), "scales must be"),
        ("opacities", (5, 1), "opacities must be"),
        ("transport", (5, 2, 6), "transport must be"),
    ],
)
def test_every_shape_contract_is_enforced(field, shape, message):
    axes, sharp = make_sg_atoms(6, dtype=DTYPE)
    kwargs = dict(
        means=torch.zeros(5, 3, dtype=DTYPE),
        quats=torch.zeros(5, 4, dtype=DTYPE),
        scales=torch.zeros(5, 3, dtype=DTYPE),
        opacities=torch.zeros(5, dtype=DTYPE),
        transport=torch.zeros(5, 3, 6, dtype=DTYPE),
        atom_axes=axes,
        atom_sharpness=sharp,
    )
    kwargs[field] = torch.zeros(*shape, dtype=DTYPE)
    with pytest.raises(ValueError, match=message):
        RelightSplats(**kwargs)


def test_render_without_gsplat_names_the_extra_to_install():
    """The error a laptop must give: actionable, not a bare ModuleNotFoundError."""
    try:
        import gsplat  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("gsplat is installed, so the guidance path cannot be exercised")

    model = _model()
    with pytest.raises(ImportError, match=r"\[gpu\]"):
        model.render(
            viewmats=torch.eye(4, dtype=DTYPE).unsqueeze(0),
            Ks=torch.eye(3, dtype=DTYPE).unsqueeze(0),
            width=8,
            height=8,
            ell=torch.ones(3, model.num_atoms, dtype=DTYPE),
        )


def test_to_and_requires_grad_cover_every_parameter():
    model = _model().requires_grad_(True)
    for name in ("means", "quats", "scales", "opacities", "transport"):
        assert getattr(model, name).requires_grad, name
    moved = model.to(torch.float32)
    assert moved.transport.dtype == torch.float32
    assert moved.atom_axes.dtype == torch.float32


# --- the densification view, and the learned basis --------------------------


def _splats(num=8, atoms=6, seed=0):
    from atlas.functional.atoms import make_sg_atoms

    generator = torch.Generator().manual_seed(seed)
    axes, sharpnesses = make_sg_atoms(atoms)
    return RelightSplats(
        means=torch.randn(num, 3, generator=generator),
        quats=torch.randn(num, 4, generator=generator),
        scales=torch.randn(num, 3, generator=generator),
        opacities=torch.randn(num, generator=generator),
        transport=torch.randn(num, 3, atoms, generator=generator),
        atom_axes=axes,
        atom_sharpness=sharpnesses,
    )


def test_every_entry_in_the_parameter_dict_is_indexed_by_primitive():
    """The contract that makes gsplat's generic densification path correct.

    ``gsplat.strategy.ops`` names only ``means``, ``scales`` and ``opacities``;
    everything else it splits as ``p[sel].repeat([2] + [1] * (p.dim() - 1))``
    and concatenates along dimension zero. That is right for any tensor whose
    first dimension is the primitive, and wrong for any tensor whose is not.
    """
    from atlas.model import PRIMITIVE_PARAMETERS

    model = _splats(num=11, atoms=5)
    params = model.as_parameter_dict()
    assert set(params.keys()) == set(PRIMITIVE_PARAMETERS)
    for name, tensor in params.items():
        assert tensor.shape[0] == 11, (name, tuple(tensor.shape))


def test_the_atoms_are_kept_out_of_the_densification_view():
    """They are ``[B, ...]``. Concatenating them along dimension zero would
    grow the basis every time a primitive split, silently."""
    from atlas.model import ATOM_PARAMETERS

    params = _splats().as_parameter_dict()
    for name in ATOM_PARAMETERS:
        assert name not in params


def test_the_parameter_dict_shares_storage_rather_than_copying():
    model = _splats()
    params = model.as_parameter_dict()
    params["transport"].data.add_(1.0)
    assert torch.allclose(model.transport, params["transport"].data)


def test_a_model_rebuilds_around_a_dict_that_densification_replaced():
    """What a training step does after a split: the dict is new tensors, the
    basis is not, and the model is a view over both."""
    model = _splats(num=6, atoms=4)
    params = model.as_parameter_dict()
    grown = torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(torch.cat([p, p], dim=0))
            for name, p in params.items()
        }
    )
    rebuilt = RelightSplats.from_parameter_dict(
        grown, model.atom_axes, model.atom_sharpness
    )
    assert rebuilt.num_primitives == 12
    assert rebuilt.num_atoms == 4
    assert rebuilt.transport.shape == (12, 3, 4)


def test_rebuilding_from_an_incomplete_dict_names_what_is_missing():
    model = _splats()
    params = model.as_parameter_dict()
    del params["transport"]
    with pytest.raises(ValueError, match=r"missing \['transport'\]"):
        RelightSplats.from_parameter_dict(params, model.atom_axes, model.atom_sharpness)


def test_the_transport_survives_the_generic_split_rule_with_its_channels_intact():
    """Applied here exactly as ``gsplat.strategy.ops.split`` applies it, so the
    reuse claim is checked without needing CUDA to check it."""
    model = _splats(num=10, atoms=7)
    transport = model.as_parameter_dict()["transport"]
    selected = torch.tensor([1, 4, 9])
    repeats = [2] + [1] * (transport.dim() - 1)
    split = transport[selected].repeat(repeats)
    assert split.shape == (6, 3, 7)
    assert torch.equal(split[:3], transport[selected])
    assert torch.equal(split[3:], transport[selected])


# --- learned atoms ----------------------------------------------------------


def test_the_basis_is_held_out_of_the_gradient_by_default():
    """A first run needs a fixed-basis baseline to attribute a later gain to.
    Learning the basis is a change to the method, not a tuning knob."""
    model = _splats().requires_grad_(True)
    assert model.transport.requires_grad is True
    assert model.atom_axes.requires_grad is False
    assert model.atom_sharpness.requires_grad is False


def test_the_basis_becomes_trainable_when_asked_for():
    model = _splats().requires_grad_(True, atoms=True)
    assert model.atom_axes.requires_grad and model.atom_sharpness.requires_grad


def test_a_learned_axis_is_projected_back_onto_the_unit_sphere():
    """An atom is ``exp(lambda (w . xi - 1))``. A non-unit ``xi`` rotates and
    rescales the lobe by the same number, so the two parameters stop meaning
    separate things."""
    model = _splats()
    model.atom_axes.data.mul_(3.7)
    assert float((model.atom_axes.norm(dim=-1) - 1).abs().max()) > 2.0  # premise
    model.normalise_atoms_()
    assert float((model.atom_axes.norm(dim=-1) - 1).abs().max()) < 1e-6


def test_a_learned_sharpness_is_kept_positive():
    """A negative lambda inverts the lobe into a trough that grows without
    bound away from its axis."""
    from atlas.model import MIN_SHARPNESS

    model = _splats()
    model.atom_sharpness.data.fill_(-42.0)
    model.normalise_atoms_()
    assert float(model.atom_sharpness.min()) == pytest.approx(MIN_SHARPNESS)


def test_the_projection_leaves_an_already_valid_basis_alone():
    model = _splats()
    axes, sharpnesses = model.atom_axes.clone(), model.atom_sharpness.clone()
    model.normalise_atoms_()
    assert torch.allclose(model.atom_axes, axes, atol=1e-7)
    assert torch.equal(model.atom_sharpness, sharpnesses)


def test_a_gradient_reaches_the_basis_and_the_projection_keeps_it_legal():
    """The L in ATLAS, end to end: a loss that depends on the atoms moves them,
    and the projection puts them back on the constraint set."""
    from atlas.functional.atoms import evaluate_atoms

    model = _splats(num=4, atoms=5).requires_grad_(True, atoms=True)
    directions = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    evaluate_atoms(directions, model.atom_axes, model.atom_sharpness).sum().backward()

    assert model.atom_axes.grad is not None
    assert float(model.atom_axes.grad.abs().max()) > 0.0
    before = model.atom_axes.detach().clone()
    with torch.no_grad():
        model.atom_axes.add_(0.3 * model.atom_axes.grad)
    model.normalise_atoms_()
    assert not torch.allclose(model.atom_axes, before)  # it moved
    legality = (model.atom_axes.detach().norm(dim=-1) - 1).abs().max()
    assert float(legality) < 1e-6  # and is legal
