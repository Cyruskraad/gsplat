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
