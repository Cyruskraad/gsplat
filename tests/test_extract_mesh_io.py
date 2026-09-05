# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for examples/extract_mesh.py's texture-map writing.

A normal map that survives the trip to disk with its channels swapped loads
without complaint and shades wrong, so the round trip is pinned here rather
than left to be noticed in a DCC tool.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

pytest.importorskip("cv2", reason="opencv is not installed")
pytest.importorskip("torch")
pytest.importorskip(
    "open3d", reason="open3d is not installed (pip install gsplat[mesh])"
)

from extract_mesh import _write_map  # noqa: E402


def _read_back(path):
    import cv2

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"could not read back {path}"
    return image[:, :, ::-1] if image.ndim == 3 else image


def test_write_map_round_trips_16_bit_rgb_exactly(tmp_path):
    """Every one of the 65536 levels must survive, in the right channel.

    Pillow -- imageio's default PNG backend, and what the 8-bit path uses --
    cannot write 16-bit RGB at all, so this path goes through OpenCV, which is
    BGR. Reversing the channels is not optional: without it a normal map's X
    and Z come back swapped.
    """
    rng = np.random.default_rng(0)
    image = (rng.random((37, 53, 3)) * 65535).round().astype(np.uint16)
    # Make the channels unmistakably different, so a swap cannot pass.
    image[..., 0] = 1000
    image[..., 2] = 60000

    path = str(tmp_path / "normal.png")
    _write_map(path, image)
    back = _read_back(path)

    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, image)
    # Premise: the channels really are distinguishable, so the equality above
    # is not satisfiable by a swapped write.
    assert image[..., 0].mean() != pytest.approx(image[..., 2].mean())


def test_write_map_still_writes_8_bit_through_imageio(tmp_path):
    import imageio.v2 as imageio

    rng = np.random.default_rng(1)
    image = (rng.random((16, 16, 3)) * 255).round().astype(np.uint8)
    path = str(tmp_path / "albedo.png")
    _write_map(path, image)

    back = imageio.imread(path)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, image)


def test_write_map_handles_a_single_channel_map(tmp_path):
    """The AO map is (H, W) or (H, W, 1); the BGR reversal must not touch it."""
    rng = np.random.default_rng(2)
    image = (rng.random((16, 16)) * 65535).round().astype(np.uint16)
    path = str(tmp_path / "ao.png")
    _write_map(path, image)

    back = _read_back(path)
    np.testing.assert_array_equal(back, image)


def test_16_bit_normal_map_recovers_detail_8_bit_cannot(tmp_path):
    """The reason the option exists, measured against an analytic sphere.

    On a lightly decimated mesh the low-poly normals are already within an
    8-bit map's resolution, so the map stores quantization noise rather than
    recovered geometry. Ground truth here needs no renderer: on a unit sphere
    the true surface normal at a point *is* that point.
    """
    import open3d as o3d

    from gsplat.photogrammetry import bake_normal_map, simplify_mesh
    from gsplat.photogrammetry.texturing import _unwrap_and_rasterize

    dense = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    dense.compute_vertex_normals()
    low = simplify_mesh(dense, target_triangles=3000)
    low.compute_vertex_normals()

    errors = {}
    for bits in (8, 16):
        mesh = o3d.geometry.TriangleMesh(low)
        mesh.compute_vertex_normals()
        mesh, normal_map, stats = bake_normal_map(
            dense, mesh, texture_size=256, space="object", bits=bits
        )
        assert normal_map.dtype == (np.uint8 if bits == 8 else np.uint16)
        assert stats["bits"] == bits
        assert stats["quantization_floor"] == pytest.approx(2.0 / (2**bits - 1))

        atlas = _unwrap_and_rasterize(mesh, 256)
        decoded = (
            normal_map[atlas.rows, atlas.cols].astype(np.float64) / (2**bits - 1)
        ) * 2.0 - 1.0
        decoded /= np.clip(np.linalg.norm(decoded, axis=1, keepdims=True), 1e-12, None)
        truth = atlas.positions / np.linalg.norm(atlas.positions, axis=1, keepdims=True)
        errors[bits] = float(np.linalg.norm(decoded - truth, axis=1).mean())

    # Premise: at 8 bits the error must be dominated by *quantization*, or
    # there is nothing for more bits to recover. Uniform rounding over a step
    # of 2/255 costs about a quarter of a step per channel, ~0.0034 in L2 over
    # three channels -- which is what the 8-bit bake measures.
    assert errors[8] == pytest.approx(0.0034, abs=0.0015), errors
    # And 16 bits is materially better: measured 2.5x on this pair of meshes.
    assert errors[16] < errors[8] / 2.0, errors


def test_bake_normal_map_rejects_an_unsupported_bit_depth():
    import open3d as o3d

    from gsplat.photogrammetry import bake_normal_map

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    mesh.compute_vertex_normals()
    with pytest.raises(ValueError, match="bits must be 8 or 16"):
        bake_normal_map(mesh, mesh, texture_size=32, bits=12)
