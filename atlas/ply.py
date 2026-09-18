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

"""Reading the Gaussian PLY that a fixed-light reconstruction leaves behind.

Geometry initialisation is the single biggest accelerator available for the
first real run: an existing gsplat reconstruction of the same object already
knows where the surface is, so training starts from a shape instead of from a
sparse point cloud.

Hand-rolled rather than adding `plyfile`. The format needed here is narrow --
one `vertex` element, scalar properties, ascii or little-endian binary -- and
`AGENTS.md` forbids new required dependencies. Parsing it is forty lines;
carrying a dependency into every install is forever.

The layout written by `gsplat.export_splats` is::

    x y z  nx ny nz  f_dc_0..2  f_rest_0..(3K-1)  opacity  scale_0..2  rot_0..3

with opacity stored pre-sigmoid and scales pre-exponential, which is what the
rasteriser's activations expect back.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = ["PlyData", "read_ply", "SH_C0"]

# Zeroth-order spherical-harmonic coefficient, so that
# rgb = SH_C0 * f_dc + 0.5 recovers the colour gsplat encoded.
SH_C0 = 0.28209479177387814

_BINARY_FORMATS = {
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
}


class PlyData:
    """A parsed PLY: named columns, each a list of floats, all the same length."""

    def __init__(self, columns: Dict[str, List[float]], count: int):
        self.columns = columns
        self.count = count

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> List[float]:
        return self.columns[name]

    def names(self) -> List[str]:
        return list(self.columns)

    def prefixed(self, prefix: str) -> List[str]:
        """Column names starting with ``prefix``, in numeric order of the suffix.

        `f_rest_10` must sort after `f_rest_9`, which lexicographic order gets
        wrong and which would silently permute the spherical-harmonic
        coefficients.
        """

        def key(name: str) -> Tuple[int, str]:
            tail = name[len(prefix) :]
            return (int(tail), "") if tail.isdigit() else (1 << 30, tail)

        return sorted((n for n in self.columns if n.startswith(prefix)), key=key)


def _parse_header(handle) -> Tuple[str, int, List[Tuple[str, str]]]:
    magic = handle.readline().strip()
    if magic != b"ply":
        raise ValueError("not a PLY file: missing the 'ply' magic line")
    fmt = None
    count = None
    properties: List[Tuple[str, str]] = []
    in_vertex = False
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PLY header ended without 'end_header'")
        parts = line.strip().split()
        if not parts:
            continue
        keyword = parts[0].decode()
        if keyword == "format":
            fmt = parts[1].decode()
        elif keyword == "element":
            name = parts[1].decode()
            in_vertex = name == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif keyword == "property" and in_vertex:
            if parts[1].decode() == "list":
                raise ValueError(
                    "list properties are not supported; this reader expects the "
                    "flat vertex layout gsplat writes"
                )
            properties.append((parts[1].decode(), parts[2].decode()))
        elif keyword == "end_header":
            break
    if fmt is None or count is None:
        raise ValueError("PLY header has no format or no vertex element")
    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(
            f"unsupported PLY format {fmt!r}; expected ascii or binary_little_endian"
        )
    return fmt, count, properties


def read_ply(path: Path | str) -> PlyData:
    """Read the vertex element of a PLY into named float columns.

    Args:
        path: The file to read.

    Returns:
        A :class:`PlyData`.

    Raises:
        ValueError: On a header this reader does not cover, naming what it found.
    """
    path = Path(path)
    with path.open("rb") as handle:
        fmt, count, properties = _parse_header(handle)
        names = [name for _, name in properties]
        columns: Dict[str, List[float]] = {name: [] for name in names}

        if fmt == "ascii":
            for _ in range(count):
                line = handle.readline()
                if not line:
                    raise ValueError(
                        f"{path.name}: header promised {count} vertices, file ended early"
                    )
                values = line.split()
                if len(values) < len(names):
                    raise ValueError(
                        f"{path.name}: a vertex row has {len(values)} values, "
                        f"expected {len(names)}"
                    )
                for name, value in zip(names, values):
                    columns[name].append(float(value))
        else:
            codes = []
            size = 0
            for type_name, _ in properties:
                if type_name not in _BINARY_FORMATS:
                    raise ValueError(
                        f"{path.name}: unsupported property type {type_name!r}"
                    )
                code, width = _BINARY_FORMATS[type_name]
                codes.append(code)
                size += width
            layout = struct.Struct("<" + "".join(codes))
            payload = handle.read(size * count)
            if len(payload) < size * count:
                raise ValueError(
                    f"{path.name}: header promised {count} vertices "
                    f"({size * count} bytes), found {len(payload)}"
                )
            for offset in range(0, size * count, size):
                for name, value in zip(names, layout.unpack_from(payload, offset)):
                    columns[name].append(float(value))

    return PlyData(columns, count)
