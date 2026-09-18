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

"""PNG in and out, and a bitmap font, using nothing but the standard library.

This exists so that :mod:`atlas.eval` can write a comparison sheet on a
workstation that has just been handed the repo, before anyone has installed
anything optional. A comparison sheet is the artifact a person actually looks
at when a number moves, and making it depend on Pillow would mean it is missing
exactly when it is most wanted -- on the GPU runner, in CI, in a container
someone built in a hurry.

PNG is a small enough format to write honestly: a signature, three chunks, and
zlib. The only part with any subtlety is scanline filtering, and encoding it is
easy because every filter is a *shift and subtract*; only decoding is
sequential.

**Decoding is slow on purpose.** Reconstructing the Sub, Average and Paeth
filters genuinely depends on the pixel to the left, so it is a Python loop.
:func:`read_png` is for tests and for small reference images. Nothing in a
training loop should call it.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "write_png",
    "read_png",
    "to_uint8",
    "draw_text",
    "text_size",
    "GLYPH_WIDTH",
    "GLYPH_HEIGHT",
]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --- conversion -------------------------------------------------------------


def to_uint8(image: Tensor) -> Tensor:
    """Convert a float image in ``[0, 1]`` to ``uint8``, or pass one through.

    Rounds rather than truncates. Truncation biases every channel down by half
    a level, which is invisible in one image and a systematic difference when
    two sheets are diffed.
    """
    if image.dtype == torch.uint8:
        return image
    if not image.is_floating_point():
        raise ValueError(f"expected a float or uint8 image, got {image.dtype}")
    return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _as_hwc(image: Tensor) -> Tensor:
    if image.ndim == 2:
        return image.unsqueeze(-1)
    if image.ndim == 3:
        return image
    raise ValueError(f"image must be [H, W] or [H, W, C], got {tuple(image.shape)}")


# --- writing ----------------------------------------------------------------


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _paeth(left: Tensor, up: Tensor, upleft: Tensor) -> Tensor:
    """The PNG Paeth predictor, vectorised over a scanline."""
    estimate = left + up - upleft
    da = (estimate - left).abs()
    db = (estimate - up).abs()
    dc = (estimate - upleft).abs()
    pick_left = (da <= db) & (da <= dc)
    pick_up = (~pick_left) & (db <= dc)
    return torch.where(pick_left, left, torch.where(pick_up, up, upleft))


def _shift_right(row: Tensor, offset: int) -> Tensor:
    """``row`` moved ``offset`` bytes to the right, zero-filled at the start."""
    shifted = torch.zeros_like(row)
    if offset < row.numel():
        shifted[offset:] = row[:-offset] if offset else row
    return shifted


def _to_bytes(row: Tensor) -> bytes:
    """A ``uint8`` row as bytes, without requiring numpy.

    ``Tensor.numpy().tobytes()`` is the fast path and is what runs everywhere
    torch was installed the usual way, but torch does not itself require numpy
    and neither does this package.
    """
    try:
        return row.numpy().tobytes()
    except (ImportError, RuntimeError):  # pragma: no cover - numpy is usual
        return bytes(row.tolist())


def _filter_scanlines(raw: Tensor, bpp: int) -> bytes:
    """Encode every scanline with the filter that compresses it best.

    The heuristic is the one in the PNG specification: pick the filter whose
    output has the smallest sum of absolute *signed* byte values, on the theory
    that bytes near zero are what zlib is good at. Choosing per row rather than
    once for the image typically halves a comparison sheet.
    """
    height = raw.shape[0]
    previous = torch.zeros(raw.shape[1], dtype=torch.int16)
    out = bytearray()
    for y in range(height):
        row = raw[y].to(torch.int16)
        left = _shift_right(row, bpp)
        upleft = _shift_right(previous, bpp)
        candidates = (
            row,
            row - left,
            row - previous,
            row - torch.div(left + previous, 2, rounding_mode="floor"),
            row - _paeth(left, previous, upleft),
        )
        best_kind, best_row, best_cost = 0, None, None
        for kind, candidate in enumerate(candidates):
            wrapped = candidate % 256
            signed = torch.where(wrapped > 127, wrapped - 256, wrapped)
            cost = int(signed.abs().sum())
            if best_cost is None or cost < best_cost:
                best_kind, best_row, best_cost = kind, wrapped, cost
        out.append(best_kind)
        out.extend(_to_bytes(best_row.to(torch.uint8)))
        previous = row
    return bytes(out)


def write_png(path: Path | str, image: Tensor) -> Path:
    """Write ``[H, W]``, ``[H, W, 1]``, ``[H, W, 3]`` or ``[H, W, 4]`` as a PNG.

    Args:
        path: Destination. Parent directories are created.
        image: ``uint8``, or floating point in ``[0, 1]`` (clamped and rounded).

    Returns:
        The path written.
    """
    image = _as_hwc(image).detach().cpu()
    height, width, channels = image.shape
    if height == 0 or width == 0:
        raise ValueError(f"cannot write a {height}x{width} image")
    colour_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(channels)
    if colour_type is None:
        raise ValueError(f"expected 1, 2, 3 or 4 channels, got {channels}")

    raw = to_uint8(image).contiguous().view(height, width * channels)
    payload = zlib.compress(_filter_scanlines(raw, channels), 6)

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    blob = (
        _SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", payload)
        + _chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


# --- reading ----------------------------------------------------------------


def _unfilter(data: bytes, height: int, stride: int, bpp: int) -> bytearray:
    out = bytearray(height * stride)
    cursor = 0
    for y in range(height):
        kind = data[cursor]
        cursor += 1
        row = bytearray(data[cursor : cursor + stride])
        cursor += stride
        base = y * stride
        above = base - stride
        if kind == 0:
            pass
        elif kind == 2 and y == 0:
            pass
        elif kind == 2:
            for i in range(stride):
                row[i] = (row[i] + out[above + i]) & 0xFF
        elif kind in (1, 3, 4):
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = out[above + i] if y > 0 else 0
                upleft = out[above + i - bpp] if (y > 0 and i >= bpp) else 0
                if kind == 1:
                    row[i] = (row[i] + left) & 0xFF
                elif kind == 3:
                    row[i] = (row[i] + ((left + up) >> 1)) & 0xFF
                else:
                    estimate = left + up - upleft
                    da, db, dc = (
                        abs(estimate - left),
                        abs(estimate - up),
                        abs(estimate - upleft),
                    )
                    if da <= db and da <= dc:
                        predictor = left
                    elif db <= dc:
                        predictor = up
                    else:
                        predictor = upleft
                    row[i] = (row[i] + predictor) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {kind} on row {y}")
        out[base : base + stride] = row
    return out


def read_png(path: Path | str) -> Tensor:
    """Read an 8-bit PNG as a ``uint8`` ``[H, W, C]`` tensor.

    Supports the greyscale, greyscale+alpha, RGB and RGBA colour types at a bit
    depth of 8, which is what :func:`write_png` produces and what every tool
    that might hand us a reference image produces. Interlaced, 16-bit and
    palette images raise rather than being silently mis-decoded.
    """
    blob = Path(path).read_bytes()
    if not blob.startswith(_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")

    cursor = len(_SIGNATURE)
    header: Tuple[int, ...] = ()
    compressed = bytearray()
    while cursor + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[cursor : cursor + 4])
        kind = blob[cursor + 4 : cursor + 8]
        payload = blob[cursor + 8 : cursor + 8 + length]
        (stored_crc,) = struct.unpack(
            ">I", blob[cursor + 8 + length : cursor + 12 + length]
        )
        if stored_crc != zlib.crc32(kind + payload):
            raise ValueError(
                f"{kind.decode('ascii', 'replace')} chunk CRC mismatch in {path}; "
                f"the file is corrupt, and decoding it anyway would hand back "
                f"plausible-looking wrong pixels"
            )
        cursor += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break

    if not header:
        raise ValueError(f"PNG has no IHDR chunk: {path}")
    width, height, depth, colour_type, compression, filtering, interlace = header
    if depth != 8:
        raise ValueError(f"only 8-bit PNGs are supported, got {depth}-bit")
    if interlace:
        raise ValueError("interlaced PNGs are not supported")
    if compression or filtering:
        raise ValueError("unsupported PNG compression or filter method")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(colour_type)
    if channels is None:
        raise ValueError(f"unsupported PNG colour type {colour_type}")

    stride = width * channels
    raw = _unfilter(zlib.decompress(bytes(compressed)), height, stride, channels)
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return flat.view(height, width, channels).clone()


# --- a font, because an unlabelled comparison sheet is a puzzle -------------

GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7

# Each glyph is seven rows of five bits, most significant bit leftmost. Small
# and ugly, but it means a sheet says which light and which view it is showing
# without a font file, a font package, or a guess about what is installed on
# the machine that renders it.
_FONT = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    "/": (0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "#": (0x0A, 0x1F, 0x0A, 0x0A, 0x0A, 0x1F, 0x0A),
    "%": (0x11, 0x12, 0x02, 0x04, 0x08, 0x09, 0x11),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "*": (0x00, 0x15, 0x0E, 0x1F, 0x0E, 0x15, 0x00),
}
# Anything not in the table draws as a hollow box, which is legible as "this
# character was dropped" rather than as a space, which is not.
_MISSING = (0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F)


def text_size(text: str, *, scale: int = 1, tracking: int = 1) -> Tuple[int, int]:
    """The ``(width, height)`` in pixels that :func:`draw_text` would cover."""
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")
    if not text:
        return (0, GLYPH_HEIGHT * scale)
    advance = (GLYPH_WIDTH + tracking) * scale
    return (advance * len(text) - tracking * scale, GLYPH_HEIGHT * scale)


def draw_text(
    canvas: Tensor,
    text: str,
    origin: Tuple[int, int],
    *,
    colour: Sequence[float] | float = 1.0,
    scale: int = 1,
    tracking: int = 1,
) -> Tensor:
    """Draw ``text`` into ``canvas`` in place, top-left at ``origin``.

    Lower case is folded to upper case; the font has one case. Drawing is
    clipped at every edge, so a label longer than the canvas is truncated
    rather than raising -- a sheet with a clipped caption is still useful and a
    sheet that failed to render is not.

    Args:
        canvas: ``[H, W, C]`` float image, modified in place.
        text: What to draw.
        origin: ``(y, x)`` of the top-left corner, in pixels.
        colour: A scalar or a per-channel sequence.
        scale: Integer pixel size of one font pixel.
        tracking: Blank font-pixel columns between glyphs.

    Returns:
        ``canvas``, for chaining.
    """
    if canvas.ndim != 3:
        raise ValueError(f"canvas must be [H, W, C], got {tuple(canvas.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")
    height, width, channels = canvas.shape
    ink = torch.as_tensor(colour, dtype=canvas.dtype, device=canvas.device)
    if ink.ndim == 0:
        ink = ink.expand(channels)
    if ink.shape != (channels,):
        raise ValueError(f"colour must be a scalar or {channels} values")

    top, left = origin
    for index, character in enumerate(text.upper()):
        glyph = _FONT.get(character, _MISSING)
        glyph_left = left + index * (GLYPH_WIDTH + tracking) * scale
        for row, bits in enumerate(glyph):
            y0 = top + row * scale
            if y0 + scale <= 0 or y0 >= height:
                continue
            for column in range(GLYPH_WIDTH):
                if not bits & (1 << (GLYPH_WIDTH - 1 - column)):
                    continue
                x0 = glyph_left + column * scale
                if x0 + scale <= 0 or x0 >= width:
                    continue
                canvas[max(y0, 0) : y0 + scale, max(x0, 0) : x0 + scale] = ink
    return canvas
