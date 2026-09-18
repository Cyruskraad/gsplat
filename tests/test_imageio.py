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

"""The PNG codec, checked against something other than itself.

A round trip through one's own encoder and decoder proves only that the two
agree, which they would even if both were wrong in the same way. So the
decoder is also pointed at scanlines whose filter bytes are set by hand -- all
five of them, with the expected pixels worked out from the specification -- and
the encoder's output is checked against ``zlib`` and the chunk CRCs directly.
"""

import struct
import zlib

import pytest

torch = pytest.importorskip("torch")

from atlas.imageio import (  # noqa: E402
    GLYPH_HEIGHT,
    GLYPH_WIDTH,
    draw_text,
    read_png,
    text_size,
    to_uint8,
    write_png,
)

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _build_png(width, height, channels, filtered_rows, *, interlace=0, depth=8):
    """A PNG whose scanline filter bytes are whatever the caller says.

    Written here rather than with :func:`write_png` so that the decoder is
    tested against the specification and not against its own encoder, which
    picks its filters adaptively and would never emit some of these.
    """
    colour_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]

    def chunk(kind, payload):
        body = kind + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, depth, colour_type, 0, 0, interlace)
    data = b"".join(bytes([k]) + bytes(row) for k, row in filtered_rows)
    return (
        _SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


# --- round trips ------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_a_uint8_image_round_trips_exactly(tmp_path, channels):
    generator = torch.Generator().manual_seed(channels)
    image = (torch.rand(17, 23, channels, generator=generator) * 255).to(torch.uint8)
    write_png(tmp_path / "x.png", image)
    assert bool((read_png(tmp_path / "x.png") == image).all())


def test_a_greyscale_two_dimensional_image_comes_back_with_one_channel(tmp_path):
    image = (torch.rand(9, 11) * 255).to(torch.uint8)
    write_png(tmp_path / "g.png", image)
    assert read_png(tmp_path / "g.png").shape == (9, 11, 1)


def test_a_float_image_is_rounded_not_truncated(tmp_path):
    """Truncation biases every channel down by half a level, which is invisible
    in one image and a systematic difference when two sheets are compared."""
    image = torch.tensor([[[0.5, 0.2509804, 1.0]]])
    write_png(tmp_path / "f.png", image)
    assert read_png(tmp_path / "f.png")[0, 0].tolist() == [128, 64, 255]


def test_values_outside_zero_to_one_are_clamped(tmp_path):
    write_png(tmp_path / "c.png", torch.tensor([[[-3.0, 0.5, 7.0]]]))
    assert read_png(tmp_path / "c.png")[0, 0].tolist() == [0, 128, 255]


def test_to_uint8_passes_uint8_through_untouched():
    image = torch.tensor([[[1, 254]]], dtype=torch.uint8)
    assert to_uint8(image) is image


def test_to_uint8_refuses_an_integer_type_it_cannot_interpret():
    with pytest.raises(ValueError, match="float or uint8"):
        to_uint8(torch.zeros(4, 4, 3, dtype=torch.int32))


# --- the file really is a PNG ----------------------------------------------


def test_the_file_has_the_signature_and_chunks_a_png_must_have(tmp_path):
    write_png(tmp_path / "s.png", torch.zeros(4, 4, 3))
    blob = (tmp_path / "s.png").read_bytes()
    assert blob.startswith(_SIGNATURE)
    assert (
        b"IHDR" in blob and b"IDAT" in blob and blob.endswith(b"IEND\xae\x42\x60\x82")
    )


def test_every_chunk_carries_a_correct_crc(tmp_path):
    """Checked independently of the writer: a viewer will reject the file if
    this is wrong, and nothing else in the suite would notice."""
    write_png(tmp_path / "s.png", (torch.rand(12, 12, 3) * 255).to(torch.uint8))
    blob = (tmp_path / "s.png").read_bytes()
    cursor, seen = len(_SIGNATURE), []
    while cursor < len(blob):
        (length,) = struct.unpack(">I", blob[cursor : cursor + 4])
        body = blob[cursor + 4 : cursor + 8 + length]
        (stored,) = struct.unpack(
            ">I", blob[cursor + 8 + length : cursor + 12 + length]
        )
        assert stored == zlib.crc32(body), body[:4]
        seen.append(body[:4])
        cursor += 12 + length
    assert seen == [b"IHDR", b"IDAT", b"IEND"]


def test_the_header_declares_the_size_and_colour_type_it_wrote(tmp_path):
    write_png(tmp_path / "h.png", torch.zeros(7, 13, 4))
    header = (tmp_path / "h.png").read_bytes()[16:29]
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", header)
    assert (width, height, depth, colour, interlace) == (13, 7, 8, 6, 0)


# --- decoding, against hand-built scanlines --------------------------------


def test_the_decoder_reconstructs_each_filter_type(tmp_path):
    """All five, with the expected pixels worked out from the specification
    rather than taken from this package's own encoder.

    One channel, four columns, five rows, one filter type per row. The
    residuals are chosen so that each row's answer can be written down::

        row 0  None   raw                       -> 10 20 30 40
        row 1  Sub    +5 on the pixel to the left -> 5 10 15 20
        row 2  Up     +1 on the pixel above     -> 6 11 16 21
        row 3  Avg    floor((left + above) / 2) -> 3  7 11 16
        row 4  Paeth  the chosen predictor      -> 3  7 11 16

    Row 4 repeats row 3 because with a zero residual Paeth picks the pixel
    above at every column: the estimate ``left + up - upleft`` equals ``up``
    exactly, so its own error is zero and it always wins.
    """
    blob = _build_png(
        4,
        5,
        1,
        [
            (0, [10, 20, 30, 40]),
            (1, [5, 5, 5, 5]),
            (2, [1, 1, 1, 1]),
            (3, [0, 0, 0, 0]),
            (4, [0, 0, 0, 0]),
        ],
    )
    (tmp_path / "filters.png").write_bytes(blob)
    decoded = read_png(tmp_path / "filters.png")[..., 0].tolist()

    assert decoded[0] == [10, 20, 30, 40]
    assert decoded[1] == [5, 10, 15, 20]
    assert decoded[2] == [6, 11, 16, 21]
    assert decoded[3] == [3, 7, 11, 16]
    assert decoded[4] == [3, 7, 11, 16]


def test_a_corrupt_chunk_is_refused_rather_than_decoded_into_wrong_pixels(tmp_path):
    blob = bytearray(_build_png(4, 1, 1, [(0, [1, 2, 3, 4])]))
    blob[20] = (blob[20] + 1) % 256  # a height byte inside IHDR
    (tmp_path / "corrupt.png").write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="IHDR chunk CRC mismatch"):
        read_png(tmp_path / "corrupt.png")


def test_a_file_that_is_not_a_png_says_so(tmp_path):
    (tmp_path / "nope.png").write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="not a PNG"):
        read_png(tmp_path / "nope.png")


def test_an_interlaced_png_is_refused_rather_than_mis_decoded(tmp_path):
    """Adam7 is a different scanline layout entirely. Decoding it as if it were
    progressive produces an image, which is what makes it worth refusing."""
    (tmp_path / "i.png").write_bytes(_build_png(2, 1, 1, [(0, [1, 2])], interlace=1))
    with pytest.raises(ValueError, match="interlaced"):
        read_png(tmp_path / "i.png")


def test_a_sixteen_bit_png_is_refused_rather_than_read_as_eight(tmp_path):
    (tmp_path / "d.png").write_bytes(_build_png(1, 1, 1, [(0, [0, 0])], depth=16))
    with pytest.raises(ValueError, match="only 8-bit"):
        read_png(tmp_path / "d.png")


def test_an_unknown_filter_type_names_the_row(tmp_path):
    (tmp_path / "bad.png").write_bytes(_build_png(2, 1, 1, [(9, [1, 2])]))
    with pytest.raises(ValueError, match="filter type 9 on row 0"):
        read_png(tmp_path / "bad.png")


# --- filtering pays for itself ---------------------------------------------


def test_adaptive_filtering_beats_storing_the_rows_raw(tmp_path):
    """A smooth gradient is the case filtering exists for.

    The baseline is the *same image* with every filter byte set to zero, which
    is what a writer that did not bother would emit, and the comparison is
    compressed-stream to compressed-stream so that the chunk overhead does not
    flatter either side. Measured: 306 bytes filtered against 681 unfiltered,
    a factor of 2.23. The assertion is set at 1.5 so that a small regression in
    the heuristic is tolerated and abandoning it is not.
    """
    ramp = torch.linspace(0, 1, 128).view(1, 128, 1).expand(128, 128, 3)
    write_png(tmp_path / "ramp.png", ramp)

    blob = (tmp_path / "ramp.png").read_bytes()
    cursor, filtered = len(_SIGNATURE), None
    while filtered is None:
        (length,) = struct.unpack(">I", blob[cursor : cursor + 4])
        if blob[cursor + 4 : cursor + 8] == b"IDAT":
            filtered = length
        cursor += 12 + length

    rows = (ramp.clamp(0, 1) * 255).round().to(torch.uint8).reshape(128, 128 * 3)
    unfiltered = len(
        zlib.compress(b"".join(b"\x00" + bytes(row.tolist()) for row in rows), 6)
    )
    assert filtered * 1.5 < unfiltered, (filtered, unfiltered)


# --- the font ---------------------------------------------------------------


def test_text_lands_where_it_was_asked_to():
    """``H`` is the alignment probe: it is the one glyph that inks every row
    and every column of its box, so its extent is exactly the box."""
    canvas = torch.zeros(20, 60, 3)
    draw_text(canvas, "H", (3, 5), colour=1.0, scale=1)
    ink = canvas.sum(dim=-1) > 0
    rows = ink.any(dim=1).nonzero().flatten().tolist()
    columns = ink.any(dim=0).nonzero().flatten().tolist()
    assert rows == list(range(3, 3 + GLYPH_HEIGHT))
    assert columns == list(range(5, 5 + GLYPH_WIDTH))


def test_no_glyph_draws_outside_its_own_box():
    """A stray bit in the table would make two neighbouring characters touch,
    and every label in the repo would be slightly wrong in the same way."""
    from atlas.imageio import _FONT, _MISSING

    for glyph in list(_FONT.values()) + [_MISSING]:
        assert len(glyph) == GLYPH_HEIGHT
        for bits in glyph:
            assert 0 <= bits < (1 << GLYPH_WIDTH), (glyph, bits)


def test_text_size_matches_what_was_drawn():
    canvas = torch.zeros(40, 200, 3)
    draw_text(canvas, "ATLAS", (2, 2), scale=3)
    ink = canvas.sum(dim=-1) > 0
    width, height = text_size("ATLAS", scale=3)
    assert int(ink.any(dim=0).nonzero().max()) < 2 + width
    assert int(ink.any(dim=1).nonzero().max()) < 2 + height


def test_lower_case_is_folded_rather_than_dropped():
    upper, lower = torch.zeros(12, 40, 1), torch.zeros(12, 40, 1)
    draw_text(upper, "PSNR", (1, 1))
    draw_text(lower, "psnr", (1, 1))
    assert bool((upper == lower).all())


def test_an_unknown_character_draws_a_box_rather_than_a_space():
    canvas = torch.zeros(12, 20, 1)
    draw_text(canvas, "é", (1, 1))
    assert float(canvas.sum()) > 0


def test_drawing_off_the_edge_clips_instead_of_raising():
    canvas = torch.zeros(12, 20, 1)
    draw_text(canvas, "LONG LABEL THAT DOES NOT FIT", (1, 1))
    draw_text(canvas, "X", (-4, -4))
    draw_text(canvas, "X", (400, 400))
    assert canvas.shape == (12, 20, 1)


def test_the_colour_is_per_channel_when_a_sequence_is_given():
    canvas = torch.zeros(12, 20, 3)
    draw_text(canvas, "X", (1, 1), colour=(1.0, 0.5, 0.0))
    lit = canvas[canvas.sum(dim=-1) > 0]
    assert lit.shape[0] > 0
    assert torch.allclose(lit[0], torch.tensor([1.0, 0.5, 0.0]))


def test_a_colour_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="scalar or 3 values"):
        draw_text(torch.zeros(12, 20, 3), "X", (1, 1), colour=(1.0, 0.0))


def test_every_glyph_is_distinct_so_a_label_can_be_read_back():
    """A duplicated row in the font table would make two characters identical
    and a label ambiguous, and nothing else would notice."""
    from atlas.imageio import _FONT

    rendered = {}
    for character, glyph in _FONT.items():
        if character == " ":
            continue
        assert glyph not in rendered, f"{character!r} and {rendered[glyph]!r} match"
        rendered[glyph] = character
