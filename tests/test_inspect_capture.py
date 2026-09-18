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

"""The inspector, against synthetic captures whose answer is known.

The point of the tool is to be run once on real data and believed. So the two
verdicts that matter most are pinned here with fixtures built to deserve them: a
fixed-light set that must be rejected, and a light-varying set that must not be.
"""

import json
import math
import os

import pytest

from atlas.tools.inspect_capture import (
    format_summary,
    inspect_capture,
    main,
)

Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to build fixtures")


def _frame(path, spot_xy, background=40, spot=250, size=(160, 120), radius=9):
    """An image with a bright blob at ``spot_xy`` in normalised coordinates."""
    width, height = size
    image = Image.new("L", size, background)
    pixels = image.load()
    cx, cy = spot_xy[0] * (width - 1), spot_xy[1] * (height - 1)
    for y in range(height):
        for x in range(width):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 <= radius * radius:
                falloff = math.exp(-d2 / (2.0 * (radius / 2.0) ** 2))
                pixels[x, y] = min(255, int(background + (spot - background) * falloff))
    image.save(path)


def _fixed_light_capture(root, count=30):
    """The light and the camera both hold still: the blob does not move."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _frame(root / f"IMG_{index:04d}.png", (0.5, 0.5), background=40)
    return root


def _light_varying_capture(root, count=30):
    """The bright region sweeps across the frame and brightness swings."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        spot = (0.5 + 0.34 * math.cos(angle), 0.5 + 0.34 * math.sin(angle))
        background = 20 + int(60 * (0.5 + 0.5 * math.sin(3.0 * angle)))
        _frame(root / f"IMG_{index:04d}.png", spot, background=background)
    return root


def _write_colmap(root, registered=30):
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    lines = ["# Image list with two lines of data per image"]
    for index in range(registered):
        lines.append(f"{index + 1} 1 0 0 0 0 0 0 1 IMG_{index:04d}.png")
        lines.append("")
    (sparse / "images.txt").write_text("\n".join(lines))
    return sparse / "images.txt"


# --- the two verdicts that matter -------------------------------------------


def test_a_fixed_light_capture_is_rejected(tmp_path):
    """The failure this tool exists to catch in one minute rather than one week."""
    report = inspect_capture(_fixed_light_capture(tmp_path / "fixed"))
    assert report.verdict == "not_a_relighting_capture"
    assert report.highlight["centroid_rms_normalised"] < 0.02
    assert any("illumination changing" in step for step in report.next_steps)


def test_a_light_varying_capture_with_poses_is_usable(tmp_path):
    root = _light_varying_capture(tmp_path / "varying")
    _write_colmap(root)
    report = inspect_capture(root)
    assert report.verdict == "usable"
    assert report.highlight["centroid_rms_normalised"] > 0.05
    assert report.camera_solve["kind"] == "colmap-text"
    assert report.camera_solve["registered_images"] == 30


def test_a_light_varying_capture_without_poses_needs_work(tmp_path):
    report = inspect_capture(_light_varying_capture(tmp_path / "varying"))
    assert report.verdict == "usable_with_work"
    assert any("Run SfM" in step for step in report.next_steps)


def test_an_empty_directory_is_not_a_capture(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = inspect_capture(empty)
    assert report.verdict == "not_a_capture"
    assert report.num_images == 0


# --- the honesty the verdict depends on -------------------------------------


def test_the_report_says_pixels_cannot_separate_a_moving_light_from_a_moving_camera():
    """Over-claiming here would be worse than saying nothing.

    A camera orbiting a fixed light moves the highlight exactly as a moving
    light does. The note must survive refactoring, so it is asserted.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = _light_varying_capture(Path(tmp) / "v", count=12)
        report = inspect_capture(root)
    note = report.highlight["note"]
    assert "does not by itself separate" in note
    assert "moving camera" in note


def test_a_camera_solve_unlocks_the_definitive_lighting_test(tmp_path):
    root = _light_varying_capture(tmp_path / "v")
    _write_colmap(root)
    report = inspect_capture(root)
    assert any("Definitive lighting test" in step for step in report.next_steps)


# --- structural discovery ----------------------------------------------------


def test_metashape_xml_is_recognised(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=6)
    (root / "doc.xml").write_text(
        '<document version="1.2"><chunk>'
        '<camera id="0" label="a"/><camera id="1" label="b"/>'
        "</chunk></document>"
    )
    report = inspect_capture(root)
    assert report.camera_solve["kind"] == "metashape-xml"
    assert report.camera_solve["registered_images"] == 2


def test_nerf_transforms_are_recognised(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=6)
    (root / "transforms.json").write_text(
        json.dumps({"frames": [{"file_path": "a"}, {"file_path": "b"}]})
    )
    report = inspect_capture(root)
    assert report.camera_solve["kind"] == "nerf-transforms"
    assert report.camera_solve["registered_images"] == 2


def test_masks_are_found_and_matched_by_stem(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=6)
    masks = root / "masks"
    masks.mkdir()
    for index in range(4):
        Image.new("L", (16, 16), 255).save(masks / f"IMG_{index:04d}.png")
    Image.new("L", (16, 16), 255).save(masks / "unrelated.png")
    report = inspect_capture(root)
    assert report.masks["count"] == 5
    assert report.masks["matched_by_stem"] == 4


def test_mask_directories_are_not_counted_as_capture_images(tmp_path):
    """A mask that lands in the image inventory corrupts every statistic."""
    root = _light_varying_capture(tmp_path / "v", count=6)
    masks = root / "masks"
    masks.mkdir()
    for index in range(6):
        Image.new("L", (16, 16), 255).save(masks / f"IMG_{index:04d}.png")
    report = inspect_capture(root)
    assert report.num_images == 6


def test_raw_files_are_counted_and_the_missing_decoder_is_named(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for index in range(4):
        (root / f"DSC_{index:04d}.NEF").write_bytes(b"not really a raw file")
    report = inspect_capture(root)
    assert report.formats[".nef"] == 4
    assert any("rawpy" in item for item in report.missing_optional)


# --- behaviour and reporting -------------------------------------------------


def test_no_pixels_mode_still_reports_structure(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=8)
    _write_colmap(root, registered=8)
    report = inspect_capture(root, read_pixels=False)
    assert report.num_images == 8
    assert report.pixels_read is False
    assert report.camera_solve["kind"] == "colmap-text"
    # Without pixels there is no lighting evidence, so it must not claim usable.
    assert report.verdict != "usable"


def test_max_images_bounds_the_work(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=30)
    report = inspect_capture(root, max_images=7)
    assert report.num_images == 7


def test_a_short_capture_is_flagged_even_when_otherwise_fine(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=8)
    _write_colmap(root, registered=8)
    report = inspect_capture(root)
    assert any("Only 8 images" in reason for reason in report.reasons)


def test_the_summary_renders_and_leads_with_the_verdict(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=8)
    text = format_summary(inspect_capture(root))
    assert "VERDICT:" in text
    assert "Why:" in text
    assert "Next:" in text


def test_the_inspector_never_writes_inside_the_capture(tmp_path):
    root = _light_varying_capture(tmp_path / "v", count=8)
    before = {p.name for p in root.iterdir()}
    inspect_capture(root)
    assert {p.name for p in root.iterdir()} == before


def test_the_cli_writes_a_json_report(tmp_path, capsys, monkeypatch):
    root = _light_varying_capture(tmp_path / "v", count=8)
    _write_colmap(root, registered=8)
    destination = tmp_path / "report.json"
    code = main([str(root), "-o", str(destination)])
    assert code == 0
    payload = json.loads(destination.read_text())
    assert payload["verdict"] == "usable"
    assert payload["num_images"] == 8
    assert "VERDICT:" in capsys.readouterr().out


def test_the_cli_reports_a_bad_path_without_a_traceback(tmp_path, capsys):
    code = main([str(tmp_path / "does-not-exist")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_a_missing_directory_raises_rather_than_returning_a_blank_report(tmp_path):
    with pytest.raises(NotADirectoryError):
        inspect_capture(tmp_path / "nope")
