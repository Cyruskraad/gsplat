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

"""Look at a capture directory and say whether it can support relighting.

This runs **before** any loader exists, and the loader is written against what
it reports rather than against an assumption. It reads; it never writes into the
capture.

What it can establish, and what it cannot
-----------------------------------------

It can establish the structural facts: how many images, in what formats, at what
bit depth, whether a camera solve is present and how many images it registers,
whether masks exist and match, and -- where EXIF survives -- whether the flash
fired, per shot.

It can measure how much the image content varies: overall luminance, and the
position of the brightest region.

It **cannot**, from pixels alone, separate "the light moved" from "the camera
moved". Both move the highlight. The report says so rather than guessing, and
names the two things that do settle it: the EXIF flash tag, and -- once a camera
solve is in hand -- whether the highlight directions agree on a single fixed
world light. The second is a follow-up, not something to fake here.

Dependencies
------------

Structural checks need nothing but the standard library. Pixel statistics and
EXIF need Pillow; RAW pixel statistics need rawpy. Both are optional, both
degrade to a stated "unavailable" rather than to a crash, so this can be run on
a workstation without installing anything first::

    python -m atlas.tools.inspect_capture /path/to/capture -o capture_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CaptureReport",
    "inspect_capture",
    "format_summary",
    "main",
]

RAW_SUFFIXES = {
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
}
LDR_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HDR_SUFFIXES = {".exr", ".hdr"}
IMAGE_SUFFIXES = RAW_SUFFIXES | LDR_SUFFIXES | HDR_SUFFIXES
MASK_DIR_NAMES = {"mask", "masks", "alpha", "alphas", "segmentation"}

# Highlight movement below this fraction of the image diagonal is not
# distinguishable from noise in the centroid estimate.
HIGHLIGHT_STILL = 0.02
# Above this, something is definitely moving -- the light, the camera, or both.
HIGHLIGHT_MOVING = 0.05
# Relative spread of mean luminance that indicates the illumination changed
# rather than merely the viewpoint.
LUMA_VARIES = 0.15


@dataclass
class ImageFacts:
    """Everything cheap that can be learned about one frame."""

    path: str
    suffix: str
    bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    mtime: Optional[float] = None
    # EXIF, where available.
    flash_fired: Optional[bool] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    datetime: Optional[str] = None
    camera: Optional[str] = None
    # Pixel statistics, from a thumbnail.
    mean_luma: Optional[float] = None
    p99_luma: Optional[float] = None
    clipped_fraction: Optional[float] = None
    highlight_x: Optional[float] = None  # normalised [0, 1]
    highlight_y: Optional[float] = None
    highlight_area: Optional[float] = None  # fraction of the frame
    error: Optional[str] = None


@dataclass
class CaptureReport:
    """The whole finding, serialised to ``capture_report.json``."""

    root: str
    tool_version: str = "1"
    num_images: int = 0
    formats: Dict[str, int] = field(default_factory=dict)
    resolutions: Dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    pixels_read: bool = False
    exif_read: bool = False
    missing_optional: List[str] = field(default_factory=list)

    camera_solve: Dict[str, Any] = field(default_factory=dict)
    masks: Dict[str, Any] = field(default_factory=dict)

    flash_fired_count: Optional[int] = None
    flash_absent_count: Optional[int] = None
    flash_unknown_count: Optional[int] = None
    flash_no_flash_pairs: Optional[int] = None

    luminance: Dict[str, Any] = field(default_factory=dict)
    highlight: Dict[str, Any] = field(default_factory=dict)

    verdict: str = "unknown"
    reasons: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)


# --- optional backends ------------------------------------------------------


def _load_pillow():
    try:
        from PIL import ExifTags, Image  # noqa: F401

        return Image, ExifTags
    except Exception:
        return None, None


def _exif_of(image, exif_tags) -> Dict[str, Any]:
    """Pull the handful of EXIF fields that matter here."""
    out: Dict[str, Any] = {}
    try:
        raw = image.getexif()
    except Exception:
        return out
    if not raw:
        return out
    names = {v: k for k, v in exif_tags.TAGS.items()} if exif_tags else {}

    def get(tag_name):
        tag = names.get(tag_name)
        return raw.get(tag) if tag is not None else None

    out["camera"] = get("Model")
    out["datetime"] = get("DateTime")
    # Exposure parameters live in the Exif IFD rather than IFD0.
    try:
        ifd = raw.get_ifd(0x8769)
    except Exception:
        ifd = {}
    exif_names = {v: k for k, v in exif_tags.TAGS.items()} if exif_tags else {}

    def get_exif(tag_name):
        tag = exif_names.get(tag_name)
        return ifd.get(tag) if tag is not None else None

    flash = get_exif("Flash")
    if flash is not None:
        try:
            # Bit 0 of the Flash tag is "flash fired".
            out["flash_fired"] = bool(int(flash) & 1)
        except Exception:
            out["flash_fired"] = None
    for key, tag in (
        ("exposure_time", "ExposureTime"),
        ("f_number", "FNumber"),
        ("iso", "ISOSpeedRatings"),
        ("focal_length", "FocalLength"),
    ):
        value = get_exif(tag)
        if value is not None:
            try:
                out[key] = int(value) if key == "iso" else float(value)
            except Exception:
                pass
    if out.get("datetime") is None:
        out["datetime"] = get_exif("DateTimeOriginal")
    return {k: v for k, v in out.items() if v is not None}


def _thumbnail_stats(image, thumb: int) -> Dict[str, Any]:
    """Luminance and highlight-centroid statistics from a small thumbnail.

    The centroid is of the brightest one percent of pixels, which is where a
    specular highlight or a chrome-sphere reflection lands. Reported in
    normalised image coordinates so that frames of different sizes compare.
    """
    grey = image.convert("L")
    grey.thumbnail((thumb, thumb))
    width, height = grey.size
    # tobytes() on an "L" image is exactly the pixel values, and unlike
    # getdata() it has not been renamed across Pillow versions. This tool has to
    # run first time on a workstation whose Pillow version is not ours.
    pixels = grey.tobytes()
    if not pixels or width < 1 or height < 1:
        return {}
    count = len(pixels)
    mean = sum(pixels) / count / 255.0
    ordered = sorted(pixels)
    p99 = ordered[min(count - 1, int(0.99 * count))] / 255.0
    clipped = sum(1 for p in pixels if p >= 254) / count

    threshold = ordered[min(count - 1, int(0.99 * count))]
    total = 0.0
    sum_x = 0.0
    sum_y = 0.0
    bright = 0
    for index, value in enumerate(pixels):
        if value >= threshold and value > 0:
            weight = float(value)
            total += weight
            sum_x += weight * (index % width)
            sum_y += weight * (index // width)
            bright += 1
    stats: Dict[str, Any] = {
        "mean_luma": mean,
        "p99_luma": p99,
        "clipped_fraction": clipped,
    }
    if total > 0 and width > 1 and height > 1:
        stats["highlight_x"] = sum_x / total / (width - 1)
        stats["highlight_y"] = sum_y / total / (height - 1)
        stats["highlight_area"] = bright / count
    return stats


# --- structural discovery ---------------------------------------------------


def _find_images(root: Path, max_images: Optional[int]) -> List[Path]:
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not d.startswith(".") and d.lower() not in MASK_DIR_NAMES
        ]
        for name in sorted(filenames):
            if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                found.append(Path(dirpath) / name)
                if max_images is not None and len(found) >= max_images:
                    return found
    return found


def _find_camera_solve(root: Path) -> Dict[str, Any]:
    """Look for a COLMAP model, a Metashape export, or a transforms.json."""
    out: Dict[str, Any] = {"kind": None, "path": None, "registered_images": None}

    for candidate in sorted(root.rglob("images.txt")):
        if candidate.parent.name.startswith("0") or candidate.parent.name == "sparse":
            registered = 0
            try:
                with candidate.open("r", errors="ignore") as handle:
                    for line in handle:
                        if line.startswith("#") or not line.strip():
                            continue
                        # COLMAP images.txt alternates a pose line and a
                        # points line; only the pose lines start with an id.
                        parts = line.split()
                        if len(parts) >= 10 and parts[0].isdigit():
                            registered += 1
            except Exception:
                registered = None
            out.update(
                kind="colmap-text",
                path=str(candidate),
                registered_images=registered,
            )
            return out

    for candidate in sorted(root.rglob("images.bin")):
        out.update(kind="colmap-binary", path=str(candidate), registered_images=None)
        return out

    for candidate in sorted(root.rglob("*.xml")):
        try:
            head = candidate.read_text(errors="ignore")[:4096]
        except Exception:
            continue
        if "<document" in head or "<chunk" in head or "<cameras" in head:
            try:
                text = candidate.read_text(errors="ignore")
                registered = len(re.findall(r"<camera\b[^>]*>", text))
            except Exception:
                registered = None
            out.update(
                kind="metashape-xml", path=str(candidate), registered_images=registered
            )
            return out

    for candidate in sorted(root.rglob("transforms*.json")):
        try:
            data = json.loads(candidate.read_text(errors="ignore"))
            registered = len(data.get("frames", []))
        except Exception:
            registered = None
        out.update(
            kind="nerf-transforms", path=str(candidate), registered_images=registered
        )
        return out

    return out


def _find_masks(root: Path, image_paths: Sequence[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"directory": None, "count": 0, "matched_by_stem": 0}
    mask_dirs = [
        Path(dirpath) / d
        for dirpath, dirnames, _ in os.walk(root)
        for d in dirnames
        if d.lower() in MASK_DIR_NAMES
    ]
    if not mask_dirs:
        return out
    directory = sorted(mask_dirs)[0]
    masks = [
        p
        for p in sorted(directory.rglob("*"))
        if p.suffix.lower() in LDR_SUFFIXES and p.is_file()
    ]
    stems = {p.stem for p in image_paths}
    out.update(
        directory=str(directory),
        count=len(masks),
        matched_by_stem=sum(1 for m in masks if m.stem in stems),
    )
    return out


def _detect_pairs(facts: Sequence[ImageFacts]) -> Optional[int]:
    """Count plausible flash / no-flash pairs.

    A pair is two frames close together in time whose mean luminance differs
    markedly. Timestamps come from EXIF when present and from the file mtime
    otherwise, which is weaker but usually preserved by a card copy.
    """
    stamped = [
        f
        for f in facts
        if f.mtime is not None and f.mean_luma is not None and f.error is None
    ]
    if len(stamped) < 4:
        return None
    stamped = sorted(stamped, key=lambda f: f.mtime)
    pairs = 0
    index = 0
    while index < len(stamped) - 1:
        first, second = stamped[index], stamped[index + 1]
        close_in_time = abs(second.mtime - first.mtime) <= 8.0
        brighter = max(first.mean_luma, second.mean_luma)
        darker = min(first.mean_luma, second.mean_luma)
        differs = brighter > 1.6 * max(darker, 1e-4)
        if close_in_time and differs:
            pairs += 1
            index += 2
        else:
            index += 1
    return pairs


def _spread(values: Sequence[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return None
    return statistics.pstdev(clean)


# --- the inspection itself --------------------------------------------------


def inspect_capture(
    root: Path,
    *,
    max_images: Optional[int] = None,
    thumb: int = 128,
    read_pixels: bool = True,
) -> CaptureReport:
    """Inspect ``root`` and return a report. Never writes inside ``root``."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    report = CaptureReport(root=str(root))
    image_paths = _find_images(root, max_images)
    report.num_images = len(image_paths)

    Image, ExifTags = _load_pillow() if read_pixels else (None, None)
    if read_pixels and Image is None:
        report.missing_optional.append(
            "Pillow -- no pixel statistics, no EXIF. pip install pillow"
        )
    report.pixels_read = Image is not None
    report.exif_read = Image is not None

    facts: List[ImageFacts] = []
    raw_seen = False
    for path in image_paths:
        suffix = path.suffix.lower()
        raw_seen = raw_seen or suffix in RAW_SUFFIXES
        try:
            size = path.stat().st_size
            mtime = path.stat().st_mtime
        except OSError as exc:
            facts.append(ImageFacts(str(path), suffix, 0, error=str(exc)))
            continue
        item = ImageFacts(
            path=str(path.relative_to(root)), suffix=suffix, bytes=size, mtime=mtime
        )
        report.total_bytes += size
        report.formats[suffix] = report.formats.get(suffix, 0) + 1

        if Image is not None and suffix not in RAW_SUFFIXES | HDR_SUFFIXES:
            try:
                with Image.open(path) as image:
                    item.width, item.height = image.size
                    item.mode = image.mode
                    key = f"{image.size[0]}x{image.size[1]}"
                    report.resolutions[key] = report.resolutions.get(key, 0) + 1
                    for name, value in _exif_of(image, ExifTags).items():
                        setattr(item, name, value)
                    # draft() makes JPEG decode at a fraction of full size,
                    # which is the difference between seconds and minutes over
                    # a few hundred 24-megapixel frames.
                    try:
                        image.draft("L", (thumb, thumb))
                    except Exception:
                        pass
                    for name, value in _thumbnail_stats(image, thumb).items():
                        setattr(item, name, value)
            except Exception as exc:  # pragma: no cover - depends on the file
                item.error = f"{type(exc).__name__}: {exc}"
        facts.append(item)

    if raw_seen:
        report.missing_optional.append(
            "rawpy -- RAW frames were counted but not decoded, so their "
            "statistics are absent. pip install rawpy"
        )

    report.camera_solve = _find_camera_solve(root)
    report.masks = _find_masks(root, image_paths)

    flags = [f.flash_fired for f in facts]
    if any(flag is not None for flag in flags):
        report.flash_fired_count = sum(1 for f in flags if f is True)
        report.flash_absent_count = sum(1 for f in flags if f is False)
        report.flash_unknown_count = sum(1 for f in flags if f is None)
    report.flash_no_flash_pairs = _detect_pairs(facts)

    lumas = [f.mean_luma for f in facts if f.mean_luma is not None]
    if lumas:
        mean_luma = statistics.fmean(lumas)
        spread = _spread(lumas)
        report.luminance = {
            "mean": mean_luma,
            "stdev": spread,
            "relative_spread": (spread / mean_luma) if (spread and mean_luma) else None,
            "min": min(lumas),
            "max": max(lumas),
            "clipped_fraction_mean": statistics.fmean(
                [f.clipped_fraction for f in facts if f.clipped_fraction is not None]
                or [0.0]
            ),
        }

    xs = [f.highlight_x for f in facts if f.highlight_x is not None]
    ys = [f.highlight_y for f in facts if f.highlight_y is not None]
    if len(xs) >= 2:
        cx, cy = statistics.fmean(xs), statistics.fmean(ys)
        rms = math.sqrt(
            statistics.fmean([(x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)])
        )
        report.highlight = {
            "centroid_rms_normalised": rms,
            "mean_area_fraction": statistics.fmean(
                [f.highlight_area for f in facts if f.highlight_area is not None]
                or [0.0]
            ),
            "note": (
                "Highlight movement shows the *image* changed. It does not by "
                "itself separate a moving light from a moving camera; see the "
                "flash tags and the camera-solve follow-up."
            ),
        }

    report.images = [asdict(f) for f in facts]
    _decide(report)
    return report


def _decide(report: CaptureReport) -> None:
    """Fill in the verdict, the reasons for it, and what to do next."""
    reasons: List[str] = []
    steps: List[str] = []

    if report.num_images == 0:
        report.verdict = "not_a_capture"
        report.reasons = ["No image files found under the given directory."]
        report.next_steps = [
            "Check the path, and that the images are not in an archive."
        ]
        return

    if report.num_images < 20:
        reasons.append(
            f"Only {report.num_images} images. Relighting needs the light to be "
            f"sampled from many directions; expect this to underfit."
        )

    # The strongest evidence available without poses: the flash tag.
    lighting_evidence = None
    if report.flash_fired_count is not None:
        fired = report.flash_fired_count
        absent = report.flash_absent_count or 0
        if fired == 0:
            reasons.append(
                "EXIF says the flash never fired. If the light was a separate "
                "strobe this is fine; if there was no moving light at all, this "
                "is a fixed-light capture and cannot train a relighting model."
            )
            lighting_evidence = False
        elif absent > 0 and fired > 0:
            reasons.append(
                f"EXIF shows {fired} flash frames and {absent} without. That is "
                f"the flash / no-flash pattern the protocol wants -- ambient "
                f"subtraction is available."
            )
            lighting_evidence = True
        else:
            reasons.append(f"EXIF shows the flash fired on all {fired} frames.")
            lighting_evidence = True
    else:
        reasons.append(
            "No EXIF flash tag was readable, so whether a flash fired is unknown."
        )

    rms = report.highlight.get("centroid_rms_normalised")
    relative = report.luminance.get("relative_spread")
    if rms is not None:
        if rms < HIGHLIGHT_STILL:
            reasons.append(
                f"The brightest region barely moves between frames "
                f"(RMS {rms:.3f} of the frame). Either the camera and the light "
                f"both held still, or these are frames of a static setup."
            )
            if lighting_evidence is not True:
                lighting_evidence = False
        elif rms >= HIGHLIGHT_MOVING:
            reasons.append(
                f"The brightest region moves substantially (RMS {rms:.3f} of the "
                f"frame). Consistent with a moving light, a moving camera, or both."
            )
        else:
            reasons.append(
                f"The brightest region moves a little (RMS {rms:.3f} of the frame)."
            )
    if relative is not None:
        if relative >= LUMA_VARIES:
            reasons.append(
                f"Overall brightness varies a lot across frames "
                f"(relative spread {relative:.2f}), which a pure camera orbit "
                f"under fixed light does not usually produce."
            )
        else:
            reasons.append(
                f"Overall brightness is fairly constant across frames "
                f"(relative spread {relative:.2f})."
            )

    solve = report.camera_solve
    if solve.get("kind"):
        registered = solve.get("registered_images")
        reasons.append(
            f"Found a {solve['kind']} camera solve at {solve['path']}"
            + (f", registering {registered} images." if registered else ".")
        )
        steps.append(
            "Definitive lighting test: with these poses, map each frame's "
            "highlight to a world direction and check whether they agree on one "
            "fixed light. Agreement means fixed light; disagreement means the "
            "light moved. This is the check pixels alone cannot make."
        )
    else:
        reasons.append(
            "No camera solve found (no COLMAP model, Metashape XML or transforms.json)."
        )
        steps.append(
            "Run SfM. If flash / no-flash pairs exist, run it on the no-flash "
            "half -- those are constant-lit, so matching behaves, and the flash "
            "frames inherit the poses."
        )

    masks = report.masks
    if masks.get("count"):
        reasons.append(
            f"{masks['count']} masks in {masks['directory']}, "
            f"{masks['matched_by_stem']} matching an image by filename stem."
        )
    else:
        steps.append(
            "No masks found. Object masks are optional but they were worth a lot "
            "on the fixed-light runs; tools/generate_sam2_masks.py in the gsplat "
            "fork produces them."
        )

    if report.flash_no_flash_pairs:
        reasons.append(
            f"About {report.flash_no_flash_pairs} flash / no-flash pairs are "
            f"detectable from timestamps and brightness."
        )

    if report.missing_optional:
        steps.extend(f"Optional: install {item}" for item in report.missing_optional)

    # Verdict.
    moving = (rms is not None and rms >= HIGHLIGHT_MOVING) or (
        relative is not None and relative >= LUMA_VARIES
    )
    if lighting_evidence is False and not moving:
        report.verdict = "not_a_relighting_capture"
        steps.insert(
            0,
            "Nothing here shows the illumination changing between frames. A "
            "relighting model needs that. Shoot a flash-varying set of the same "
            "object -- the existing reconstruction still supplies the geometry.",
        )
    elif solve.get("kind") and (lighting_evidence is True or moving):
        report.verdict = "usable"
        steps.insert(
            0,
            "Enough to attempt a first training run. Send this report back and "
            "the loader gets written against it.",
        )
    else:
        report.verdict = "usable_with_work"
        steps.insert(
            0,
            "Probably usable, but something is missing -- see the steps below.",
        )

    report.reasons = reasons
    report.next_steps = steps


# --- presentation -----------------------------------------------------------


def format_summary(report: CaptureReport) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"Capture: {report.root}")
    add("")
    add(f"  images        {report.num_images}")
    if report.formats:
        formats = ", ".join(
            f"{k} x{v}"
            for k, v in sorted(report.formats.items(), key=lambda kv: -kv[1])
        )
        add(f"  formats       {formats}")
    if report.resolutions:
        resolutions = ", ".join(
            f"{k} x{v}"
            for k, v in sorted(report.resolutions.items(), key=lambda kv: -kv[1])[:4]
        )
        add(f"  resolutions   {resolutions}")
    add(f"  total size    {report.total_bytes / 1e9:.2f} GB")
    if report.flash_fired_count is not None:
        add(
            f"  flash         {report.flash_fired_count} fired / "
            f"{report.flash_absent_count} not / {report.flash_unknown_count} unknown"
        )
    if report.luminance:
        add(
            f"  luminance     mean {report.luminance['mean']:.3f}, "
            f"relative spread "
            f"{_fmt(report.luminance.get('relative_spread'))}"
        )
    if report.highlight:
        add(
            f"  highlight     centroid RMS "
            f"{_fmt(report.highlight.get('centroid_rms_normalised'))} of the frame"
        )
    solve = report.camera_solve
    add(f"  camera solve  {solve.get('kind') or 'none found'}")
    add(f"  masks         {report.masks.get('count', 0)}")
    add("")
    add(f"VERDICT: {report.verdict}")
    add("")
    add("Why:")
    for reason in report.reasons:
        add(f"  - {reason}")
    if report.next_steps:
        add("")
        add("Next:")
        for step in report.next_steps:
            add(f"  - {step}")
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas-inspect",
        description="Report whether a capture directory can support relighting.",
    )
    parser.add_argument("capture", type=Path, help="directory to inspect")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write the full JSON report here (default: capture_report.json in cwd)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="stop after this many images, for a quick look at a large capture",
    )
    parser.add_argument(
        "--thumb", type=int, default=128, help="thumbnail size for pixel statistics"
    )
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="structural checks only; skip decoding any image",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="include the per-image table in the printed summary as well as the JSON",
    )
    args = parser.parse_args(argv)

    try:
        report = inspect_capture(
            args.capture,
            max_images=args.max_images,
            thumb=args.thumb,
            read_pixels=not args.no_pixels,
        )
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_summary(report))
    if args.per_image:
        print("\nPer image:")
        for item in report.images:
            print(f"  {json.dumps(item)}")

    destination = args.output or Path("capture_report.json")
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(f"\nFull report written to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
