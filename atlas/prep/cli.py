# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``atlas-prepare PHOTOS OUT``: photographs to a trainable capture."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from .pipeline import PrepConfig, prepare_capture
from .sfm import SfMConfig

__all__ = ["main"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas-prepare",
        description=(
            "Decode (LibRaw), pose (COLMAP/GLOMAP), undistort, mask (BiRefNet) and "
            "export a co-located flash capture for ATLAS training."
        ),
    )
    parser.add_argument("photos", type=Path, help="folder of RAW and/or JPEG shots")
    parser.add_argument("out", type=Path, help="capture directory to write")
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument("--features", choices=("aliked", "sift"), default="aliked")
    parser.add_argument(
        "--matcher", choices=("lightglue", "bruteforce"), default="lightglue"
    )
    parser.add_argument(
        "--pairing", choices=("exhaustive", "sequential"), default="exhaustive"
    )
    parser.add_argument("--mapper", choices=("global", "incremental"), default="global")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--flash-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="flash position in camera axes (x right, y down, z forward), metres",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.0,
        help="typical camera-to-object distance in metres; sets the scene scale",
    )
    parser.add_argument("--masks", choices=("none", "rembg"), default="none")
    parser.add_argument("--mask-model", default="birefnet-general")
    parser.add_argument("--init-radius", type=float, default=1.0)
    parser.add_argument("--no-exif-exposure", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = PrepConfig(
        max_side=args.max_side,
        sfm=SfMConfig(
            features=args.features,
            matcher=args.matcher,
            pairing=args.pairing,
            mapper=args.mapper,
            camera_model=args.camera_model,
            device=args.device,
        ),
        flash_offset=tuple(args.flash_offset),
        camera_distance=args.camera_distance,
        masks=args.masks,
        mask_model=args.mask_model,
        init_radius=args.init_radius,
        use_exif_exposure=not args.no_exif_exposure,
    )
    report = prepare_capture(args.photos, args.out, config)
    sfm, export = report["sfm"], report["export"]
    print(
        f"registered {sfm['num_registered']}/{sfm['num_images']} images "
        f"({sfm['features']}, {sfm['mapper']} mapping, "
        f"{sfm['mean_reprojection_error_px']:.2f} px reprojection error)"
    )
    print(
        f"wrote {export['num_frames']} frames at {export['image_size'][0]}x"
        f"{export['image_size'][1]}, {export['num_init_points']} init points "
        f"-> {args.out / 'transforms.json'}"
    )
    for note in report["notes"]:
        print(f"note: {note}")
    print(json.dumps(report["timings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
