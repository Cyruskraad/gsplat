#!/usr/bin/env python3
"""Validate and stage an object-aware COLMAP dataset for gsplat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from colmap_scripts.gsplat_hq import (
    ViewRecord,
    inspect_object_mask,
    stage_gsplat_dataset,
    validate_image_mask_contract,
    write_split_manifest,
)


def _camera_records(model_path: Path) -> list[ViewRecord]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    records: list[ViewRecord] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        pose = image.cam_from_world
        if callable(pose):
            pose = pose()
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :4] = np.asarray(pose.matrix(), dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)
        records.append(
            ViewRecord(
                image.name,
                tuple(float(value) for value in camera_to_world[:3, 3]),
                tuple(float(value) for value in camera_to_world[:3, 2]),
            )
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=300.0)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=11)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    free_gib = shutil.disk_usage(args.destination.parent).free / 1024**3
    if free_gib < args.minimum_free_disk_gib:
        raise SystemExit(
            f"free disk {free_gib:.1f} GiB is below {args.minimum_free_disk_gib:.1f} GiB"
        )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu", "--format=csv,noheader"],
        check=False,
        text=True,
        capture_output=True,
    )
    if gpu.returncode != 0:
        raise SystemExit(f"GPU preflight failed: {gpu.stderr.strip()}")

    records = _camera_records(args.sparse)
    registered_names = sorted(record.name for record in records)
    image_paths = {
        str(path.relative_to(args.images)): path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    }
    mask_paths = {
        str(path.relative_to(args.masks)): path
        for path in args.masks.rglob("*.png")
        if path.is_file()
    }
    if set(registered_names) != set(image_paths):
        missing = sorted(set(registered_names) - set(image_paths))
        extra = sorted(set(image_paths) - set(registered_names))
        raise SystemExit(f"registered image mismatch: missing={missing} extra={extra}")
    image_to_mask = validate_image_mask_contract(registered_names, mask_paths)

    mask_reports: list[dict[str, object]] = []
    failures: list[str] = []
    for image_name in registered_names:
        image_path = image_paths[image_name]
        mask_path = mask_paths[image_to_mask[image_name]]
        with Image.open(image_path) as image, Image.open(mask_path) as source_mask:
            if image.size != source_mask.size:
                failures.append(f"{image_name}: resolution {image.size} != {source_mask.size}")
            mask = np.asarray(source_mask.convert("L")) > 127
        report = inspect_object_mask(mask)
        valid = bool(
            0.02 <= report["fill_fraction"] <= 0.65
            and report["largest_component_fraction"] >= 0.90
        )
        if not valid:
            failures.append(f"{image_name}: {report}")
        mask_reports.append({"image": image_name, "valid": valid, **report})
    if failures:
        raise SystemExit("mask preflight failed:\n" + "\n".join(failures))

    stage_gsplat_dataset(
        images_dir=args.images,
        sparse_dir=args.sparse.parent if args.sparse.name == "0" else args.sparse,
        masks_dir=args.masks,
        destination=args.destination,
    )
    split_path = write_split_manifest(
        records,
        args.destination / "split.json",
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    payload = {
        "schema_version": 1,
        "registered_images": len(records),
        "free_disk_gib": round(free_gib, 2),
        "gpu": gpu.stdout.strip(),
        "split_manifest": str(split_path.resolve()),
        "mask_reports": mask_reports,
    }
    (args.destination / "preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "mask_reports"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
