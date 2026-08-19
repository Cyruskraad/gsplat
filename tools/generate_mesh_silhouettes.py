#!/usr/bin/env python3
"""Rasterize conservative object silhouettes from an existing COLMAP-aligned mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--names", nargs="*", default=[])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.factor < 1:
        raise SystemExit("factor must be positive")
    if not args.model.is_dir():
        raise SystemExit(f"COLMAP model directory missing: {args.model}")
    if not args.mesh.is_file():
        raise SystemExit(f"object mesh missing: {args.mesh}")
    if not args.images.is_dir():
        raise SystemExit(f"image directory missing: {args.images}")

    import pycolmap

    reconstruction = pycolmap.Reconstruction(args.model)
    ply = PlyData.read(args.mesh)
    vertices = ply["vertex"]
    points = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float64)
    faces = np.asarray(
        [face for face in ply["face"]["vertex_indices"]], dtype=np.int32
    )
    if len(points) == 0 or len(faces) == 0 or faces.shape[1] != 3:
        raise SystemExit("mesh must contain triangular faces")

    mask_dir = args.output / "masks"
    overlay_dir = args.output / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    allowed = {name.casefold() for name in args.names}
    images = {
        image.name: image
        for image in reconstruction.images.values()
        if not allowed or image.name.casefold() in allowed
    }
    if not images:
        raise SystemExit("no matching registered images")

    records: list[dict[str, object]] = []
    for name in sorted(images):
        stem = Path(name).stem
        mask_path = mask_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.jpg"
        if mask_path.exists() and args.resume:
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            records.append(
                {
                    "image": name,
                    "resumed": True,
                    "fill_fraction": float(mask.mean()),
                }
            )
            continue
        if mask_path.exists() or overlay_path.exists():
            raise SystemExit(f"refusing to overwrite existing silhouette for {name}")

        record = images[name]
        camera = reconstruction.cameras[record.camera_id]
        pose = record.cam_from_world
        if callable(pose):
            pose = pose()
        matrix = np.asarray(pose.matrix(), dtype=np.float64)
        rotation = matrix[:, :3]
        translation = matrix[:, 3]
        camera_points = points @ rotation.T + translation
        params = np.asarray(camera.params, dtype=np.float64)
        if str(camera.model_name) not in {"OPENCV", "PINHOLE"}:
            raise SystemExit(f"unsupported camera model: {camera.model_name}")
        fx, fy, cx, cy = params[:4]
        intrinsics = np.asarray(
            [
                [fx / args.factor, 0, cx / args.factor],
                [0, fy / args.factor, cy / args.factor],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(rotation)
        distortion = (
            params[4:8] if str(camera.model_name) == "OPENCV" else np.zeros(4)
        )
        projected, _ = cv2.projectPoints(
            points, rvec, translation, intrinsics, distortion
        )
        xy = np.rint(projected[:, 0, :]).astype(np.int32)
        width = camera.width // args.factor
        height = camera.height // args.factor
        silhouette = np.zeros((height, width), dtype=np.uint8)
        valid_faces = faces[np.all(camera_points[faces, 2] > 0, axis=1)]
        for triangle in valid_faces:
            polygon = xy[triangle]
            if (
                polygon[:, 0].max() < 0
                or polygon[:, 0].min() >= width
                or polygon[:, 1].max() < 0
                or polygon[:, 1].min() >= height
            ):
                continue
            cv2.fillConvexPoly(silhouette, polygon, 255)

        source = np.asarray(Image.open(args.images / name).convert("RGB"))
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        overlay = source.copy()
        outside = silhouette == 0
        overlay[outside] = (
            0.4 * overlay[outside] + 0.6 * np.asarray([0, 255, 0])
        ).astype(np.uint8)
        Image.fromarray(silhouette).save(mask_path)
        Image.fromarray(overlay).save(overlay_path, quality=92)
        result = {
            "image": name,
            "fill_fraction": float((silhouette > 0).mean()),
            "width": width,
            "height": height,
        }
        records.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    report = {
        "schema_version": 1,
        "method": "colmap-aligned-triangle-union-silhouette-v1",
        "mesh": str(args.mesh.resolve()),
        "mesh_sha256": _sha256(args.mesh),
        "mesh_vertices": int(len(points)),
        "mesh_faces": int(len(faces)),
        "factor": args.factor,
        "image_count": len(records),
        "records": records,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
