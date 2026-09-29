# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A folder of flash photographs in, a trainable ATLAS capture out.

::

    photos/  --decode-->  work/linear/*.png   16-bit linear, as decoded
                          work/sfm/*.png      8-bit sRGB, for feature matching
             --COLMAP-->  work/sparse/final   poses, intrinsics, points
             --export-->  images/*.png        16-bit linear, undistorted
                          masks/*.png         optional, BiRefNet via rembg
                          init.ply            3DGS initialisation from SfM
                          transforms.json     the manifest the loader reads
                          prep_report.json    what happened, with warnings

Each stage is a standard tool (LibRaw, COLMAP/GLOMAP, BiRefNet); this module
only connects them and makes the radiometric and metric choices explicit.
The capture is assumed **co-located**: one flash rigidly mounted on the camera,
at ``flash_offset`` in camera axes. That is what every existing dataset here
is, and it is what the loader's coupling report will confirm.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..data.synthetic import viewmat_to_nerf
from ..imageio import read_image, write_image, write_png
from .geometry import (
    flash_positions,
    focus_point,
    normalise_scene,
    remap,
    undistortion_grid,
    write_gaussian_ply,
)
from .images import decode_image, linear_to_srgb, list_images, srgb_to_linear
from .masks import make_masker
from .sfm import SfMConfig, require_pycolmap, run_sfm

__all__ = ["PrepConfig", "prepare_capture", "export_reconstruction"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepConfig:
    #: Longest side of every output image. Full-frame RAW is ~7k pixels; the
    #: relighting model has no use for that and the memory cost is quadratic.
    max_side: Optional[int] = 2000
    sfm: SfMConfig = field(default_factory=SfMConfig)
    #: Flash position in OpenCV camera axes (x right, y down, z forward), in
    #: the unit ``camera_distance`` is in. ``(0, -0.12, 0)``: 12 cm above.
    flash_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Typical camera-to-object distance. Sets the metric scale of the scene.
    camera_distance: float = 1.0
    #: ``"none"`` or ``"rembg"``.
    masks: str = "none"
    mask_model: str = "birefnet-general"
    #: Drop SfM points farther than this many camera distances from the focus
    #: from the initialisation -- background, not object.
    init_radius: float = 1.0
    #: Keep only points seen in at least this many images.
    init_min_track: int = 3
    #: Divide out EXIF ``ISO / N^2`` so shots at different settings share a scale.
    use_exif_exposure: bool = True
    #: Reuse decoded images in ``work/`` when present.
    reuse_decoded: bool = True


# --- stage 1: decode ---------------------------------------------------------


def _decode_all(sources: Sequence[Path], work: Path, config: PrepConfig) -> List[dict]:
    linear_dir, sfm_dir = work / "linear", work / "sfm"
    records = []
    for index, source in enumerate(sources):
        linear_path = linear_dir / f"{source.stem}.png"
        sfm_path = sfm_dir / f"{source.stem}.png"
        info_path = linear_dir / f"{source.stem}.json"
        if config.reuse_decoded and linear_path.is_file() and info_path.is_file():
            records.append(json.loads(info_path.read_text()))
            continue
        decoded = decode_image(source, max_side=config.max_side)
        linear = decoded.linear.clamp(0.0, 1.0)
        write_image(linear_path, linear, bit_depth=16)
        # Features want a display-referred image. A JPEG already is one, so it
        # goes back through the exact inverse of its decode, unchanged. A RAW
        # is exposed so its bright end sits near white -- with the gain capped,
        # so a mostly-black frame is not amplified into noise.
        if decoded.linear_exact:
            level = float(torch.quantile(linear.flatten()[::31], 0.995))
            gain = min(0.9 / max(level, 1e-6), 64.0)
            write_png(sfm_path, linear_to_srgb(linear * gain))
        else:
            write_png(sfm_path, linear_to_srgb(linear))
        record = {
            "source": str(source),
            "stem": source.stem,
            "linear_exact": decoded.linear_exact,
            "exposure": asdict(decoded.exposure),
            "flash_gain": decoded.exposure.flash_gain(),
            "peak": float(linear.max()),
        }
        info_path.write_text(json.dumps(record))
        records.append(record)
        log.info("decoded %d/%d %s", index + 1, len(sources), source.name)
    return records


# --- stage 3: export ---------------------------------------------------------


def _pinhole(pycolmap, camera):
    options = pycolmap.UndistortCameraOptions()
    return pycolmap.undistort_camera(options, camera)


def export_reconstruction(
    reconstruction,
    records: Sequence[dict],
    work: Path,
    out: Path,
    config: PrepConfig,
    *,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write the manifest, undistorted linear frames, masks and ``init.ply``."""
    pycolmap = require_pycolmap()
    notes = notes if notes is not None else []
    by_name = {f"{r['stem']}.png": r for r in records}

    images = sorted(reconstruction.images.values(), key=lambda im: im.name)
    images = [im for im in images if im.has_pose and im.name in by_name]
    if len(images) < 3:
        raise RuntimeError(f"only {len(images)} images have poses; cannot export")
    camera_ids = {im.camera_id for im in images}
    if len(camera_ids) != 1:
        raise RuntimeError(
            f"the reconstruction has {len(camera_ids)} cameras; the loader needs "
            f"one shared camera (SfMConfig.single_camera=True)"
        )
    source_camera = reconstruction.cameras[camera_ids.pop()]
    target_camera = _pinhole(pycolmap, source_camera)
    grid = undistortion_grid(source_camera, target_camera)

    viewmats = torch.stack(
        [
            torch.from_numpy(im.cam_from_world().matrix()).to(torch.float64)
            for im in images
        ]
    )
    viewmats = torch.cat(
        [viewmats, torch.tensor([0, 0, 0, 1.0]).expand(len(images), 1, 4)], dim=1
    )
    rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
    centres = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    forward = rotation[:, 2, :]  # camera +z in world axes
    focus = focus_point(centres, forward)
    viewmats, scale = normalise_scene(viewmats, focus, config.camera_distance)
    lights = flash_positions(viewmats, config.flash_offset)

    # Radiometry: exposure from EXIF, then one global scale so nothing clips.
    gains = [by_name[im.name]["flash_gain"] for im in images]
    if config.use_exif_exposure and all(g is not None for g in gains):
        median = sorted(gains)[len(gains) // 2]
        exposures = [g / median for g in gains]
    else:
        exposures = [1.0] * len(images)
        if config.use_exif_exposure:
            notes.append("EXIF ISO/aperture missing on some shots; exposure left at 1")
    # Stored pixels are decoded values over one global scale; the loader
    # multiplies the scale back and divides the per-shot exposure out.
    storage_scale = max(max(by_name[im.name]["peak"] for im in images), 1e-6)
    if not all(by_name[im.name]["linear_exact"] for im in images):
        notes.append(
            "some frames are display-referred (JPEG/PNG): linearised through the sRGB "
            "curve, which ignores the camera's own tone curve. Use RAW for "
            "radiometric accuracy."
        )

    masker = make_masker(config.masks, config.mask_model)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames = []
    for index, (image, exposure) in enumerate(zip(images, exposures)):
        record = by_name[image.name]
        linear = read_image(work / "linear" / image.name)[..., :3]
        undistorted = remap(linear, grid).clamp_min(0.0)
        rel_image = f"images/{record['stem']}.png"
        write_image(
            out / rel_image, (undistorted / storage_scale).clamp(0, 1), bit_depth=16
        )
        block = {
            "view_index": index,
            "light_index": index,
            "light_position": lights[index].tolist(),
            "light_intensity": [1.0, 1.0, 1.0],
            "reference_distance": config.camera_distance,
            "exposure": exposure,
        }
        if masker is not None:
            rel_mask = f"masks/{record['stem']}.png"
            srgb = linear_to_srgb(
                undistorted
                / float(undistorted.flatten().quantile(0.99).clamp_min(1e-6))
            )
            write_png(out / rel_mask, masker(srgb))
            block["mask_path"] = rel_mask
        frames.append(
            {
                "file_path": rel_image,
                "transform_matrix": viewmat_to_nerf(viewmats[index]).tolist(),
                "atlas": block,
            }
        )

    fx, fy, cx, cy = (float(v) for v in target_camera.params[:4])
    manifest = {
        "camera_model": "PINHOLE",
        "w": int(target_camera.width),
        "h": int(target_camera.height),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames,
        "atlas": {
            "format_version": 1,
            "synthetic": False,
            "colour_space": "linear",
            "scale": storage_scale,
            "ambient": 0.0,
            "flash_mode": "co_located",
            "splits_are_independent": False,
            "flash_offset": list(config.flash_offset),
            "camera_distance": config.camera_distance,
            "num_views": len(frames),
            "num_lights": len(frames),
            "sfm_to_scene_scale": scale,
            "sfm_focus": focus.tolist(),
        },
    }
    (out / "transforms.json").write_text(json.dumps(manifest, indent=1))

    # Initialisation from the SfM points, in the normalised frame.
    xyz, rgb = [], []
    for point in reconstruction.points3D.values():
        if point.track.length() < config.init_min_track:
            continue
        xyz.append(point.xyz)
        rgb.append(point.color)
    num_init = 0
    if xyz:
        import numpy as np

        points = (torch.from_numpy(np.asarray(xyz)) - focus) * scale
        colours = srgb_to_linear(
            torch.from_numpy(np.asarray(rgb, dtype=np.float64)) / 255.0
        )
        keep = points.norm(dim=-1) <= config.init_radius * config.camera_distance
        num_init = int(keep.sum())
        if num_init:
            write_gaussian_ply(out / "init.ply", points[keep], colours[keep])
    if num_init == 0:
        notes.append(
            "no SfM points inside init_radius; train without init.ply "
            "(the random shell init) or raise init_radius"
        )

    return {
        "num_frames": len(frames),
        "num_dropped": len(records) - len(frames),
        "dropped": sorted(set(by_name) - {im.name for im in images}),
        "image_size": [manifest["w"], manifest["h"]],
        "focal_px": [fx, fy],
        "storage_scale": storage_scale,
        "exposure_range": [min(exposures), max(exposures)],
        "num_init_points": num_init,
        "masks": config.masks,
    }


# --- the whole thing ---------------------------------------------------------


def prepare_capture(
    image_dir: Path | str, out_dir: Path | str, config: Optional[PrepConfig] = None
) -> Dict[str, Any]:
    """Run decode -> SfM -> export and write ``prep_report.json``."""
    config = config or PrepConfig()
    image_dir, out = Path(image_dir), Path(out_dir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    timings = {}

    start = time.perf_counter()
    sources = list_images(image_dir)
    records = _decode_all(sources, work, config)
    timings["decode_s"] = time.perf_counter() - start

    start = time.perf_counter()
    sfm = run_sfm(work / "sfm", work, config.sfm)
    timings["sfm_s"] = time.perf_counter() - start

    start = time.perf_counter()
    notes = list(sfm.notes)
    export = export_reconstruction(
        sfm.reconstruction, records, work, out, config, notes=notes
    )
    timings["export_s"] = time.perf_counter() - start

    report = {
        "source": str(image_dir),
        "num_sources": len(sources),
        "config": _jsonable(asdict(config)),
        "sfm": sfm.report(),
        "export": export,
        "timings": timings,
        "notes": notes,
    }
    (out / "prep_report.json").write_text(json.dumps(report, indent=1))
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
