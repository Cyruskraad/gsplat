# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera poses from photographs, with COLMAP as the engine.

Nothing here is new. Structure-from-motion is solved well enough by COLMAP
that the only defensible thing for a relighting project to do is call it:

* **Features**: ALIKED (rotation-robust ``N16ROT`` variant), matched with
  **LightGlue** -- the learned pair that replaced SIFT + ratio test as the
  default in current pipelines. SIFT is the fallback, and is used
  automatically when the learned models cannot be loaded (they are fetched on
  first use, and an offline workstation cannot).
* **Mapping**: **global** SfM (GLOMAP, merged into COLMAP) by default. On an
  object orbit it is an order of magnitude faster than incremental mapping and
  at least as accurate; ``mapper="incremental"`` is there for the captures
  where rotation averaging struggles.
* **One camera**: every shot of a capture comes from one body and one lens,
  and sharing the intrinsics makes them far better constrained than
  per-image estimates. The loader assumes it too.

The only decisions this module makes itself are which model to keep (the
one with the most registered images) and what to report.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

__all__ = [
    "SfMConfig",
    "SfMResult",
    "run_sfm",
    "require_pycolmap",
    "learned_features_available",
]

log = logging.getLogger(__name__)


def require_pycolmap():
    try:
        import pycolmap
    except ImportError as error:
        raise ImportError(
            "structure-from-motion needs pycolmap: pip install 'atlas-relight[capture]'"
        ) from error
    return pycolmap


_PROBE = """
import sys, tempfile, pathlib
import numpy as np
import pycolmap
root = pathlib.Path(tempfile.mkdtemp())
(root / "img").mkdir()
rng = np.random.default_rng(0)
import zlib, struct
def png(path, a):
    raw = b"".join(b"\\x00" + row.tobytes() for row in a)
    chunk = lambda k, d: struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d))
    path.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
png(root / "img" / "a.png", (rng.random((64, 64)) * 255).astype(np.uint8))
options = pycolmap.FeatureExtractionOptions()
options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
pycolmap.extract_features(root / "db.db", root / "img", extraction_options=options, device=pycolmap.Device.cpu)
"""

_LEARNED: Dict[str, bool] = {}


def learned_features_available() -> bool:
    """Whether this pycolmap can run ALIKED/LightGlue.

    Wheels built without ONNX Runtime *abort the process* (a C++ exception
    escaping a worker thread) rather than raising, so the capability cannot be
    probed with ``try``. It is probed once in a child process instead, and the
    answer cached for the life of this one.
    """
    if "aliked" not in _LEARNED:
        import subprocess
        import sys

        try:
            result = subprocess.run(
                [sys.executable, "-c", _PROBE], capture_output=True, timeout=600
            )
            _LEARNED["aliked"] = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _LEARNED["aliked"] = False
    return _LEARNED["aliked"]


@dataclass(frozen=True)
class SfMConfig:
    #: ``"aliked"`` or ``"sift"``.
    features: str = "aliked"
    #: ``"lightglue"`` or ``"bruteforce"``.
    matcher: str = "lightglue"
    #: ``"exhaustive"`` (every pair; right up to a few hundred images) or
    #: ``"sequential"`` (neighbours in filename order, for long video-like runs).
    pairing: str = "exhaustive"
    #: ``"global"`` (GLOMAP) or ``"incremental"``.
    mapper: str = "global"
    #: A COLMAP camera model with distortion; undistortion happens on export.
    camera_model: str = "OPENCV"
    single_camera: bool = True
    #: Longest side features are extracted at. Poses are unaffected by it.
    max_image_size: int = 3200
    #: ``"auto"``, ``"cuda"`` or ``"cpu"``.
    device: str = "auto"
    #: Retry with SIFT when the learned features cannot be loaded.
    fallback_to_sift: bool = True
    #: Retry with incremental mapping when global mapping registers fewer than
    #: this share of the images.
    min_registered_fraction: float = 0.8
    seed: int = 0


@dataclass
class SfMResult:
    reconstruction: Any
    database_path: Path
    config: SfMConfig
    num_images: int
    features_used: str
    mapper_used: str
    notes: list = field(default_factory=list)

    @property
    def num_registered(self) -> int:
        return int(self.reconstruction.num_reg_images())

    def report(self) -> Dict[str, Any]:
        rec = self.reconstruction
        return {
            "num_images": self.num_images,
            "num_registered": self.num_registered,
            "num_points3D": int(rec.num_points3D()),
            "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
            "mean_track_length": float(rec.compute_mean_track_length()),
            "features": self.features_used,
            "mapper": self.mapper_used,
            "config": asdict(self.config),
            "notes": list(self.notes),
        }


def _extract(pycolmap, database: Path, image_dir: Path, names, config, kind: str):
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = config.max_image_size
    if kind == "aliked":
        options.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    else:
        options.type = pycolmap.FeatureExtractorType.SIFT
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = config.camera_model
    pycolmap.extract_features(
        database,
        image_dir,
        image_names=list(names),
        camera_mode=(
            pycolmap.CameraMode.SINGLE
            if config.single_camera
            else pycolmap.CameraMode.AUTO
        ),
        reader_options=reader,
        extraction_options=options,
        device=getattr(pycolmap.Device, config.device),
    )


def _match(pycolmap, database: Path, config, kind: str, matcher_kind: str) -> None:
    options = pycolmap.FeatureMatchingOptions()
    matcher = {
        ("aliked", "lightglue"): "ALIKED_LIGHTGLUE",
        ("aliked", "bruteforce"): "ALIKED_BRUTEFORCE",
        ("sift", "lightglue"): "SIFT_LIGHTGLUE",
        ("sift", "bruteforce"): "SIFT_BRUTEFORCE",
    }[(kind, matcher_kind)]
    options.type = getattr(pycolmap.FeatureMatcherType, matcher)
    device = getattr(pycolmap.Device, config.device)
    if config.pairing == "exhaustive":
        pycolmap.match_exhaustive(database, matching_options=options, device=device)
    elif config.pairing == "sequential":
        pycolmap.match_sequential(database, matching_options=options, device=device)
    else:
        raise ValueError(f"unknown pairing {config.pairing!r}")


def _map(pycolmap, database: Path, image_dir: Path, out: Path, config, which: str):
    out.mkdir(parents=True, exist_ok=True)
    if which == "global":
        options = pycolmap.GlobalPipelineOptions()
        options.random_seed = config.seed
        models = pycolmap.global_mapping(database, image_dir, out, options)
    elif which == "incremental":
        options = pycolmap.IncrementalPipelineOptions()
        options.random_seed = config.seed
        models = pycolmap.incremental_mapping(database, image_dir, out, options)
    else:
        raise ValueError(f"unknown mapper {which!r}")
    if not models:
        return None
    return max(models.values(), key=lambda rec: rec.num_reg_images())


def run_sfm(
    image_dir: Path | str,
    work_dir: Path | str,
    config: Optional[SfMConfig] = None,
    *,
    image_names: Optional[Sequence[str]] = None,
) -> SfMResult:
    """Poses and a sparse point cloud for the images in ``image_dir``.

    Args:
        image_dir: 8-bit display-referred images. Feature detectors are trained
            on those, not on linear radiance.
        work_dir: Receives ``database.db`` and ``sparse/``. Replaced if present,
            so a rerun never matches against a stale database.
        config: See :class:`SfMConfig`.
        image_names: Restrict to these files (relative to ``image_dir``).

    Returns:
        The largest reconstruction, with a report of how it was obtained.
    """
    pycolmap = require_pycolmap()
    config = config or SfMConfig()
    if config.features not in ("aliked", "sift"):
        raise ValueError(
            f"features must be 'aliked' or 'sift', got {config.features!r}"
        )
    if config.matcher not in ("lightglue", "bruteforce"):
        raise ValueError(
            f"matcher must be 'lightglue' or 'bruteforce', got {config.matcher!r}"
        )
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    names = sorted(image_names or [p.name for p in image_dir.iterdir() if p.is_file()])
    if len(names) < 3:
        raise ValueError(
            f"structure-from-motion needs at least 3 images, got {len(names)}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    nonlocal_matcher = [config.matcher]

    def attempt(kind: str) -> Path:
        if kind == "sift" and config.matcher == "lightglue":
            # SIFT+LightGlue needs the same ONNX runtime ALIKED does.
            nonlocal_matcher[0] = (
                "lightglue" if learned_features_available() else "bruteforce"
            )
        database = work_dir / f"database_{kind}.db"
        if database.exists():
            database.unlink()
        _extract(pycolmap, database, image_dir, names, config, kind)
        _match(pycolmap, database, config, kind, nonlocal_matcher[0])
        return database

    kind = config.features
    if kind == "aliked" and not learned_features_available():
        if not config.fallback_to_sift:
            raise RuntimeError(
                "ALIKED/LightGlue are unavailable in this pycolmap build (no ONNX "
                "Runtime, or the models could not be fetched); install a build "
                "with ONNX support or set features='sift'"
            )
        notes.append("ALIKED/LightGlue unavailable in this pycolmap build; used SIFT")
        log.warning(notes[-1])
        kind = "sift"
    try:
        database = attempt(kind)
    except Exception as error:  # e.g. model download failed after the probe
        if kind == "sift" or not config.fallback_to_sift:
            raise
        notes.append(f"{kind} features failed ({error}); fell back to SIFT")
        log.warning(notes[-1])
        kind = "sift"
        database = attempt(kind)

    sparse = work_dir / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    which = config.mapper
    reconstruction = _map(pycolmap, database, image_dir, sparse / which, config, which)
    registered = 0 if reconstruction is None else reconstruction.num_reg_images()
    if which == "global" and registered < config.min_registered_fraction * len(names):
        notes.append(
            f"global mapping registered {registered}/{len(names)} images; "
            f"retried incrementally"
        )
        log.warning(notes[-1])
        fallback = _map(
            pycolmap, database, image_dir, sparse / "incremental", config, "incremental"
        )
        if fallback is not None and fallback.num_reg_images() > registered:
            reconstruction, which = fallback, "incremental"
    if reconstruction is None:
        raise RuntimeError(
            f"COLMAP could not reconstruct any model from {len(names)} images in "
            f"{image_dir}. Check overlap between neighbouring shots and focus."
        )
    final = sparse / "final"
    final.mkdir(parents=True, exist_ok=True)
    reconstruction.write(final)
    return SfMResult(
        reconstruction=reconstruction,
        database_path=database,
        config=config,
        num_images=len(names),
        features_used=kind,
        mapper_used=which,
        notes=notes,
    )
