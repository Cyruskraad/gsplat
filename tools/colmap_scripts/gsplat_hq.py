"""Pure planning and evaluation helpers for fixed-light gsplat runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import cv2

from .utils import PipelineError


SPLIT_METHOD = "camera-pose-farthest-point-v1"


@dataclass(frozen=True)
class ViewRecord:
    """A registered view reduced to the pose values needed for splitting."""

    name: str
    camera_center: tuple[float, float, float]
    view_direction: tuple[float, float, float]


@dataclass(frozen=True)
class ValidationRecord:
    """Metrics used to promote a checkpoint or compact model."""

    step: int
    lpips: float
    psnr: float
    alpha_iou: float
    num_gaussians: int
    fps: float | None = None


@dataclass(frozen=True)
class EarlyStopDecision:
    progressed: bool
    should_stop: bool
    evaluations_without_progress: int


class EarlyStopper:
    """Stop when neither perceptual nor pixel quality improves for a window."""

    def __init__(
        self,
        *,
        patience: int = 4,
        lpips_min_delta: float = 0.005,
        psnr_min_delta: float = 0.1,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.lpips_min_delta = lpips_min_delta
        self.psnr_min_delta = psnr_min_delta
        self.best_lpips = math.inf
        self.best_psnr = -math.inf
        self.evaluations_without_progress = 0

    def observe(self, *, lpips: float, psnr: float) -> EarlyStopDecision:
        lpips_progress = lpips <= self.best_lpips - self.lpips_min_delta
        psnr_progress = psnr >= self.best_psnr + self.psnr_min_delta
        first = not math.isfinite(self.best_lpips)
        progressed = first or lpips_progress or psnr_progress
        self.best_lpips = min(self.best_lpips, lpips)
        self.best_psnr = max(self.best_psnr, psnr)
        if progressed:
            self.evaluations_without_progress = 0
        else:
            self.evaluations_without_progress += 1
        return EarlyStopDecision(
            progressed=progressed,
            should_stop=self.evaluations_without_progress >= self.patience,
            evaluations_without_progress=self.evaluations_without_progress,
        )


def _pose_features(views: list[ViewRecord]) -> np.ndarray:
    centers = np.asarray([view.camera_center for view in views], dtype=np.float64)
    directions = np.asarray([view.view_direction for view in views], dtype=np.float64)
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    scene_radius = float(np.max(radii))
    if scene_radius <= 0:
        raise PipelineError("Camera centers must span more than one position")
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(direction_norms <= 0):
        raise PipelineError("View directions must be non-zero")
    return np.concatenate(
        [centered / scene_radius, 0.25 * directions / direction_norms], axis=1
    )


def _farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    if count < 1:
        return []
    selected = [0]
    nearest = np.linalg.norm(features - features[0], axis=1)
    nearest[0] = -math.inf
    while len(selected) < count:
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        distances = np.linalg.norm(features - features[candidate], axis=1)
        nearest = np.minimum(nearest, distances)
        nearest[selected] = -math.inf
    return selected


def create_split_manifest(
    views: Iterable[ViewRecord],
    *,
    validation_count: int = 12,
    test_count: int = 11,
) -> dict[str, object]:
    """Create a deterministic, pose-spread train/validation/test split."""

    ordered = sorted(views, key=lambda view: view.name.casefold())
    names = [view.name for view in ordered]
    if len(names) != len(set(names)):
        raise PipelineError("Registered image names must be unique")
    heldout_count = validation_count + test_count
    if validation_count < 1 or test_count < 1 or len(ordered) <= heldout_count:
        raise PipelineError("Split counts must leave at least one training view")
    selected = _farthest_point_indices(_pose_features(ordered), heldout_count)
    validation_indices = set(selected[0::2][:validation_count])
    test_indices = set(selected[1::2][:test_count])
    # With unequal counts, the alternating slices leave the final selected view
    # for validation, which is the larger split in the production plan.
    for index in selected:
        if len(validation_indices) < validation_count and index not in test_indices:
            validation_indices.add(index)
        elif len(test_indices) < test_count and index not in validation_indices:
            test_indices.add(index)
    train = [name for index, name in enumerate(names) if index not in validation_indices | test_indices]
    validation = [names[index] for index in sorted(validation_indices)]
    test = [names[index] for index in sorted(test_indices)]
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    return {
        "schema_version": 1,
        "selection_method": SPLIT_METHOD,
        "counts": {key: len(value) for key, value in sorted(splits.items())},
        "splits": splits,
    }


def write_split_manifest(
    views: Iterable[ViewRecord],
    output: Path,
    *,
    validation_count: int = 12,
    test_count: int = 11,
) -> Path:
    payload = create_split_manifest(
        views,
        validation_count=validation_count,
        test_count=test_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _relative_stem(name: str) -> str:
    path = PurePosixPath(name)
    return str(path.with_suffix("")).casefold()


def validate_image_mask_contract(
    image_names: Iterable[str], mask_names: Iterable[str]
) -> dict[str, str]:
    images = list(image_names)
    masks = list(mask_names)
    image_by_stem = {_relative_stem(name): name for name in images}
    mask_by_stem = {_relative_stem(name): name for name in masks}
    if len(image_by_stem) != len(images) or len(mask_by_stem) != len(masks):
        raise PipelineError("Image and mask relative stems must be unique")
    missing = sorted(set(image_by_stem) - set(mask_by_stem))
    extra = sorted(set(mask_by_stem) - set(image_by_stem))
    if missing or extra:
        missing_names = [image_by_stem[key] for key in missing]
        extra_names = [mask_by_stem[key] for key in extra]
        raise PipelineError(f"Mask contract mismatch: missing={missing_names} extra={extra_names}")
    return {
        image_by_stem[key]: mask_by_stem[key]
        for key in sorted(image_by_stem)
    }


def padded_foreground_bbox(
    mask: np.ndarray, *, padding_fraction: float = 0.1
) -> tuple[int, int, int, int]:
    """Return an exclusive XYXY crop around foreground pixels."""

    if mask.ndim != 2:
        raise PipelineError("Object mask must be two-dimensional")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise PipelineError("Object mask is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = math.ceil((x1 - x0) * padding_fraction)
    pad_y = math.ceil((y1 - y0) * padding_fraction)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(mask.shape[1], x1 + pad_x),
        min(mask.shape[0], y1 + pad_y),
    )


def inspect_object_mask(mask: np.ndarray) -> dict[str, object]:
    if mask.ndim != 2:
        raise PipelineError("Object mask must be two-dimensional")
    binary = mask.astype(bool)
    foreground_pixels = int(binary.sum())
    if foreground_pixels == 0:
        raise PipelineError("Object mask is empty")
    component_count_with_background, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    component_count = component_count_with_background - 1
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count else np.asarray([], dtype=int)
    largest = int(areas.max()) if len(areas) else 0
    touches_border = bool(
        binary[0, :].any()
        or binary[-1, :].any()
        or binary[:, 0].any()
        or binary[:, -1].any()
    )
    return {
        "foreground_pixels": foreground_pixels,
        "total_pixels": int(binary.size),
        "fill_fraction": foreground_pixels / binary.size,
        "component_count": component_count,
        "largest_component_fraction": largest / foreground_pixels,
        "touches_border": touches_border,
    }


def stage_gsplat_dataset(
    *,
    images_dir: Path,
    sparse_dir: Path,
    masks_dir: Path,
    destination: Path,
) -> Path:
    sources = {
        "images": images_dir.resolve(),
        "sparse": sparse_dir.resolve(),
        "masks": masks_dir.resolve(),
    }
    for label, source in sources.items():
        if not source.is_dir():
            raise PipelineError(f"Gaussian-splatting {label} directory missing: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for label, source in sources.items():
        target = destination / label
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise PipelineError(f"Gaussian-splatting staging path occupied: {target}")
        target.symlink_to(source, target_is_directory=True)
    return destination


def select_best_checkpoint(
    records: Iterable[ValidationRecord], *, minimum_alpha_iou: float = 0.98
) -> ValidationRecord:
    eligible = [record for record in records if record.alpha_iou >= minimum_alpha_iou]
    if not eligible:
        raise PipelineError(
            f"No checkpoint reaches alpha IoU >= {minimum_alpha_iou:.3f}"
        )
    best_lpips = min(record.lpips for record in eligible)
    perceptual_ties = [record for record in eligible if record.lpips <= best_lpips + 0.005]
    best_psnr = max(record.psnr for record in perceptual_ties)
    pixel_ties = [record for record in perceptual_ties if record.psnr >= best_psnr - 0.2]
    return min(pixel_ties, key=lambda record: (record.num_gaussians, record.step))


def compact_candidate_passes(compact: ValidationRecord, hq: ValidationRecord) -> bool:
    quality_ok = compact.psnr >= hq.psnr - 0.3 and compact.lpips <= hq.lpips + 0.02
    gaussian_reduction = 1.0 - compact.num_gaussians / hq.num_gaussians
    fps_gain = (
        compact.fps is not None
        and hq.fps is not None
        and compact.fps >= hq.fps * 1.25
    )
    return quality_ok and (gaussian_reduction >= 0.30 or fps_gain)


def validation_record_to_dict(record: ValidationRecord) -> dict[str, object]:
    return asdict(record)
