"""Prompt derivation and candidate selection for SAM object masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .utils import PipelineError


@dataclass(frozen=True)
class SamPrompts:
    points: np.ndarray
    labels: np.ndarray
    box: np.ndarray


def mask_evidence_is_valid(
    *, seed_recall: float, mesh_recall: float | None
) -> bool:
    if seed_recall >= 0.85:
        return True
    return bool(
        mesh_recall is not None and seed_recall >= 0.70 and mesh_recall >= 0.80
    )


def refine_sam_with_mesh_envelope(
    sam_mask: np.ndarray,
    mesh_mask: np.ndarray,
    *,
    envelope_fraction: float = 0.02,
) -> np.ndarray:
    sam = np.asarray(sam_mask, dtype=bool)
    mesh = np.asarray(mesh_mask, dtype=bool)
    if sam.ndim != 2 or mesh.shape != sam.shape:
        raise PipelineError("SAM and mesh masks must have the same two-dimensional shape")
    if not 0 < envelope_fraction < 0.5:
        raise PipelineError("Mesh envelope fraction must be between zero and 0.5")
    radius = max(1, round(min(sam.shape) * envelope_fraction))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    envelope = cv2.dilate(mesh.astype(np.uint8), kernel).astype(bool)
    return sam & envelope


def blue_seed_mask(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise PipelineError("SAM prompt image must be RGB")
    hsv = cv2.cvtColor(image_rgb[..., :3], cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    return (
        (hue >= 95)
        & (hue <= 135)
        & (saturation >= 75)
        & (value >= 20)
    )


def _sample_mask_points(mask: np.ndarray, count: int) -> np.ndarray:
    coordinates_yx = np.argwhere(mask)
    if len(coordinates_yx) < count:
        raise PipelineError(f"Cannot sample {count} prompts from {len(coordinates_yx)} pixels")
    if len(coordinates_yx) > 50_000:
        sample_indices = np.linspace(0, len(coordinates_yx) - 1, 50_000, dtype=int)
        coordinates_yx = coordinates_yx[sample_indices]
    coordinates = coordinates_yx[:, ::-1].astype(np.float64)
    center = coordinates.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(coordinates - center, axis=1)))]
    nearest = np.linalg.norm(coordinates - coordinates[selected[0]], axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(coordinates - coordinates[index], axis=1))
    return np.rint(coordinates[selected]).astype(np.float32)


def build_sam_prompts(
    image_rgb: np.ndarray,
    seeds: np.ndarray | None = None,
    *,
    positive_count: int = 8,
    negative_count: int = 8,
    padding_fraction: float = 0.1,
    box_override: np.ndarray | None = None,
) -> SamPrompts:
    seeds = blue_seed_mask(image_rgb) if seeds is None else seeds.astype(bool)
    if box_override is None:
        ys, xs = np.nonzero(seeds)
        if len(xs) == 0:
            raise PipelineError("No blue object seeds found for SAM prompting")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad_x = math.ceil((x1 - x0 + 1) * padding_fraction)
        pad_y = math.ceil((y1 - y0 + 1) * padding_fraction)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(image_rgb.shape[1] - 1, x1 + pad_x)
        y1 = min(image_rgb.shape[0] - 1, y1 + pad_y)
    else:
        box = np.asarray(box_override, dtype=np.float64).reshape(-1)
        if len(box) != 4 or not np.isfinite(box).all():
            raise PipelineError("SAM box override must contain four finite coordinates")
        x0 = max(0, int(math.floor(box[0])))
        y0 = max(0, int(math.floor(box[1])))
        x1 = min(image_rgb.shape[1] - 1, int(math.ceil(box[2])))
        y1 = min(image_rgb.shape[0] - 1, int(math.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            raise PipelineError("SAM box override has no image area")

    box_region = np.zeros(seeds.shape, dtype=bool)
    box_region[y0 : y1 + 1, x0 : x1 + 1] = True
    seeds = seeds & box_region
    positives = _sample_mask_points(seeds, positive_count)
    hsv = cv2.cvtColor(image_rgb[..., :3], cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 35) & (hue <= 90) & (saturation >= 50) & (value >= 20)
    bright_support = (saturation <= 45) & (value >= 130)
    dark_shadow = value <= 70
    exclusion_radius = max(1, round(min(image_rgb.shape[:2]) * 0.005))
    kernel = np.ones((2 * exclusion_radius + 1, 2 * exclusion_radius + 1), np.uint8)
    far_from_object = cv2.dilate(seeds.astype(np.uint8), kernel) == 0
    support_candidates = box_region & far_from_object & (bright_support | dark_shadow)
    negative_parts: list[np.ndarray] = []
    selected_mask = np.zeros(seeds.shape, dtype=bool)
    if support_candidates.any():
        support_count = min(
            max(1, negative_count // 2), int(support_candidates.sum())
        )
        support_points = _sample_mask_points(support_candidates, support_count)
        negative_parts.append(support_points)
        selected_mask[
            support_points[:, 1].astype(int), support_points[:, 0].astype(int)
        ] = True
    remaining_count = negative_count - sum(len(part) for part in negative_parts)
    if remaining_count:
        negative_candidates = (
            box_region
            & far_from_object
            & (green | bright_support | dark_shadow)
            & ~selected_mask
        )
        if int(negative_candidates.sum()) < remaining_count:
            negative_candidates = box_region & far_from_object & ~selected_mask
        negative_parts.append(_sample_mask_points(negative_candidates, remaining_count))
    negatives = np.concatenate(negative_parts, axis=0)
    return SamPrompts(
        points=np.concatenate([positives, negatives], axis=0),
        labels=np.concatenate(
            [np.ones(len(positives), dtype=np.int32), np.zeros(len(negatives), dtype=np.int32)]
        ),
        box=np.asarray([x0, y0, x1, y1], dtype=np.float32),
    )


def select_sam_candidate(
    candidates: np.ndarray, model_scores: np.ndarray, seeds: np.ndarray
) -> tuple[np.ndarray, int, dict[str, float]]:
    masks = candidates.astype(bool)
    if masks.ndim != 3 or len(masks) != len(model_scores):
        raise PipelineError("SAM candidates and scores have incompatible shapes")
    seed_pixels = int(seeds.sum())
    if seed_pixels == 0:
        raise PipelineError("Cannot rank SAM candidates without object seeds")
    radius = max(1, round(min(seeds.shape) * 0.005))
    prior = cv2.dilate(
        seeds.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    ).astype(bool)
    reports: list[dict[str, float]] = []
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        seed_recall = float(np.logical_and(mask, seeds).sum() / seed_pixels)
        union = int(np.logical_or(mask, prior).sum())
        prior_iou = float(np.logical_and(mask, prior).sum() / max(union, 1))
        area_ratio = float(area / seed_pixels)
        objective = (
            float(model_scores[index])
            + 1.5 * seed_recall
            + prior_iou
            - 0.15 * math.log(max(area_ratio, 1.0))
        )
        if seed_recall < 0.9:
            objective -= 10.0
        reports.append(
            {
                "model_score": float(model_scores[index]),
                "seed_recall": seed_recall,
                "prior_iou": prior_iou,
                "area_to_seed_ratio": area_ratio,
                "objective": objective,
            }
        )
    selected_index = max(range(len(reports)), key=lambda index: reports[index]["objective"])
    return masks[selected_index], selected_index, reports[selected_index]
