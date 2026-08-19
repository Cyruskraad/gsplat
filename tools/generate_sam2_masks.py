#!/usr/bin/env python3
"""Generate reproducible object masks with blue-derived SAM 2.1 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from colmap_scripts.gsplat_hq import inspect_object_mask
from colmap_scripts.sam_masks import (
    blue_seed_mask,
    build_sam_prompts,
    mask_evidence_is_valid,
    refine_sam_with_mesh_envelope,
    select_sam_candidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay(image: np.ndarray, mask: np.ndarray, maximum_size: int = 1600) -> Image.Image:
    scale = min(1.0, maximum_size / max(image.shape[:2]))
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    preview = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    preview_mask = cv2.resize(
        mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    result = preview.copy()
    result[~preview_mask] = (
        0.4 * result[~preview_mask] + 0.6 * np.asarray([0, 255, 0])
    ).astype(np.uint8)
    return Image.fromarray(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--grounding-model")
    parser.add_argument(
        "--text-prompt",
        action="append",
        default=[],
        help="Grounding DINO text prompt; repeat for fallback descriptions",
    )
    parser.add_argument("--grounding-threshold", type=float, default=0.15)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.15)
    parser.add_argument("--mesh-mask-dir", type=Path)
    parser.add_argument("--mesh-envelope-fraction", type=float, default=0.02)
    return parser


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask.astype(bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _mesh_mask_for_image(directory: Path, image_path: Path) -> Path:
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem.casefold() in {
            image_path.stem.casefold(),
            f"{image_path.stem}-mask".casefold(),
        }
    )
    if len(matches) != 1:
        raise SystemExit(
            f"expected one mesh mask for {image_path.name}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    args = build_parser().parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"image directory missing: {args.images}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM checkpoint missing: {args.checkpoint}")
    if args.mesh_mask_dir is not None and not args.mesh_mask_dir.is_dir():
        raise SystemExit(f"mesh mask directory missing: {args.mesh_mask_dir}")
    mask_dir = args.output / "masks"
    overlay_dir = args.output / "overlays"
    record_dir = args.output / "records"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.model_config, str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)
    grounding_processor = None
    grounding_model = None
    grounding_prompts = args.text_prompt or [
        "a blue winged statue.",
        "a blue figurine.",
        "a fantasy angel demon figurine.",
        "a statue.",
    ]
    if args.grounding_model:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model
        ).to(args.device)
        grounding_model.eval()
    allowed = {name.casefold() for name in args.names}
    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        and (not allowed or path.name.casefold() in allowed)
    )
    if not image_paths:
        raise SystemExit("no matching source images")

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for image_path in image_paths:
        started = time.time()
        mask_path = mask_dir / f"{image_path.stem}.png"
        overlay_path = overlay_dir / f"{image_path.stem}.jpg"
        record_path = record_dir / f"{image_path.stem}.json"
        if mask_path.exists() and args.resume:
            if not overlay_path.is_file():
                raise SystemExit(f"resumed mask has no overlay: {mask_path}")
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["resumed"] = True
                records.append(record)
                if not record.get("valid", False):
                    failures.append(image_path.name)
                continue
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            report = inspect_object_mask(mask)
            image = np.asarray(Image.open(image_path).convert("RGB"))
            seeds = blue_seed_mask(image)
            recovered: dict[str, object] = {
                "image": image_path.name,
                "resumed": True,
                "recovered_after_interruption": True,
                "global_seed_recall": float(
                    np.logical_and(mask, seeds).sum() / max(int(seeds.sum()), 1)
                ),
                **report,
            }
            if args.mesh_mask_dir is not None:
                mesh_path = _mesh_mask_for_image(args.mesh_mask_dir, image_path)
                mesh_mask = np.asarray(Image.open(mesh_path).convert("L")) > 127
                mesh_mask = cv2.resize(
                    mesh_mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                recovered["mesh_fill_fraction"] = float(mesh_mask.mean())
                recovered["mesh_recall"] = float(
                    np.logical_and(mask, mesh_mask).sum()
                    / max(int(mesh_mask.sum()), 1)
                )
            recovered["valid"] = bool(
                0.02 <= report["fill_fraction"] <= 0.65
                and report["largest_component_fraction"] >= 0.98
                and (
                    args.mesh_mask_dir is None
                    or float(recovered["mesh_recall"]) >= 0.80
                )
            )
            record_path.write_text(
                json.dumps(recovered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(recovered)
            if not recovered["valid"]:
                failures.append(image_path.name)
            continue
        if mask_path.exists() or overlay_path.exists():
            raise SystemExit(f"refusing to overwrite existing output for {image_path.name}")

        image_pil = Image.open(image_path).convert("RGB")
        image = np.asarray(image_pil).copy()
        seeds = blue_seed_mask(image)
        detection: dict[str, object] | None = None
        if grounding_model is not None and grounding_processor is not None:
            detections: list[dict[str, object]] = []
            for text_prompt in grounding_prompts:
                inputs = grounding_processor(
                    images=image_pil, text=text_prompt, return_tensors="pt"
                ).to(args.device)
                with torch.inference_mode():
                    grounding_outputs = grounding_model(**inputs)
                grounded = grounding_processor.post_process_grounded_object_detection(
                    grounding_outputs,
                    inputs.input_ids,
                    threshold=args.grounding_threshold,
                    text_threshold=args.grounding_text_threshold,
                    target_sizes=[image_pil.size[::-1]],
                )[0]
                text_labels = grounded.get("text_labels", grounded["labels"])
                for box, score, label in zip(
                    grounded["boxes"], grounded["scores"], text_labels
                ):
                    detections.append(
                        {
                            "prompt": text_prompt,
                            "label": str(label),
                            "score": float(score),
                            "box": [float(value) for value in box],
                        }
                    )
            if not detections:
                failures.append(image_path.name)
                records.append(
                    {
                        "image": image_path.name,
                        "valid": False,
                        "failure": "no grounded object detection",
                    }
                )
                print(json.dumps(records[-1], sort_keys=True), flush=True)
                continue
            detection = max(detections, key=lambda item: float(item["score"]))
            prompts = None
            prediction_box = np.asarray(detection["box"], dtype=np.float32)
        else:
            prompts = build_sam_prompts(image, seeds)
            prediction_box = prompts.box
        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
            predictor.set_image(image)
            candidates, model_scores, _ = predictor.predict(
                point_coords=None if prompts is None else prompts.points,
                point_labels=None if prompts is None else prompts.labels,
                box=prediction_box,
                multimask_output=True,
            )
        if detection is None:
            mask, candidate_index, selection = select_sam_candidate(
                candidates, model_scores, seeds
            )
        else:
            candidate_index = int(np.argmax(model_scores))
            mask = candidates[candidate_index].astype(bool)
            x0, y0, x1, y1 = np.rint(prediction_box).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.shape[1] - 1, x1), min(image.shape[0] - 1, y1)
            box_region = np.zeros(seeds.shape, dtype=bool)
            box_region[y0 : y1 + 1, x0 : x1 + 1] = True
            box_seeds = seeds & box_region
            seed_count = int(box_seeds.sum())
            selection = {
                "model_score": float(model_scores[candidate_index]),
                "seed_recall": float(
                    np.logical_and(mask, box_seeds).sum() / max(seed_count, 1)
                ),
                "area_to_seed_ratio": float(mask.sum() / max(seed_count, 1)),
            }
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ).astype(bool)
        mesh_report: dict[str, float] = {}
        if args.mesh_mask_dir is not None:
            mesh_path = _mesh_mask_for_image(args.mesh_mask_dir, image_path)
            mesh_mask = np.asarray(Image.open(mesh_path).convert("L")) > 127
            mesh_mask = cv2.resize(
                mesh_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            before = int(mask.sum())
            mask = refine_sam_with_mesh_envelope(
                mask,
                mesh_mask,
                envelope_fraction=args.mesh_envelope_fraction,
            )
            mesh_report = {
                "mesh_fill_fraction": float(mesh_mask.mean()),
                "mesh_recall": float(
                    np.logical_and(mask, mesh_mask).sum() / max(int(mesh_mask.sum()), 1)
                ),
                "envelope_removed_fraction": float(
                    (before - int(mask.sum())) / max(before, 1)
                ),
            }
        mask = _largest_component(mask)
        report = inspect_object_mask(mask)
        evidence_valid = mask_evidence_is_valid(
            seed_recall=float(selection["seed_recall"]),
            mesh_recall=mesh_report.get("mesh_recall"),
        )
        valid = bool(
            evidence_valid
            and selection["model_score"] >= 0.70
            and 0.02 <= report["fill_fraction"] <= 0.65
            and report["largest_component_fraction"] >= 0.98
        )
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        _overlay(image, mask).save(overlay_path, quality=92)
        record = {
            "image": image_path.name,
            "candidate_index": candidate_index,
            "valid": valid,
            "seconds": round(time.time() - started, 3),
            "grounding": detection,
            **selection,
            **mesh_report,
            **report,
        }
        records.append(record)
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not valid:
            failures.append(image_path.name)
        print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "schema_version": 1,
        "method": (
            "grounding-dino-sam2.1-mesh-envelope-v1"
            if args.grounding_model
            else "sam2.1-hiera-large-blue-prompts-v1"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model_config": args.model_config,
        "grounding_model": args.grounding_model,
        "grounding_prompts": grounding_prompts if args.grounding_model else [],
        "mesh_mask_dir": (
            str(args.mesh_mask_dir.resolve()) if args.mesh_mask_dir is not None else None
        ),
        "mesh_envelope_fraction": args.mesh_envelope_fraction,
        "image_count": len(records),
        "failures": failures,
        "records": records,
    }
    (args.output / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
