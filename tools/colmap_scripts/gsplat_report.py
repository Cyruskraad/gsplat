"""Convert gsplat validation JSON artifacts into stable CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


_SUMMARY_PATTERN = re.compile(r"val_step(?P<step>\d+)\.json$")
_PER_VIEW_PATTERN = re.compile(r"val_step(?P<step>\d+)_per_view\.json$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fieldnames(rows: Iterable[dict[str, object]], preferred: tuple[str, ...]) -> list[str]:
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _write_csv(path: Path, rows: list[dict[str, object]], preferred: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows, preferred))
        writer.writeheader()
        writer.writerows(rows)


def export_metrics_csv(stats_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Export validation summaries and per-view records found in ``stats_dir``."""

    checkpoints: list[dict[str, object]] = []
    per_view: list[dict[str, object]] = []
    for path in sorted(stats_dir.glob("val_step*.json")):
        per_view_match = _PER_VIEW_PATTERN.fullmatch(path.name)
        if per_view_match:
            step = int(per_view_match.group("step"))
            payload = _read_json(path)
            if not isinstance(payload, list):
                raise ValueError(f"Per-view metrics must be a JSON list: {path}")
            for raw in payload:
                if not isinstance(raw, dict):
                    raise ValueError(f"Per-view metric rows must be JSON objects: {path}")
                row = {"step": step, **raw}
                per_view.append(row)
            continue
        summary_match = _SUMMARY_PATTERN.fullmatch(path.name)
        if summary_match:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Validation metrics must be a JSON object: {path}")
            checkpoints.append({"step": int(summary_match.group("step")), **payload})

    outputs: dict[str, Path] = {}
    if checkpoints:
        checkpoints.sort(key=lambda row: int(row["step"]))
        output = output_dir / "validation_checkpoints.csv"
        _write_csv(output, checkpoints, ("step",))
        outputs["checkpoints"] = output
    if per_view:
        per_view.sort(key=lambda row: (int(row["step"]), str(row.get("image_name", ""))))
        output = output_dir / "validation_per_view.csv"
        _write_csv(output, per_view, ("step", "image_name"))
        outputs["per_view"] = output
    return outputs
