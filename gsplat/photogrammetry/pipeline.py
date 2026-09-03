# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Stage orchestration and reporting for the photogrammetry pipeline.

Each stage of the pipeline (bundle adjustment, dense MVS, training, mesh
extraction, ...) is independently useful and independently runnable, but a
real capture goes through all of them in order, and the interesting question
is how the *whole* run went: which stages ran, how long each took, and what
each one measured. This module provides that layer:

- :class:`StageResult` / :class:`PipelineReport` -- a stable, JSON-serializable
  record of one run: per-stage status, wall-clock duration, declared outputs
  and metrics, plus a printable summary table.
- :func:`run_stage` -- a context manager that times a stage, captures failures
  as ``status="failed"`` (rather than losing the report to a traceback), and
  lets the stage attach its own metrics/outputs.
- :func:`collect_artifact_metrics` -- reads the ``stats/*.json`` files the
  individual stage CLIs already write (``bundle_adjust_stats.json``,
  ``dense_stats.json``, ``mesh_metrics.json``, the trainer's
  ``stats/val_step*.json``/``stats/mesh_step*.json``) into one dict, so the
  end-to-end runner (``examples/run_pipeline.py``) and the after-the-fact
  summarizer (``examples/summarize_photogrammetry_stats.py``) agree on one
  schema instead of each inventing their own.

Everything here is pure Python + stdlib: no torch, no CUDA, no COLMAP. That
keeps the orchestration layer importable (and unit-testable) anywhere, with
the heavy stages invoked as subprocesses by the runner.
"""

import glob
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

# Bumped when the on-disk report schema changes incompatibly.
REPORT_SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass
class StageResult:
    """The outcome of one pipeline stage."""

    name: str
    status: str = STATUS_OK
    duration_s: float = 0.0
    # Human-readable note (why a stage was skipped, what it produced, ...).
    message: str = ""
    # Files/directories this stage produced.
    outputs: List[str] = field(default_factory=list)
    # Quantitative results for this stage (see gsplat.photogrammetry.metrics).
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Exception text, when status == "failed".
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            "message": self.message,
            "outputs": list(self.outputs),
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class PipelineReport:
    """An ordered record of the stages in one pipeline run."""

    stages: List[StageResult] = field(default_factory=list)
    # Free-form run context (data_dir, result_dir, CLI config, ...).
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: StageResult) -> StageResult:
        self.stages.append(result)
        return result

    def get(self, name: str) -> Optional[StageResult]:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    @property
    def total_duration_s(self) -> float:
        return sum(stage.duration_s for stage in self.stages)

    @property
    def failed(self) -> List[StageResult]:
        return [s for s in self.stages if s.status == STATUS_FAILED]

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for stage in self.stages:
            counts[stage.status] = counts.get(stage.status, 0) + 1
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "context": self.context,
            "summary": {
                "num_stages": len(self.stages),
                "status_counts": counts,
                "total_duration_s": round(self.total_duration_s, 3),
                "succeeded": not self.failed,
            },
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def write(self, path: str) -> str:
        """Write the report as JSON to *path*, creating parent dirs."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def format_table(self) -> str:
        """A fixed-width summary table of stages, timings and key metrics."""
        rows = [("STAGE", "STATUS", "TIME (s)", "KEY METRICS")]
        for stage in self.stages:
            rows.append(
                (
                    stage.name,
                    stage.status,
                    f"{stage.duration_s:.1f}",
                    _format_key_metrics(stage),
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        lines = []
        for i, row in enumerate(rows):
            lines.append(
                f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  "
                f"{row[2]:>{widths[2]}}  {row[3]}".rstrip()
            )
            if i == 0:
                lines.append("-" * max(len(line) for line in lines))
        lines.append(
            f"total: {self.total_duration_s:.1f}s over {len(self.stages)} stage(s)"
            + ("" if not self.failed else f", {len(self.failed)} FAILED")
        )
        return "\n".join(lines)


def _format_key_metrics(stage: StageResult, max_items: int = 4) -> str:
    """Pick a few scalar metrics from *stage* for the summary table."""
    if stage.status == STATUS_FAILED:
        return stage.error.splitlines()[0][:80] if stage.error else "failed"
    parts = []
    for key, value in stage.metrics.items():
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        if len(parts) >= max_items:
            break
    if not parts:
        return stage.message[:80]
    return ", ".join(parts)


@contextmanager
def run_stage(
    report: PipelineReport,
    name: str,
    reraise: bool = True,
) -> Iterator[StageResult]:
    """Run one stage, recording its timing, status and metrics in *report*.

    The stage body attaches its own results to the yielded
    :class:`StageResult` (``result.metrics``, ``result.outputs``,
    ``result.message``). Exceptions are recorded as ``status="failed"`` with
    the error text, so a partial report still gets written; set
    ``reraise=False`` to keep the pipeline going after a failed stage.

    To record a stage that didn't run at all, use :func:`record_skipped`
    instead -- a context manager cannot skip its own body, so the decision
    stays explicit at the call site.

    Args:
        report: The report to append this stage's result to.
        name: Stage name (e.g. ``"bundle_adjust"``).
        reraise: Whether to re-raise an exception raised by the stage body
            after recording it (default True -- fail fast).

    Yields:
        The :class:`StageResult` for this stage, to be filled in by the body.
    """
    result = StageResult(name=name)
    report.add(result)

    start = time.perf_counter()
    try:
        yield result
    except Exception as e:  # noqa: BLE001 -- recorded, optionally re-raised
        result.duration_s = time.perf_counter() - start
        result.status = STATUS_FAILED
        result.error = f"{type(e).__name__}: {e}"
        if reraise:
            raise
        return
    result.duration_s = time.perf_counter() - start


def record_skipped(report: PipelineReport, name: str, reason: str) -> StageResult:
    """Record a stage that didn't run (disabled, unavailable, or up to date).

    A skipped stage still appears in the report -- "we didn't run dense MVS
    because the colmap CLI isn't installed" is exactly the kind of thing a
    pipeline report should say out loud rather than silently omit.
    """
    return report.add(StageResult(name=name, status=STATUS_SKIPPED, message=reason))


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def collect_artifact_metrics(
    result_dir: Optional[str], data_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Collect the ``stats/*.json`` files the stage CLIs write.

    The stages don't share one naming scheme, because they run at different
    scopes: bundle adjustment and dense MVS run once per *dataset* (writing
    next to the dataset, under ``data_dir``) while training and mesh
    extraction run once per *training run* (writing under ``result_dir``).
    This walks both roots for the known filenames.

    Args:
        result_dir: A trainer/``extract_mesh.py`` output directory.
        data_dir: The dataset root ``bundle_adjust.py``/``dense_mvs.py`` ran
            against. Optional -- omit if those stages weren't run, or wrote
            under ``result_dir`` instead.

    Returns:
        ``{category: {path: stats}}`` for the categories ``render_quality``,
        ``mesh_quality``, ``dense_point_cloud`` and ``bundle_adjustment``,
        omitting categories with no files found.
    """
    categories = {
        "render_quality": [(result_dir, "val_step*.json")],
        "mesh_quality": [
            (result_dir, "mesh_step*.json"),
            (result_dir, "mesh_metrics.json"),
        ],
        "dense_point_cloud": [
            (result_dir, "dense_stats.json"),
            (data_dir, "dense_stats.json"),
        ],
        "bundle_adjustment": [
            (result_dir, "bundle_adjust_stats.json"),
            (data_dir, "bundle_adjust_stats.json"),
        ],
    }

    collected: Dict[str, Dict[str, Any]] = {}
    for category, sources in categories.items():
        entries: Dict[str, Any] = {}
        for root, pattern in sources:
            for path in _find(root, pattern):
                entries[path] = _load_json(path)
        if entries:
            collected[category] = entries
    return collected


def _find(root: Optional[str], pattern: str) -> List[str]:
    if root is None or not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def latest_metrics(entries: Dict[str, Any]) -> Optional[Any]:
    """The last entry of a :func:`collect_artifact_metrics` category.

    Paths sort lexicographically, and the step-numbered filenames the trainer
    writes (``val_step0007000.json``) are zero-padded, so the last one is the
    newest.
    """
    if not entries:
        return None
    return entries[sorted(entries)[-1]]


def check_prior_quality(
    depth_stats: Optional[Dict[str, Any]] = None,
    mask_stats: Optional[Dict[str, Any]] = None,
    max_excluded_fraction: float = 0.9,
    max_degenerate_fraction: float = 0.5,
    min_finite_fraction: float = 0.5,
) -> List[str]:
    """Turn AI-prior sanity stats into concrete problems worth stopping for.

    :func:`gsplat.photogrammetry.metrics.mask_coverage_stats` and
    :func:`~gsplat.photogrammetry.metrics.depth_prior_stats` only *describe* a
    prior directory. This judges those numbers, so a pipeline can refuse to
    spend hours training on priors that cannot help -- an empty directory, a
    segmenter that masked away the whole scene (or nothing at all), depth maps
    that are constant or mostly NaN.

    Every check is on the *directory as a whole*: a single odd frame is normal,
    a directory-wide pattern is a setup mistake.

    Args:
        depth_stats: A ``depth_prior_stats(...)`` dict, or None if
            ``--mono_depth_dir`` wasn't given.
        mask_stats: A ``mask_coverage_stats(...)`` dict, or None if
            ``--mask_dir`` wasn't given.
        max_excluded_fraction: Flag masks excluding more than this fraction of
            the average frame.
        max_degenerate_fraction: Flag a depth directory where more than this
            fraction of maps are constant or entirely non-finite. (Maps that
            aren't loadable at all are always flagged, however few.)
        min_finite_fraction: Flag depth maps whose average finite-pixel
            fraction falls below this.

    Returns:
        A list of human-readable problem descriptions, in a stable order.
        Empty means the priors look usable.
    """
    problems: List[str] = []

    if mask_stats is not None:
        num_masks = int(mask_stats.get("num_masks", 0))
        if num_masks == 0:
            problems.append(
                "--mask_dir contains no .png masks: training would silently "
                "run with no transient-object masking at all."
            )
        else:
            excluded = float(mask_stats.get("mean_excluded_fraction", 0.0))
            if excluded > max_excluded_fraction:
                problems.append(
                    f"masks exclude {excluded:.1%} of the average frame "
                    f"(> {max_excluded_fraction:.1%}): little photometric "
                    "signal would be left to train on."
                )
            elif excluded <= 0.0:
                problems.append(
                    "masks exclude nothing (every pixel is kept in every "
                    "mask): --mask_dir is a no-op as generated."
                )
            if float(mask_stats.get("min_kept_fraction", 1.0)) <= 0.0:
                problems.append(
                    "at least one mask excludes its entire frame: that image "
                    "contributes nothing to the photometric loss."
                )

    if depth_stats is not None:
        num_maps = int(depth_stats.get("num_maps", 0))
        if num_maps == 0:
            problems.append(
                "--mono_depth_dir contains no .npy depth maps: training would "
                "silently run with no depth-prior supervision at all."
            )
        else:
            degenerate = int(depth_stats.get("num_degenerate_maps", 0))
            degenerate_fraction = degenerate / num_maps
            if degenerate_fraction > max_degenerate_fraction:
                problems.append(
                    f"{degenerate}/{num_maps} depth maps "
                    f"({degenerate_fraction:.1%}) are constant or entirely "
                    "non-finite, and carry no gradient for the depth loss."
                )
            not_2d = int(depth_stats.get("num_not_2d_maps", 0))
            if not_2d:
                problems.append(
                    f"{not_2d}/{num_maps} depth maps are not a single (H, W) "
                    "array and cannot be loaded (np.squeeze the model's output "
                    "before saving it)."
                )
            finite = float(depth_stats.get("mean_finite_fraction", 1.0))
            if finite < min_finite_fraction:
                problems.append(
                    f"depth maps are only {finite:.1%} finite on average "
                    f"(< {min_finite_fraction:.1%}): most pixels would drop "
                    "out of the depth loss."
                )

    return problems


def _stage_metrics(report: "PipelineReport", name: str) -> Dict[str, Any]:
    """A stage's metrics, or ``{}`` if it didn't run (or wasn't selected)."""
    stage = report.get(name)
    if stage is None or stage.status != STATUS_OK:
        return {}
    return stage.metrics or {}


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """``numerator / denominator``, or None if either is missing or unusable."""
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0.0:
        return None
    return num / den


def _derive_cross_stage(
    sfm: Dict[str, Any],
    ba: Dict[str, Any],
    dense: Dict[str, Any],
    mesh: Dict[str, Any],
) -> Dict[str, float]:
    """Shared derivation behind :func:`derive_cross_stage_metrics` and
    :func:`cross_stage_metrics_from_artifacts`, so the pipeline runner and the
    standalone summarizer can never disagree on what a derived metric means.

    Metrics that only exist by comparing two stages' results.

    Each stage's own metrics answer "what did this stage produce?". These
    answer "did it actually improve on, or agree with, what came before?" --
    which is the question a photogrammetry run is really being judged on, and
    which no single stage can see. They also make several numbers
    *interpretable*: a cloud-to-mesh distance in raw scene units means nothing
    on its own, but divided by the point cloud's own sample spacing it says
    whether the mesh fits within the evidence's noise floor.

    Every entry is omitted rather than guessed when either input stage is
    missing, skipped or failed, so a partial run yields a partial (never
    misleading) set.

    Returns:
        A dict of derived scalars, in a stable order:

        - ``reprojection_error_reduction`` -- fraction of the input model's
          mean reprojection error that bundle adjustment removed. 0.1 means a
          10% improvement; a negative value means it made the fit worse.
        - ``points_retained_after_bundle_adjust`` -- refined 3D points over
          input 3D points. Well under 1.0 means bundle adjustment discarded a
          lot of the sparse cloud.
        - ``densification_ratio`` -- dense MVS points over sparse SfM points.
        - ``mesh_fit_over_point_spacing`` -- mean cloud-to-mesh distance over
          the dense cloud's mean k-NN spacing. **The headline end-to-end
          number**: at or below ~1 the mesh tracks the point cloud to within its
          own sampling noise; well above 1 the mesh genuinely misses geometry
          the cloud captured.
        - ``mesh_edge_over_point_spacing`` -- mean mesh edge length over that
          same spacing. Much below 1 means the mesh is tessellated finer than
          the evidence supports (``--voxel_size`` too small); much above 1
          means it is throwing away detail the cloud has.
    """
    derived: Dict[str, float] = {}

    before = ba.get("mean_reprojection_error_before")
    after = ba.get("mean_reprojection_error_after")
    if before is not None and after is not None:
        reduction = _ratio(float(before) - float(after), before)
        if reduction is not None:
            derived["reprojection_error_reduction"] = reduction

    retained = _ratio(ba.get("num_points3D"), sfm.get("num_points3D"))
    if retained is not None:
        derived["points_retained_after_bundle_adjust"] = retained

    densification = _ratio(dense.get("num_points"), sfm.get("num_points3D"))
    if densification is not None:
        derived["densification_ratio"] = densification

    spacing = dense.get("mean_knn_distance")
    point_to_mesh = mesh.get("point_to_mesh") or {}
    fit = _ratio(point_to_mesh.get("mean"), spacing)
    if fit is not None:
        derived["mesh_fit_over_point_spacing"] = fit

    edge = _ratio(mesh.get("mean_edge_length"), spacing)
    if edge is not None:
        derived["mesh_edge_over_point_spacing"] = edge

    return derived


def derive_cross_stage_metrics(report: "PipelineReport") -> Dict[str, float]:
    """Cross-stage metrics for a :class:`PipelineReport` (see
    :func:`_derive_cross_stage` for what each one means).

    Reads each stage's recorded metrics, ignoring stages that were skipped or
    failed, so a partial run yields a partial rather than a misleading set.
    """
    return _derive_cross_stage(
        _stage_metrics(report, "sfm_input"),
        _stage_metrics(report, "bundle_adjust"),
        _stage_metrics(report, "dense_mvs"),
        _stage_metrics(report, "extract_mesh"),
    )


def cross_stage_metrics_from_artifacts(
    collected: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Cross-stage metrics for a :func:`collect_artifact_metrics` result.

    The same derivation as :func:`derive_cross_stage_metrics`, for a sequence
    of stages run by hand rather than through ``run_pipeline.py``. Only the
    ``stats/*.json`` files those stages wrote are available here -- there is no
    ``sfm_input`` baseline, since nothing writes one -- so the metrics keyed on
    the input SfM model are simply absent.
    """
    return _derive_cross_stage(
        {},
        latest_metrics(collected.get("bundle_adjustment", {})) or {},
        latest_metrics(collected.get("dense_point_cloud", {})) or {},
        latest_metrics(collected.get("mesh_quality", {})) or {},
    )


def format_cross_stage_metrics(derived: Dict[str, float]) -> str:
    """A short human-readable block for :func:`derive_cross_stage_metrics`."""
    if not derived:
        return "cross-stage metrics: none (needs at least two completed stages)"
    width = max(len(k) for k in derived)
    lines = ["CROSS-STAGE METRICS"]
    lines.append("-" * max(len(lines[0]), width + 12))
    for key, value in derived.items():
        lines.append(f"{key:<{width}}  {value:.4g}")
    return "\n".join(lines)
