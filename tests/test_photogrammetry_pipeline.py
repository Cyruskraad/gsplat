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

"""Tests for gsplat.photogrammetry.pipeline (stage orchestration/reporting)
and the per-stage metrics that don't need open3d -- `track_stats`,
`mask_coverage_stats`, `depth_prior_stats` and `reconstruction_stats`. The
open3d-dependent geometry metrics live in tests/test_photogrammetry_metrics.py;
keeping these here means they still run in an environment without the
optional `gsplat[mesh]` extra installed.
"""

import json
import os

import numpy as np
import pytest

from gsplat.photogrammetry.metrics import (
    depth_prior_stats,
    mask_coverage_stats,
    track_stats,
)
from gsplat.photogrammetry.pipeline import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    PipelineReport,
    StageResult,
    collect_artifact_metrics,
    latest_metrics,
    record_skipped,
    run_stage,
)

# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


def test_run_stage_records_metrics_and_duration():
    report = PipelineReport()
    with run_stage(report, "bundle_adjust") as stage:
        stage.metrics = {"mean_reprojection_error_after": 0.41}
        stage.outputs = ["sparse/refined"]

    result = report.get("bundle_adjust")
    assert result.status == STATUS_OK
    assert result.metrics["mean_reprojection_error_after"] == 0.41
    assert result.outputs == ["sparse/refined"]
    assert result.duration_s >= 0.0


def test_run_stage_records_failure_and_reraises_by_default():
    report = PipelineReport()
    with pytest.raises(RuntimeError, match="boom"):
        with run_stage(report, "train"):
            raise RuntimeError("boom")

    result = report.get("train")
    assert result.status == STATUS_FAILED
    assert "RuntimeError: boom" in result.error
    assert report.failed == [result]


def test_run_stage_can_continue_after_failure():
    """With reraise=False the failure is recorded but the pipeline goes on --
    the whole point of writing a report even for a partially failed run.
    """
    report = PipelineReport()
    with run_stage(report, "train", reraise=False):
        raise RuntimeError("CUDA out of memory")
    with run_stage(report, "extract_mesh") as stage:
        stage.metrics = {"num_triangles": 12}

    assert report.get("train").status == STATUS_FAILED
    assert report.get("extract_mesh").status == STATUS_OK
    assert report.to_dict()["summary"]["succeeded"] is False


def test_record_skipped_keeps_the_stage_in_the_report():
    report = PipelineReport()
    record_skipped(report, "dense_mvs", "no colmap CLI on PATH")

    result = report.get("dense_mvs")
    assert result.status == STATUS_SKIPPED
    assert result.message == "no colmap CLI on PATH"
    assert result.metrics == {}
    assert result.duration_s == 0.0
    # A skipped stage is not a failure.
    assert report.to_dict()["summary"]["succeeded"] is True


def test_report_to_dict_is_json_serializable_and_counts_statuses():
    report = PipelineReport(context={"data_dir": "data/x"})
    with run_stage(report, "sfm_input") as stage:
        stage.metrics = {"num_images": 3}
    record_skipped(report, "dense_mvs", "not selected")
    with run_stage(report, "train", reraise=False):
        raise ValueError("nope")

    payload = report.to_dict()
    json.dumps(payload)  # must not raise

    assert payload["schema_version"] >= 1
    assert payload["context"]["data_dir"] == "data/x"
    assert payload["summary"]["status_counts"] == {"ok": 1, "skipped": 1, "failed": 1}
    assert payload["summary"]["num_stages"] == 3
    assert [s["name"] for s in payload["stages"]] == [
        "sfm_input",
        "dense_mvs",
        "train",
    ]


def test_report_write_creates_parent_dirs(tmp_path):
    report = PipelineReport()
    with run_stage(report, "sfm_input") as stage:
        stage.metrics = {"num_images": 2}

    path = report.write(str(tmp_path / "nested" / "dir" / "pipeline_report.json"))
    assert os.path.exists(path)
    with open(path) as f:
        assert json.load(f)["stages"][0]["name"] == "sfm_input"


def test_format_table_lists_every_stage_and_the_failure_reason():
    report = PipelineReport()
    with run_stage(report, "sfm_input") as stage:
        stage.metrics = {"num_images": 5}
    record_skipped(report, "dense_mvs", "no colmap CLI on PATH")
    with run_stage(report, "train", reraise=False):
        raise RuntimeError("CUDA out of memory")

    table = report.format_table()
    for name in ("sfm_input", "dense_mvs", "train"):
        assert name in table
    assert "num_images=5" in table
    assert "no colmap CLI on PATH" in table
    assert "CUDA out of memory" in table
    assert "FAILED" in table


def test_total_duration_sums_stages():
    report = PipelineReport()
    report.add(StageResult(name="a", duration_s=1.5))
    report.add(StageResult(name="b", duration_s=2.25))
    assert report.total_duration_s == pytest.approx(3.75)


# ---------------------------------------------------------------------------
# Artifact-stats collection (shared by run_pipeline.py and the summarizer)
# ---------------------------------------------------------------------------


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def test_collect_artifact_metrics_spans_result_and_data_dirs(tmp_path):
    result_dir = str(tmp_path / "results")
    data_dir = str(tmp_path / "data")
    _write_json(f"{result_dir}/stats/val_step0007000.json", {"psnr": 24.0})
    _write_json(f"{result_dir}/stats/val_step0030000.json", {"psnr": 28.9})
    _write_json(f"{result_dir}/mesh_metrics.json", {"is_watertight": True})
    _write_json(f"{data_dir}/dense/dense_stats.json", {"num_points": 900000})
    _write_json(
        f"{data_dir}/sparse/refined/bundle_adjust_stats.json", {"num_images": 185}
    )

    collected = collect_artifact_metrics(result_dir, data_dir)
    assert sorted(collected) == [
        "bundle_adjustment",
        "dense_point_cloud",
        "mesh_quality",
        "render_quality",
    ]
    # Newest step wins -- filenames are zero-padded, so lexical order is
    # step order.
    assert latest_metrics(collected["render_quality"])["psnr"] == 28.9
    assert latest_metrics(collected["bundle_adjustment"])["num_images"] == 185


def test_collect_artifact_metrics_tolerates_missing_dirs(tmp_path):
    assert collect_artifact_metrics(str(tmp_path / "nope"), None) == {}
    assert collect_artifact_metrics(None, None) == {}
    assert latest_metrics({}) is None


# ---------------------------------------------------------------------------
# Per-stage metrics that need no optional dependencies
# ---------------------------------------------------------------------------


def test_track_stats_counts_multi_view_tracks():
    tracks = [
        [(0, (1.0, 2.0)), (1, (3.0, 4.0)), (2, (5.0, 6.0))],  # length 3
        [(0, (1.0, 2.0)), (1, (3.0, 4.0))],  # length 2
        [(3, (7.0, 8.0))],  # length 1 (no constraint)
    ]
    stats = track_stats(tracks)
    assert stats["num_tracks"] == 3
    assert stats["num_observations"] == 6
    assert stats["mean_track_length"] == pytest.approx(2.0)
    assert stats["max_track_length"] == 3
    assert stats["multi_view_track_fraction"] == pytest.approx(2 / 3)


def test_track_stats_empty():
    stats = track_stats([])
    assert stats["num_tracks"] == 0
    assert stats["multi_view_track_fraction"] == 0.0


def test_mask_coverage_stats(tmp_path):
    imageio = pytest.importorskip("imageio.v2", reason="imageio not installed")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # Two masks keeping 75% and 50% of the frame respectively.
    m1 = np.full((10, 20), 255, dtype=np.uint8)
    m1[:, :5] = 0
    m2 = np.full((10, 20), 255, dtype=np.uint8)
    m2[:, :10] = 0
    imageio.imwrite(str(mask_dir / "a.png"), m1)
    imageio.imwrite(str(mask_dir / "b.png"), m2)

    stats = mask_coverage_stats(str(mask_dir))
    assert stats["num_masks"] == 2
    assert stats["mean_kept_fraction"] == pytest.approx(0.625)
    assert stats["min_kept_fraction"] == pytest.approx(0.5)
    assert stats["max_kept_fraction"] == pytest.approx(0.75)
    assert stats["mean_excluded_fraction"] == pytest.approx(0.375)


def test_mask_coverage_stats_empty_dir(tmp_path):
    stats = mask_coverage_stats(str(tmp_path))
    assert stats["num_masks"] == 0
    assert stats["mean_kept_fraction"] == 0.0


def test_depth_prior_stats_flags_degenerate_maps(tmp_path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    rng = np.random.default_rng(0)
    np.save(
        depth_dir / "good.npy", rng.uniform(1.0, 5.0, size=(8, 8)).astype(np.float32)
    )
    # Constant map: no gradient for a correlation loss.
    np.save(depth_dir / "constant.npy", np.full((8, 8), 2.0, dtype=np.float32))
    # Entirely non-finite map.
    np.save(depth_dir / "nan.npy", np.full((8, 8), np.nan, dtype=np.float32))

    stats = depth_prior_stats(str(depth_dir))
    assert stats["num_maps"] == 3
    assert stats["num_degenerate_maps"] == 2
    assert stats["mean_finite_fraction"] == pytest.approx(2 / 3)
    assert 1.0 <= stats["min_value"] <= stats["max_value"] <= 5.0


def test_reconstruction_stats_on_synthetic_model(tmp_path):
    """`reconstruction_stats` should report the same counts pycolmap does for
    a model built with a known number of images/points/observations.
    """
    pytest.importorskip("pycolmap", reason="pycolmap not installed")
    import sys

    # `test_colmap_dataset` owns the synthetic-reconstruction fixture builder
    # and itself imports `datasets.colmap`, so both directories go on the path.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (here, os.path.join(here, "..", "examples")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from test_colmap_dataset import _build_synthetic_reconstruction

    from gsplat.photogrammetry.metrics import reconstruction_stats

    out_dir = str(tmp_path / "sparse")
    recon = _build_synthetic_reconstruction(out_dir, num_cameras=4, num_points=25)

    stats = reconstruction_stats(out_dir)
    assert stats["num_images"] == recon.num_images()
    assert stats["num_points3D"] == recon.num_points3D()
    assert stats["num_observations"] == recon.compute_num_observations()
    assert stats["num_cameras"] == 1
    assert stats["mean_track_length"] > 1.0
    # The fixture projects points exactly, so reprojection error is ~0.
    assert stats["mean_reprojection_error"] == pytest.approx(0.0, abs=1e-6)

    # Passing an already-loaded Reconstruction gives the same answer.
    assert reconstruction_stats(recon) == stats
