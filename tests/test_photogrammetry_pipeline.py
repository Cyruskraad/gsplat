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
    check_prior_quality,
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


# ---------------------------------------------------------------------------
# AI-prior quality gate
# ---------------------------------------------------------------------------


def test_check_prior_quality_passes_healthy_priors():
    """Ordinary priors produce no problems at all."""
    assert (
        check_prior_quality(
            depth_stats={
                "num_maps": 10,
                "num_degenerate_maps": 0,
                "mean_finite_fraction": 1.0,
            },
            mask_stats={
                "num_masks": 10,
                "mean_excluded_fraction": 0.12,
                "min_kept_fraction": 0.6,
            },
        )
        == []
    )
    # Nothing to judge when neither prior was given.
    assert check_prior_quality() == []


def test_check_prior_quality_flags_empty_directories():
    """A directory that matched no files is the quietest way to waste a run."""
    problems = check_prior_quality(
        depth_stats={"num_maps": 0},
        mask_stats={"num_masks": 0},
    )
    assert len(problems) == 2
    assert any("--mask_dir contains no .png masks" in p for p in problems)
    assert any("--mono_depth_dir contains no .npy" in p for p in problems)


def test_check_prior_quality_flags_over_and_under_aggressive_masks():
    over = check_prior_quality(
        mask_stats={
            "num_masks": 5,
            "mean_excluded_fraction": 0.95,
            "min_kept_fraction": 0.02,
        }
    )
    assert len(over) == 1 and "95.0% of the average frame" in over[0]

    # A segmenter that found nothing is a different bug with the same cause.
    none_excluded = check_prior_quality(
        mask_stats={
            "num_masks": 5,
            "mean_excluded_fraction": 0.0,
            "min_kept_fraction": 1.0,
        }
    )
    assert len(none_excluded) == 1 and "exclude nothing" in none_excluded[0]

    # A fully-excluded frame is flagged even when the average looks fine.
    fully_excluded = check_prior_quality(
        mask_stats={
            "num_masks": 5,
            "mean_excluded_fraction": 0.3,
            "min_kept_fraction": 0.0,
        }
    )
    assert len(fully_excluded) == 1
    assert "excludes its entire frame" in fully_excluded[0]

    # The threshold is honoured, so a legitimately aggressive capture can opt
    # out rather than being forced to disable the gate.
    assert (
        check_prior_quality(
            mask_stats={
                "num_masks": 5,
                "mean_excluded_fraction": 0.95,
                "min_kept_fraction": 0.02,
            },
            max_excluded_fraction=0.99,
        )
        == []
    )


def test_check_prior_quality_flags_degenerate_and_sparse_depth():
    problems = check_prior_quality(
        depth_stats={
            "num_maps": 10,
            "num_degenerate_maps": 8,
            "mean_finite_fraction": 1.0,
        }
    )
    assert len(problems) == 1
    assert "8/10 depth maps (80.0%)" in problems[0]

    # A minority of bad maps is normal and must not trip the gate.
    assert (
        check_prior_quality(
            depth_stats={
                "num_maps": 10,
                "num_degenerate_maps": 1,
                "mean_finite_fraction": 1.0,
            }
        )
        == []
    )

    mostly_nan = check_prior_quality(
        depth_stats={
            "num_maps": 10,
            "num_degenerate_maps": 0,
            "mean_finite_fraction": 0.2,
        }
    )
    assert len(mostly_nan) == 1 and "20.0% finite" in mostly_nan[0]


def test_check_prior_quality_consumes_the_real_metrics_dicts(tmp_path):
    """The gate reads exactly what the metrics functions produce.

    Guards against the gate and the stats functions drifting apart on key
    names -- the failure mode that would make it silently pass everything.
    """
    imageio = pytest.importorskip("imageio.v2", reason="imageio not installed")

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # Both masks exclude the whole frame.
    imageio.imwrite(str(mask_dir / "a.png"), np.zeros((8, 8), dtype=np.uint8))
    imageio.imwrite(str(mask_dir / "b.png"), np.zeros((8, 8), dtype=np.uint8))

    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    np.save(depth_dir / "a.npy", np.full((8, 8), 2.0, dtype=np.float32))
    np.save(depth_dir / "b.npy", np.full((8, 8), np.nan, dtype=np.float32))
    np.save(depth_dir / "c.npy", np.full((8, 8), np.nan, dtype=np.float32))

    problems = check_prior_quality(
        depth_stats=depth_prior_stats(str(depth_dir)),
        mask_stats=mask_coverage_stats(str(mask_dir)),
    )
    joined = " ".join(problems)
    assert "100.0% of the average frame" in joined
    assert "excludes its entire frame" in joined
    assert "3/3 depth maps (100.0%)" in joined
    # 1 of 3 maps is finite -> 33.3%, below the 50% default.
    assert "33.3% finite" in joined


def test_check_prior_quality_thresholds_pass_on_equality():
    """A directory sitting exactly on a threshold is accepted, not flagged."""
    assert (
        check_prior_quality(
            depth_stats={
                "num_maps": 10,
                "num_degenerate_maps": 5,
                "mean_finite_fraction": 0.5,
            },
            mask_stats={
                "num_masks": 10,
                "mean_excluded_fraction": 0.9,
                "min_kept_fraction": 0.1,
            },
        )
        == []
    )


def test_check_prior_quality_problems_are_json_serializable():
    """The stage records these into pipeline_report.json verbatim."""
    problems = check_prior_quality(mask_stats={"num_masks": 0})
    json.dumps({"problems": problems, "num_problems": len(problems)})


def test_run_pipeline_writes_the_report_even_when_a_stage_fails(tmp_path):
    """A failed run must still leave a `pipeline_report.json` behind.

    `run_stage` carefully records a failure as `status="failed"`, but without
    `--continue_on_error` it then re-raises -- so if the report were only
    written at the end of a successful run, that record would be lost to the
    traceback and any report from an earlier run would stay on disk still
    claiming success. Driven through `run_pipeline.py` as a subprocess, the
    way a user hits it.
    """
    pytest.importorskip("tyro", reason="tyro not installed")
    imageio = pytest.importorskip("imageio.v2", reason="imageio not installed")
    import subprocess
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(repo_root, "examples", "run_pipeline.py")

    data_dir = tmp_path / "data"
    mask_dir = data_dir / "masks"
    mask_dir.mkdir(parents=True)
    # Masks that exclude every pixel of every frame: the `priors` gate fails
    # this under --strict.
    for name in ("a", "b"):
        imageio.imwrite(str(mask_dir / f"{name}.png"), np.zeros((8, 8), dtype=np.uint8))

    result_dir = tmp_path / "out"
    report_path = result_dir / "pipeline_report.json"
    # Leave a stale success report behind, so a run that never wrote its own
    # would silently leave this one in place.
    result_dir.mkdir()
    report_path.write_text(json.dumps({"stages": [{"name": "stale", "status": "ok"}]}))

    proc = subprocess.run(
        [
            sys.executable,
            script,
            "--stages",
            "priors",
            "--strict",
            "--data_dir",
            str(data_dir),
            "--result_dir",
            str(result_dir),
            "--mask_dir",
            str(mask_dir),
        ],
        capture_output=True,
        text=True,
        # The script imports `gsplat`; a source checkout isn't necessarily
        # installed, and `python examples/run_pipeline.py` only puts
        # `examples/` on sys.path.
        env={**os.environ, "PYTHONPATH": repo_root},
    )

    assert proc.returncode != 0, proc.stdout + proc.stderr
    report = json.loads(report_path.read_text())
    stages = {s["name"]: s for s in report["stages"]}
    assert "stale" not in stages, "the stale report was left in place"
    assert stages["priors"]["status"] == STATUS_FAILED
    assert stages["priors"]["metrics"]["num_problems"] > 0
    assert "excludes its entire frame" in " ".join(
        stages["priors"]["metrics"]["problems"]
    )
