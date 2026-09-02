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
"""Aggregate the photogrammetry pipeline's `stats/*.json` outputs -- from
`bundle_adjust.py`, `dense_mvs.py`, `extract_mesh.py`, and
`simple_trainer_2dgs.py`'s `eval()`/`--extract_mesh` -- into one consolidated
report.

Mirrors `examples/benchmarks/compression/summarize_stats.py`'s read-then-write
convention, generalized because this pipeline's stages don't share one
`stats/{stage}_step*.json` naming scheme: bundle adjustment and dense MVS run
once per *dataset* (writing next to the dataset, under `--data_dir`) while
training/mesh extraction run once per *training run* (writing under
`--result_dir`).

Example:

    python examples/summarize_photogrammetry_stats.py \\
        --result_dir results/garden_2dgs --data_dir data/360_v2/garden

writes `results/garden_2dgs/pipeline_report.json` and prints a summary table
covering (whichever of these are present) bundle-adjustment reprojection
error, dense point-cloud density, render PSNR/SSIM/LPIPS, and mesh
quality/cloud-to-mesh fit for one run.
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import tyro


def _find(root: Optional[str], pattern: str):
    if root is None or not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def _load_all(paths) -> Dict[str, dict]:
    stats = {}
    for path in paths:
        with open(path, "r") as f:
            stats[path] = json.load(f)
    return stats


@dataclass
class Config:
    # Directory the trainer / extract_mesh.py wrote stats to, e.g.
    # results/garden_2dgs.
    result_dir: str
    # Dataset root directory bundle_adjust.py/dense_mvs.py were run against,
    # e.g. data/360_v2/garden -- their stats live under
    # data_dir/**/bundle_adjust_stats.json and data_dir/**/dense_stats.json,
    # not under result_dir. Optional: omit if those stages weren't run, or
    # if their stats were written directly under result_dir instead.
    data_dir: Optional[str] = None


def main(cfg: Config) -> None:
    report: Dict[str, Dict[str, dict]] = {}

    render_quality = _load_all(_find(cfg.result_dir, "val_step*.json"))
    if render_quality:
        report["render_quality"] = render_quality

    mesh_quality = _load_all(_find(cfg.result_dir, "mesh_step*.json"))
    mesh_quality.update(_load_all(_find(cfg.result_dir, "mesh_metrics.json")))
    if mesh_quality:
        report["mesh_quality"] = mesh_quality

    dense_point_cloud = _load_all(_find(cfg.result_dir, "dense_stats.json"))
    dense_point_cloud.update(_load_all(_find(cfg.data_dir, "dense_stats.json")))
    if dense_point_cloud:
        report["dense_point_cloud"] = dense_point_cloud

    bundle_adjustment = _load_all(_find(cfg.result_dir, "bundle_adjust_stats.json"))
    bundle_adjustment.update(_load_all(_find(cfg.data_dir, "bundle_adjust_stats.json")))
    if bundle_adjustment:
        report["bundle_adjustment"] = bundle_adjustment

    if not report:
        print(
            "[summarize] found no stats files under "
            f"result_dir={cfg.result_dir!r} / data_dir={cfg.data_dir!r}"
        )

    for stage, entries in report.items():
        print(f"\n=== {stage} ===")
        for path, stats in entries.items():
            print(f"  {path}")
            for k, v in stats.items():
                if isinstance(v, (int, float, bool, str)):
                    print(f"    {k}: {v}")

    os.makedirs(cfg.result_dir, exist_ok=True)
    out_path = os.path.join(cfg.result_dir, "pipeline_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[summarize] wrote consolidated report to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
