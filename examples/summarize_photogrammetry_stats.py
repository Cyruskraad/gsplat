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
"""Summarize the photogrammetry pipeline's `stats/*.json` outputs after the fact.

The counterpart to `examples/run_pipeline.py`: that runs the stages and
records what each one did as it goes, while this reads whatever a finished
(or partially finished, or hand-run) result directory already contains and
prints/writes one summary. Both share
:func:`gsplat.photogrammetry.pipeline.collect_artifact_metrics`, so they
agree on which files count and how they're grouped.

Mirrors `examples/benchmarks/compression/summarize_stats.py`'s read-then-write
convention, generalized because this pipeline's stages don't share one
`stats/{stage}_step*.json` naming scheme: bundle adjustment and dense MVS run
once per *dataset* (writing next to the dataset, under `--data_dir`) while
training/mesh extraction run once per *training run* (writing under
`--result_dir`).

Example:

    python examples/summarize_photogrammetry_stats.py \\
        --result_dir results/garden_2dgs --data_dir data/360_v2/garden

writes `results/garden_2dgs/stats_summary.json` (leaving any
`pipeline_report.json` from `run_pipeline.py` untouched) and prints a summary
table covering, whichever are present, bundle-adjustment reprojection error,
dense point-cloud density, render PSNR/SSIM/LPIPS, and mesh
quality/cloud-to-mesh fit for one run.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import tyro

from gsplat.photogrammetry.pipeline import (
    collect_artifact_metrics,
    cross_stage_metrics_from_artifacts,
    format_cross_stage_metrics,
)


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
    report = collect_artifact_metrics(cfg.result_dir, cfg.data_dir)

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

    # The same derived comparisons run_pipeline.py reports, from the same
    # shared derivation -- so a hand-run sequence is judged identically to an
    # orchestrated one.
    cross_stage = cross_stage_metrics_from_artifacts(report)
    print("\n" + format_cross_stage_metrics(cross_stage))

    os.makedirs(cfg.result_dir, exist_ok=True)
    out_path = os.path.join(cfg.result_dir, "stats_summary.json")
    with open(out_path, "w") as f:
        json.dump(
            {"artifact_metrics": report, "cross_stage_metrics": cross_stage},
            f,
            indent=2,
        )
    print(f"\n[summarize] wrote consolidated summary to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
