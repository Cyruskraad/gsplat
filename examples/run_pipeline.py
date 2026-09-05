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
"""Run the whole photogrammetry pipeline end to end, with per-stage metrics.

Chains the individual stage scripts into one command, wiring each stage's
output into the next -- bundle adjustment's refined COLMAP model and dense
MVS's fused point cloud both feed the trainer, whose checkpoint feeds mesh
extraction -- and records what every stage did and measured into a single
`pipeline_report.json` (see :mod:`gsplat.photogrammetry.pipeline`).

    python examples/run_pipeline.py \\
        --data_dir data/360_v2/garden --result_dir results/garden_pipeline

Stages run in a fixed order and can be selected individually:

    --stages sfm_input bundle_adjust train extract_mesh

  sfm_input     Baseline stats for the input COLMAP model (no side effects).
  bundle_adjust Refine poses/points        -> <data_dir>/sparse/refined
  dense_mvs     Densify the point cloud    -> <data_dir>/dense/dense.ply
  priors        Gate the AI-assisted inputs (--mono_depth_dir/--mask_dir):
                warns (or, with --strict, stops) on a prior directory that
                would waste the training run.
  train         Train 2DGS on the refined  -> <result_dir>/ckpts, stats/
                poses + dense init
  extract_mesh  TSDF mesh + texture bake   -> <result_dir>/mesh.ply

Heavy stages are invoked as subprocesses of the existing per-stage scripts
(the same way `dense_mvs.py` shells out to `colmap`), so this runner stays
dependency-light and each stage keeps its own CLI as the source of truth for
its options. Stages that need something this machine doesn't have (a CUDA
`colmap` build, a GPU) are recorded as `skipped` with the reason rather than
failing the run, unless `--strict` is set -- which also turns the `priors`
stage's warnings into a failure.
"""

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import tyro

from gsplat.photogrammetry.metrics import (
    depth_prior_stats,
    mask_coverage_stats,
    reconstruction_stats,
)
from gsplat.photogrammetry.pipeline import (
    PipelineReport,
    check_prior_quality,
    collect_artifact_metrics,
    derive_cross_stage_metrics,
    format_cross_stage_metrics,
    latest_metrics,
    record_skipped,
    run_stage,
)

ALL_STAGES = [
    "sfm_input",
    "bundle_adjust",
    "dense_mvs",
    "priors",
    "train",
    "extract_mesh",
]

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # Dataset root directory (<data_dir>/images/ + <data_dir>/sparse/0/).
    data_dir: str = "data/360_v2/garden"
    # Directory for training/mesh outputs and the pipeline report.
    result_dir: str = "results/garden_pipeline"
    # Downsample factor for the dataset.
    data_factor: int = 4
    # Which stages to run, in pipeline order (see the module docstring).
    stages: List[str] = field(default_factory=lambda: list(ALL_STAGES))
    # Bundle-adjustment iterations.
    ba_iters: int = 2000
    # Training steps (passed to simple_trainer_2dgs.py --max_steps).
    max_steps: int = 30_000
    # Maximum image dimension used during dense stereo.
    dense_max_image_size: int = 2000
    # Directory of precomputed monocular depth maps (see
    # docs/photogrammetry.md). Enables --mono_depth_loss when set.
    mono_depth_dir: Optional[str] = None
    # Directory of precomputed transient-object masks (see
    # docs/photogrammetry.md).
    mask_dir: Optional[str] = None
    # How the extracted mesh is textured. "vertex" writes per-vertex colors
    # into mesh.ply; "atlas" UV-unwraps and bakes a texture image, writing
    # mesh.obj + mesh.mtl + mesh_0.png (see docs/photogrammetry.md).
    texture_mode: Literal["vertex", "atlas"] = "vertex"
    # Atlas width/height in texels (--texture_mode atlas only).
    texture_size: int = 2048
    # Texture each face from a single chosen view instead of blending every
    # view that sees it -- sharper, but pointwise less accurate, and it
    # requires --texture_mode atlas. See examples/extract_mesh.py.
    texture_view_selection: bool = False
    # Move each vertex along its normal to fit the photographs before
    # texturing (Vu et al., TPAMI 2012). Needs no GPU. Composes with
    # --photometric_align: cameras first, then the surface.
    # See examples/extract_mesh.py.
    refine_mesh: bool = False
    # Maximum depth to integrate during TSDF fusion, in scene units. Unset
    # derives it from the scene's own extent (it was hardcoded at 10.0 and not
    # reachable at all). See examples/extract_mesh.py.
    depth_trunc: Optional[float] = None
    # Solve for the texture whose reprojection best explains every view (a
    # MAP deconvolution modelling the camera PSF) instead of blending them.
    # Requires --texture_mode atlas. See examples/extract_mesh.py.
    texture_super_resolve: bool = False
    # Deliver an existing mesh instead of reconstructing one: cull, decimate,
    # texture and map the .obj/.ply at this path. The extract_mesh stage then
    # needs no checkpoint and no GPU, so the delivery half of the pipeline runs
    # on a machine that cannot train. See examples/extract_mesh.py.
    mesh_path: Optional[str] = None
    # Refine the camera poses photometrically against the extracted mesh
    # before texturing (Zhou & Koltun, SIGGRAPH 2014). Addresses the *cause* of
    # the blur/ghosting the texturing options work around. Needs no GPU.
    # See examples/extract_mesh.py.
    photometric_align: bool = False
    # Image-pyramid levels for --photometric_align.
    photometric_align_levels: int = 3
    # Optimiser steps per bake/optimise round for --photometric_align.
    photometric_align_iters: int = 60
    # Remove faces no training camera ever saw before decimating and texturing.
    # TSDF fusion returns a closed surface, so it invents the underside and the
    # unvisited back of the subject. See examples/extract_mesh.py.
    cull_unobserved: bool = False
    # Split the atlas across this many pages (--texture_mode atlas only).
    texture_pages: int = 1
    # Size the atlas from the evidence instead of --texture_size: texels per
    # source pixel covering the surface. See examples/extract_mesh.py.
    texture_texels_per_pixel: Optional[float] = None
    # Robust multi-view fusion: discard observations this many standard
    # deviations from a point's own mean colour. 0 disables it.
    texture_outlier_sigma: float = 0.0
    # Decimate to a cloud-to-mesh fit target, in units of the reference cloud's
    # own k-NN spacing, rather than to a triangle count.
    target_fit_ratio: Optional[float] = None
    # Decimate to roughly this many triangles. Mutually exclusive with
    # --target_fit_ratio.
    target_triangles: Optional[int] = None
    # Bake the pre-decimation mesh's normals into a normal map on the shipped
    # mesh's atlas. Requires --texture_mode atlas.
    normal_map: bool = False
    # Bits per channel in the normal map. 8 cannot resolve a normal deviation
    # finer than 2/255; 16 drops that floor to 3.1e-5.
    normal_map_bits: Literal[8, 16] = 8
    # Bake an ambient-occlusion map onto the same atlas. Requires atlas mode.
    ao_map: bool = False
    # Extra flags appended verbatim to the extract_mesh stage's command, for
    # options this runner does not name. Each stage's own CLI stays the source
    # of truth for its options (see the module docstring), so this is the
    # escape hatch that keeps a new extract_mesh flag from being unreachable
    # here until someone remembers to mirror it.
    #
    # Bind the FIRST element with "=", or the parser reads the leading "--" as
    # a new option of its own and rejects it:
    #     --extract_mesh_extra_args=--texture_seam_smoothness 0.25
    extract_mesh_extra_args: List[str] = field(default_factory=list)
    # Torch device for the GPU stages.
    device: str = "cuda"
    # Print each stage's command without running anything.
    dry_run: bool = False
    # Treat an unavailable stage (missing colmap CLI, missing checkpoint), or
    # an unusable AI-prior directory, as a failure instead of skipping/warning.
    strict: bool = False
    # `priors` gate: flag masks that exclude more than this fraction of the
    # average frame. Raise it if the capture really is mostly transient.
    max_excluded_fraction: float = 0.9
    # `priors` gate: flag a --mono_depth_dir where more than this fraction of
    # depth maps are constant or entirely non-finite.
    max_degenerate_fraction: float = 0.5
    # Keep going after a stage fails, instead of stopping at the first one.
    continue_on_error: bool = False


def _sparse_dir(data_dir: str) -> str:
    candidate = os.path.join(data_dir, "sparse/0")
    return candidate if os.path.exists(candidate) else os.path.join(data_dir, "sparse")


def _run(cmd: List[str], dry_run: bool) -> None:
    """Run a stage subprocess, raising on a non-zero exit code."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _latest_checkpoint(result_dir: str) -> Optional[str]:
    """Newest `ckpt_<step>.pt` under <result_dir>/ckpts, by step number."""
    paths = glob.glob(os.path.join(result_dir, "ckpts", "ckpt_*.pt"))
    if not paths:
        return None

    def step_of(path: str) -> int:
        stem = os.path.splitext(os.path.basename(path))[0]  # ckpt_<step>[_rankN]
        try:
            return int(stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(paths, key=step_of)


def _run_stages(cfg: Config, report: PipelineReport, selected: List[str]) -> None:
    """Run the selected stages, recording each one into ``report``.

    May raise: without ``--continue_on_error`` a failing stage re-raises after
    being recorded. :func:`main` writes the report either way.
    """
    reraise = not cfg.continue_on_error

    # The COLMAP model the downstream stages read: the refined one once
    # bundle adjustment has produced it, otherwise the dataset's own.
    colmap_dir = _sparse_dir(cfg.data_dir)
    refined_dir = os.path.join(cfg.data_dir, "sparse/refined")
    dense_dir = os.path.join(cfg.data_dir, "dense")
    dense_ply: Optional[str] = None

    # -- Stage: input SfM baseline -------------------------------------
    if "sfm_input" in selected:
        with run_stage(report, "sfm_input", reraise=reraise) as stage:
            stage.metrics = reconstruction_stats(colmap_dir)
            stage.outputs = [colmap_dir]
            stage.message = f"input COLMAP model at {colmap_dir}"
    else:
        record_skipped(report, "sfm_input", "not selected")

    # -- Stage: bundle adjustment --------------------------------------
    if "bundle_adjust" in selected:
        with run_stage(report, "bundle_adjust", reraise=reraise) as stage:
            _run(
                [
                    sys.executable,
                    os.path.join(EXAMPLES_DIR, "bundle_adjust.py"),
                    "--data_dir",
                    cfg.data_dir,
                    "--output_subdir",
                    "sparse/refined",
                    "--num_iters",
                    str(cfg.ba_iters),
                    "--device",
                    cfg.device,
                ],
                cfg.dry_run,
            )
            if not cfg.dry_run:
                stage.metrics = reconstruction_stats(refined_dir)
                ba_stats = latest_metrics(
                    collect_artifact_metrics(refined_dir).get("bundle_adjustment", {})
                )
                if ba_stats:
                    stage.metrics.update(
                        {
                            "mean_reprojection_error_before": ba_stats[
                                "mean_reprojection_error_before"
                            ],
                            "mean_reprojection_error_after": ba_stats[
                                "mean_reprojection_error_after"
                            ],
                        }
                    )
            # Downstream stages read the refined model from here on (set
            # even under --dry_run, so the printed commands show the real
            # wiring rather than the un-refined input model).
            colmap_dir = refined_dir
            stage.outputs = [refined_dir]
    else:
        record_skipped(report, "bundle_adjust", "not selected")
        if os.path.isdir(refined_dir):
            # A previous run already refined this dataset -- keep using it.
            colmap_dir = refined_dir

    # -- Stage: dense MVS ----------------------------------------------
    if "dense_mvs" in selected:
        colmap_cli = shutil.which("colmap")
        if colmap_cli is None and not cfg.strict and not cfg.dry_run:
            record_skipped(
                report,
                "dense_mvs",
                "no `colmap` CLI on PATH (needs a CUDA-enabled COLMAP build)",
            )
        else:
            with run_stage(report, "dense_mvs", reraise=reraise) as stage:
                _run(
                    [
                        sys.executable,
                        os.path.join(EXAMPLES_DIR, "dense_mvs.py"),
                        "--data_dir",
                        cfg.data_dir,
                        "--colmap_dir",
                        colmap_dir,
                        "--output_dir",
                        dense_dir,
                        "--max_image_size",
                        str(cfg.dense_max_image_size),
                    ],
                    cfg.dry_run,
                )
                if not cfg.dry_run:
                    stage.metrics = (
                        latest_metrics(
                            collect_artifact_metrics(dense_dir).get(
                                "dense_point_cloud", {}
                            )
                        )
                        or {}
                    )
                # As above: set even under --dry_run so the training command
                # printed below shows the dense initialization it would use.
                dense_ply = os.path.join(dense_dir, "dense.ply")
                stage.outputs = [dense_ply]
    else:
        record_skipped(report, "dense_mvs", "not selected")
        candidate = os.path.join(dense_dir, "dense.ply")
        if os.path.exists(candidate):
            dense_ply = candidate

    # -- Stage: AI-assisted priors (depth maps / transient masks) -------
    if "priors" in selected:
        if cfg.mono_depth_dir is None and cfg.mask_dir is None:
            record_skipped(
                report, "priors", "no --mono_depth_dir / --mask_dir provided"
            )
        else:
            with run_stage(report, "priors", reraise=reraise) as stage:
                metrics = {}
                if cfg.mono_depth_dir is not None:
                    metrics["mono_depth"] = depth_prior_stats(cfg.mono_depth_dir)
                if cfg.mask_dir is not None:
                    metrics["masks"] = mask_coverage_stats(cfg.mask_dir)
                # Surface the two headline numbers as scalars so they show up
                # in the summary table, keeping the full dicts alongside.
                stage.metrics = dict(metrics)
                if "masks" in metrics:
                    stage.metrics["mean_excluded_fraction"] = metrics["masks"][
                        "mean_excluded_fraction"
                    ]
                if "mono_depth" in metrics:
                    stage.metrics["num_degenerate_depth_maps"] = metrics["mono_depth"][
                        "num_degenerate_maps"
                    ]

                # This stage exists to catch a bad prior directory *before*
                # the hours-long training stage consumes it, so don't just
                # record the numbers -- judge them, and say so loudly.
                problems = check_prior_quality(
                    depth_stats=metrics.get("mono_depth"),
                    mask_stats=metrics.get("masks"),
                    max_excluded_fraction=cfg.max_excluded_fraction,
                    max_degenerate_fraction=cfg.max_degenerate_fraction,
                )
                stage.metrics["problems"] = problems
                stage.metrics["num_problems"] = len(problems)
                for problem in problems:
                    print(f"[priors] WARNING: {problem}", flush=True)
                if problems and cfg.strict:
                    raise RuntimeError(
                        f"--strict: {len(problems)} unusable AI-prior "
                        "input(s): " + " ".join(problems)
                    )
                if problems:
                    print(
                        f"[priors] {len(problems)} problem(s) found; continuing "
                        "anyway (pass --strict to stop here instead).",
                        flush=True,
                    )
    else:
        record_skipped(report, "priors", "not selected")

    # -- Stage: training -------------------------------------------------
    if "train" in selected:
        with run_stage(report, "train", reraise=reraise) as stage:
            cmd = [
                sys.executable,
                os.path.join(EXAMPLES_DIR, "simple_trainer_2dgs.py"),
                "--data_dir",
                cfg.data_dir,
                "--data_factor",
                str(cfg.data_factor),
                "--result_dir",
                cfg.result_dir,
                "--max_steps",
                str(cfg.max_steps),
                "--disable_viewer",
            ]
            if colmap_dir != _sparse_dir(cfg.data_dir):
                cmd += ["--colmap_dir", colmap_dir]
            if dense_ply is not None:
                cmd += ["--dense_points_path", dense_ply]
            if cfg.mono_depth_dir is not None:
                cmd += ["--mono_depth_loss", "--mono_depth_dir", cfg.mono_depth_dir]
            if cfg.mask_dir is not None:
                cmd += ["--mask_dir", cfg.mask_dir]
            _run(cmd, cfg.dry_run)
            if not cfg.dry_run:
                stage.metrics = (
                    latest_metrics(
                        collect_artifact_metrics(cfg.result_dir).get(
                            "render_quality", {}
                        )
                    )
                    or {}
                )
            stage.outputs = [os.path.join(cfg.result_dir, "ckpts")]
    else:
        record_skipped(report, "train", "not selected")

    # -- Stage: mesh extraction ------------------------------------------
    if "extract_mesh" in selected:
        ckpt = _latest_checkpoint(cfg.result_dir)
        # --mesh_path replaces the reconstruction, so this stage no longer
        # depends on the train stage having produced anything.
        if (
            ckpt is None
            and cfg.mesh_path is None
            and not cfg.strict
            and not cfg.dry_run
        ):
            record_skipped(
                report,
                "extract_mesh",
                f"no checkpoint under {cfg.result_dir}/ckpts (run the train stage first)",
            )
        else:
            with run_stage(report, "extract_mesh", reraise=reraise) as stage:
                cmd = [
                    sys.executable,
                    os.path.join(EXAMPLES_DIR, "extract_mesh.py"),
                ]
                if cfg.mesh_path is not None:
                    # --ckpt and --mesh_path are two different surfaces, and
                    # extract_mesh.py refuses both at once.
                    cmd += ["--mesh_path", cfg.mesh_path]
                else:
                    cmd += ["--ckpt", ckpt or "<checkpoint>"]
                cmd += [
                    "--data_dir",
                    cfg.data_dir,
                    "--data_factor",
                    str(cfg.data_factor),
                    "--result_dir",
                    cfg.result_dir,
                    "--device",
                    cfg.device,
                ]
                if cfg.mask_dir is not None:
                    cmd += ["--mask_dir", cfg.mask_dir]
                cmd += [
                    "--texture_mode",
                    cfg.texture_mode,
                    "--texture_size",
                    str(cfg.texture_size),
                ]
                if cfg.texture_view_selection:
                    cmd += ["--texture_view_selection"]
                if cfg.refine_mesh:
                    cmd += ["--refine_mesh"]
                if cfg.depth_trunc is not None:
                    cmd += ["--depth_trunc", str(cfg.depth_trunc)]
                if cfg.texture_super_resolve:
                    cmd += ["--texture_super_resolve"]
                if cfg.photometric_align:
                    cmd += [
                        "--photometric_align",
                        "--photometric_align_levels",
                        str(cfg.photometric_align_levels),
                        "--photometric_align_iters",
                        str(cfg.photometric_align_iters),
                    ]
                if cfg.cull_unobserved:
                    cmd += ["--cull_unobserved"]
                if cfg.texture_pages > 1:
                    cmd += ["--texture_pages", str(cfg.texture_pages)]
                if cfg.texture_texels_per_pixel is not None:
                    cmd += [
                        "--texture_texels_per_pixel",
                        str(cfg.texture_texels_per_pixel),
                    ]
                if cfg.texture_outlier_sigma > 0:
                    cmd += ["--texture_outlier_sigma", str(cfg.texture_outlier_sigma)]
                if cfg.target_fit_ratio is not None:
                    cmd += ["--target_fit_ratio", str(cfg.target_fit_ratio)]
                if cfg.target_triangles is not None:
                    cmd += ["--target_triangles", str(cfg.target_triangles)]
                if cfg.normal_map:
                    cmd += [
                        "--normal_map",
                        "--normal_map_bits",
                        str(cfg.normal_map_bits),
                    ]
                if cfg.ao_map:
                    cmd += ["--ao_map"]
                cmd += list(cfg.extract_mesh_extra_args)
                _run(cmd, cfg.dry_run)
                if not cfg.dry_run:
                    mesh_stats = (
                        latest_metrics(
                            collect_artifact_metrics(cfg.result_dir).get(
                                "mesh_quality", {}
                            )
                        )
                        or {}
                    )
                    stage.metrics = mesh_stats
                # A UV atlas can't live in a .ply, so extract_mesh.py writes
                # .obj on that path -- report the file it actually produced.
                mesh_name = "mesh.obj" if cfg.texture_mode == "atlas" else "mesh.ply"
                stage.outputs = [os.path.join(cfg.result_dir, mesh_name)]
    else:
        record_skipped(report, "extract_mesh", "not selected")


def main(cfg: Config) -> None:
    unknown = [s for s in cfg.stages if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}. Choose from {ALL_STAGES}.")
    # Always run in canonical pipeline order, whatever order they were given.
    selected = [s for s in ALL_STAGES if s in cfg.stages]

    os.makedirs(cfg.result_dir, exist_ok=True)
    report = PipelineReport(
        context={
            "data_dir": cfg.data_dir,
            "result_dir": cfg.result_dir,
            "data_factor": cfg.data_factor,
            "stages_requested": selected,
            "dry_run": cfg.dry_run,
        }
    )

    # The report is the point of this script, and a failed run is exactly when
    # it matters most -- so write it even when a stage raises. Otherwise the
    # `status="failed"` record `run_stage` just built is lost to the traceback,
    # and any `pipeline_report.json` left from an earlier run stays on disk
    # still claiming success.
    try:
        _run_stages(cfg, report, selected)
    finally:
        report.context["artifact_metrics"] = collect_artifact_metrics(
            cfg.result_dir, cfg.data_dir
        )
        # Derived last, from whichever stages actually completed, so a partial
        # or failed run still gets whatever comparisons its stages support.
        cross_stage = derive_cross_stage_metrics(report)
        report.context["cross_stage_metrics"] = cross_stage
        report_path = report.write(os.path.join(cfg.result_dir, "pipeline_report.json"))
        print("\n" + report.format_table())
        print("\n" + format_cross_stage_metrics(cross_stage))
        print(f"\n[run_pipeline] wrote {report_path}")

    if report.failed:
        raise SystemExit(
            f"[run_pipeline] {len(report.failed)} stage(s) failed: "
            + ", ".join(s.name for s in report.failed)
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
