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
"""Classic photogrammetry techniques wired to gsplat's SfM (COLMAP) data and
Gaussian Splatting renderers.

This package closes the SfM -> dense MVS -> Gaussian Splatting -> mesh loop:

- :mod:`gsplat.photogrammetry.bundle_adjustment` - differentiable, torch-native
  bundle adjustment that refines COLMAP poses and 3D points by minimizing
  reprojection error over the SfM point tracks.
- :mod:`gsplat.photogrammetry.dense_mvs` - densifies the sparse COLMAP point
  cloud via COLMAP's own patch-match stereo + fusion pipeline.
- :mod:`gsplat.photogrammetry.mesh_extraction` - extracts a cleaned, colored
  triangle mesh from a trained 2DGS/3DGS scene (TSDF fusion of rendered
  depth/normal maps, or Poisson reconstruction from a dense point cloud), with
  texture baking from the training images -- either per-vertex colors or a
  UV-unwrapped texture atlas.
- :mod:`gsplat.photogrammetry.neural_sfm` - imports feed-forward neural-SfM
  output (DUSt3R/MASt3R/VGGT-style, run externally) as a COLMAP model, so it
  becomes a drop-in alternative to COLMAP for the rest of the pipeline.
- :mod:`gsplat.photogrammetry.metrics` - automatic quality metrics for every
  stage above (reconstruction/track quality, mesh quality, cloud-to-mesh fit,
  point-cloud density, AI-prior coverage), written to ``stats/*.json`` files
  alongside the trainers' existing render-quality evaluation.
- :mod:`gsplat.photogrammetry.pipeline` - stage orchestration and reporting,
  turning the stages above into one end-to-end run
  (``examples/run_pipeline.py``) whose per-stage status, wall-clock timing
  and metrics land in a single ``pipeline_report.json``.

``mesh_extraction``/``metrics`` require the optional ``open3d`` dependency
(``pip install gsplat[mesh]``); ``bundle_adjustment``/``neural_sfm`` and
``metrics.reconstruction_stats`` require ``pycolmap`` (already required by
the example COLMAP data loader); ``neural_sfm``'s point-merging step and
``metrics.point_cloud_stats`` also require ``scikit-learn``; ``dense_mvs``
requires the ``colmap`` command-line tool (built with CUDA support) on
``PATH``. ``pipeline`` is pure stdlib and always importable.
"""

from .bundle_adjustment import refine_reconstruction
from .dense_mvs import run_dense_mvs
from .mesh_extraction import (
    bake_mesh_texture,
    bake_texture,
    bake_texture_atlas,
    extract_mesh_poisson,
    extract_mesh_tsdf,
)
from .metrics import (
    depth_prior_stats,
    mask_coverage_stats,
    mesh_quality_stats,
    point_cloud_stats,
    point_to_mesh_distance,
    reconstruction_stats,
    track_stats,
)
from .neural_sfm import merge_point_maps_to_tracks, write_colmap_reconstruction
from .pipeline import (
    PipelineReport,
    StageResult,
    check_prior_quality,
    collect_artifact_metrics,
    record_skipped,
    run_stage,
)

__all__ = [
    "refine_reconstruction",
    "run_dense_mvs",
    "extract_mesh_tsdf",
    "extract_mesh_poisson",
    "bake_texture",
    "bake_texture_atlas",
    "bake_mesh_texture",
    "merge_point_maps_to_tracks",
    "write_colmap_reconstruction",
    "point_to_mesh_distance",
    "mesh_quality_stats",
    "point_cloud_stats",
    "reconstruction_stats",
    "track_stats",
    "mask_coverage_stats",
    "depth_prior_stats",
    "PipelineReport",
    "StageResult",
    "run_stage",
    "record_skipped",
    "collect_artifact_metrics",
    "check_prior_quality",
]
