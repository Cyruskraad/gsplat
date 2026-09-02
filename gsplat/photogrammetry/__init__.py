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
  vertex-color texture baking from the training images.
- :mod:`gsplat.photogrammetry.neural_sfm` - imports feed-forward neural-SfM
  output (DUSt3R/MASt3R/VGGT-style, run externally) as a COLMAP model, so it
  becomes a drop-in alternative to COLMAP for the rest of the pipeline.
- :mod:`gsplat.photogrammetry.metrics` - automatic quality metrics (mesh
  quality, cloud-to-mesh fit, point-cloud density) for the stages above that
  don't already report stats, written to ``stats/*.json`` files alongside
  the trainers' existing render-quality evaluation.

``mesh_extraction``/``metrics`` require the optional ``open3d`` dependency
(``pip install gsplat[mesh]``); ``bundle_adjustment``/``neural_sfm`` require
``pycolmap`` (already required by the example COLMAP data loader);
``neural_sfm``'s point-merging step and ``metrics.point_cloud_stats`` also
require ``scikit-learn``; ``dense_mvs`` requires the ``colmap``
command-line tool (built with CUDA support) on ``PATH``.
"""

from .bundle_adjustment import refine_reconstruction
from .dense_mvs import run_dense_mvs
from .mesh_extraction import bake_texture, extract_mesh_poisson, extract_mesh_tsdf
from .metrics import mesh_quality_stats, point_cloud_stats, point_to_mesh_distance
from .neural_sfm import merge_point_maps_to_tracks, write_colmap_reconstruction

__all__ = [
    "refine_reconstruction",
    "run_dense_mvs",
    "extract_mesh_tsdf",
    "extract_mesh_poisson",
    "bake_texture",
    "merge_point_maps_to_tracks",
    "write_colmap_reconstruction",
    "point_to_mesh_distance",
    "mesh_quality_stats",
    "point_cloud_stats",
]
