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

``mesh_extraction`` requires the optional ``open3d`` dependency
(``pip install gsplat[mesh]``); ``bundle_adjustment`` requires ``pycolmap``
(already required by the example COLMAP data loader); ``dense_mvs`` requires
the ``colmap`` command-line tool (built with CUDA support) on ``PATH``.
"""

from .bundle_adjustment import refine_reconstruction
from .dense_mvs import run_dense_mvs
from .mesh_extraction import bake_texture, extract_mesh_poisson, extract_mesh_tsdf

__all__ = [
    "refine_reconstruction",
    "run_dense_mvs",
    "extract_mesh_tsdf",
    "extract_mesh_poisson",
    "bake_texture",
]
