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
"""Dense multi-view-stereo (MVS) point cloud densification.

Patch-match multi-view stereo is a mature, GPU-heavy algorithm with no good
pure-Python equivalent, and COLMAP already ships a battle-tested CUDA
implementation of it. Rather than reimplementing it, this module orchestrates
COLMAP's own dense-reconstruction pipeline (``image_undistorter`` ->
``patch_match_stereo`` -> ``stereo_fusion``) via subprocess calls to the
``colmap`` command-line tool, producing a dense, colored point cloud that can
be used to (a) densify the sparse SfM point cloud gsplat initializes Gaussians
from, and (b) provide input geometry for Poisson mesh reconstruction (see
:mod:`gsplat.photogrammetry.mesh_extraction`).
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union


def _colmap_binary() -> str:
    binary = shutil.which("colmap")
    if binary is None:
        raise RuntimeError(
            "Could not find the `colmap` command-line tool on PATH. Dense "
            "multi-view stereo requires a CUDA-enabled COLMAP install (see "
            "https://colmap.github.io/install.html); the `pycolmap` Python "
            "bindings alone are not sufficient, since patch-match stereo is "
            "only exposed through the COLMAP CLI."
        )
    return binary


def _run(cmd: List[str]) -> None:
    print(f"[dense_mvs] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"COLMAP command failed (exit code {e.returncode}): {' '.join(cmd)}. "
            "If this is `patch_match_stereo`, verify COLMAP was built with "
            "CUDA support and a CUDA device is visible to this process."
        ) from e


def run_dense_mvs(
    data_dir: Union[str, Path],
    colmap_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_dir: Optional[Union[str, Path]] = None,
    max_image_size: int = 2000,
) -> str:
    """Run COLMAP's dense reconstruction pipeline and return the fused point cloud path.

    Args:
        data_dir: Dataset root (used to locate ``images/`` if ``image_dir`` is
            not given).
        colmap_dir: Path to the sparse COLMAP model (cameras/images/points3D)
            to undistort and densify — e.g. the output of
            :func:`gsplat.photogrammetry.bundle_adjustment.refine_reconstruction`,
            or the dataset's original ``sparse/0``.
        output_dir: Workspace directory for undistorted images, depth maps,
            and the final fused point cloud (``<output_dir>/dense.ply``).
        image_dir: Directory of (possibly distorted) source images. Defaults
            to ``<data_dir>/images``.
        max_image_size: Maximum image dimension used during dense stereo, to
            bound memory/runtime (COLMAP's ``--PatchMatchStereo.max_image_size``).

    Returns:
        The path to the fused dense point cloud (``<output_dir>/dense.ply``).
    """
    data_dir = Path(data_dir)
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)
    if image_dir is None:
        image_dir = data_dir / "images"
    image_dir = Path(image_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    colmap = _colmap_binary()

    _run(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(colmap_dir),
            "--output_path",
            str(output_dir),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(max_image_size),
        ]
    )
    _run(
        [
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            str(output_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
    )
    dense_ply = output_dir / "dense.ply"
    _run(
        [
            colmap,
            "stereo_fusion",
            "--workspace_path",
            str(output_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(dense_ply),
        ]
    )
    return str(dense_ply)
