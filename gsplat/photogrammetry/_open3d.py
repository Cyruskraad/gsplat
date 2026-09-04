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
"""The optional-open3d guard, shared by the modules that need it.

Its own module so that :mod:`~gsplat.photogrammetry.mesh_extraction` (surface
reconstruction) and :mod:`~gsplat.photogrammetry.texturing` (view sampling and
map baking) can each depend on it without depending on each other.
"""


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "gsplat.photogrammetry.mesh_extraction requires open3d. Install "
            "it with `pip install gsplat[mesh]` (or `pip install open3d`)."
        ) from e
    return o3d
