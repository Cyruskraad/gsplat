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
"""Geometric-algebra (PGA) geometry and structure-from-motion (experimental).

Rigid motion is represented by *motors* in 3D projective geometric algebra
Cl(3,0,1), where points, lines and planes are grades of one algebra and the
same sandwich product transforms all of them.

- :mod:`gsplat.contrib.ga.algebra`    -- backend adapter and encoding conventions
- :mod:`gsplat.contrib.ga.motor`      -- motor exp/log, composition, and the sandwich
- :mod:`gsplat.contrib.ga.primitives` -- incidence: join, meet, and metric residuals
- :mod:`gsplat.contrib.ga.camera`     -- pinhole cameras with motor extrinsics
- :mod:`gsplat.contrib.ga.camera_opt` -- motor-parameterized pose refinement (claim C2)
- :mod:`gsplat.contrib.ga.sfm`        -- structure-from-motion stages
- :mod:`gsplat.contrib.ga.baseline`   -- the quaternion/se(3) control arm
"""

from gsplat.contrib.ga import algebra, camera, camera_opt, motor, primitives

__all__ = ["algebra", "camera", "camera_opt", "motor", "primitives"]
