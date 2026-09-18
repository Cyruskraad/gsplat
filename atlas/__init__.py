# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""ATLAS -- All-frequency Transport via Learned Atom Splatting.

Relighting whose cost does not depend on how complicated the illumination is.

A primitive's outgoing radiance is made linear in the illumination by
construction, so alpha compositing -- whose weights depend only on geometry and
opacity -- commutes with shading. Splatting the transport and contracting in
screen space is therefore *identical* to contracting per primitive and
splatting, and the renderer can choose whichever grouping suits what changed
since the last frame.

See ``docs/relighting-atlas.md`` for the design and ``docs/module-contract.md``
for the module contract.

Import layout, and it matters:

- ``atlas.functional`` is **pure PyTorch**. It does not import ``gsplat``, does
  not touch CUDA, and its tests run on any machine. Everything the method's
  correctness rests on lives there.
- ``atlas.model``, ``atlas.train`` and ``atlas.render`` import ``gsplat`` and
  need a GPU.

So ``import atlas.functional`` works in a bare virtualenv with only ``torch``
installed, which is what keeps the whole correctness argument checkable without
hardware.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
