# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

See ``docs/relighting-atlas.md`` for the full design, and ``design.md`` in this
directory for the module contract.

This subpackage is pure PyTorch and does not touch the CUDA backend, so it
imports and runs on a CPU-only machine. It is deliberately not re-exported from
``gsplat/__init__.py``: import it explicitly.

    from gsplat.relight import functional as F
"""

from . import functional

__all__ = ["functional"]
