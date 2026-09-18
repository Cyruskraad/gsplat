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

"""Capture data: the synthetic generator now, the real loader when the report
from the real capture arrives."""

from .loader import SET_NAMES, Capture, Frame, FrameSplit, load_capture
from .synthetic import (
    SyntheticConfig,
    fit_lobe_transport,
    generate_capture,
    nerf_to_viewmat,
    viewmat_to_nerf,
)

__all__ = [
    "Capture",
    "Frame",
    "FrameSplit",
    "SET_NAMES",
    "load_capture",
    "SyntheticConfig",
    "fit_lobe_transport",
    "generate_capture",
    "nerf_to_viewmat",
    "viewmat_to_nerf",
]
