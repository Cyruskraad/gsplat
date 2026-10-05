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
"""Skip the geometric-algebra tests when their optional backend is absent.

``gsplat/contrib/ga`` sits behind the ``ga`` extra: ``kingdon`` is its PGA
backend and is not a dependency of gsplat itself. Every module under
``tests/ga`` imports it transitively, so without the extra a plain
``pytest tests/`` does not skip these tests -- it fails during *collection*,
which takes the whole suite down and reports as something with no visible
connection to an optional module not being installed. That is what the CI
workflow does: it installs ``.[examples]``, not ``.[ga]``.

The two cases are deliberately told apart, which is why this is an explicit
try/except rather than ``pytest.importorskip``:

- **absent** (``ModuleNotFoundError``) -- the extra was not installed. Ignore
  this directory and let the rest of the suite run.
- **present but broken** (any other ``ImportError``) -- propagate it. A
  half-installed backend is a real problem and should be loud, not quietly
  skipped.

``importorskip`` happens to draw the same line in pytest 8.2 and later, where
it defaults to catching only ``ModuleNotFoundError``, but it caught every
``ImportError`` before that. Writing the distinction out means this behaves
the same on either.
"""

from __future__ import annotations

collect_ignore_glob: list[str] = []

try:  # noqa: SIM105 - the two failure modes are handled differently
    import kingdon  # noqa: F401
except ModuleNotFoundError:
    # Everything here, including this directory's own helpers.
    collect_ignore_glob.append("*.py")
