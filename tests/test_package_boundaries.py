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

"""The import boundary that keeps the correctness argument checkable.

``atlas.functional`` holds everything the method's correctness rests on. It runs
on any machine with PyTorch, which is what lets the exactness proof, the
superposition property and the calibration gates be checked without a GPU. The
moment something in there reaches for ``gsplat``, that stops being true and the
whole layer becomes untestable on a laptop -- silently, because the developer who
added the import had CUDA.

AGENTS.md states this rule. These tests are why it is a rule and not a wish.
"""

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent


def test_importing_the_functional_layer_does_not_pull_in_gsplat():
    """Run in a fresh interpreter: an already-imported gsplat would mask this."""
    code = (
        "import sys; import atlas.functional; "
        "sys.exit(1 if 'gsplat' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PACKAGE, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "atlas.functional imported gsplat. The functional layer must stay "
        f"importable without it.\n{result.stderr}"
    )


def test_no_module_in_the_functional_layer_mentions_gsplat():
    """Catches a deferred import inside a function, which the check above misses."""
    offenders = []
    for path in sorted((PACKAGE / "atlas" / "functional").glob("*.py")):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "import gsplat" in stripped or "from gsplat" in stripped:
                offenders.append(f"{path.name}:{number}: {stripped}")
    assert not offenders, "gsplat imported inside atlas.functional:\n" + "\n".join(
        offenders
    )


def test_the_functional_layer_imports_nothing_outside_torch_and_the_stdlib():
    """A new third-party dependency here would break the same promise."""
    allowed = {"torch", "atlas"}
    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for path in sorted((PACKAGE / "atlas" / "functional").glob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            root = None
            if stripped.startswith("import "):
                root = stripped[len("import ") :].split()[0].split(".")[0]
            elif stripped.startswith("from ") and " import " in stripped:
                module = stripped[len("from ") :].split(" import ")[0]
                if module.startswith("."):
                    continue  # relative, inside the package
                root = module.split(".")[0]
            if root and root not in allowed and root not in stdlib:
                offenders.append(f"{path.name}:{number}: {stripped}")
    assert not offenders, (
        "atlas.functional gained a dependency beyond torch and the standard "
        "library:\n" + "\n".join(offenders)
    )


def test_the_inspector_runs_without_torch_installed():
    """It is the first thing run on a workstation, often before any install.

    Importing it must not require torch, so that ``python -m
    atlas.tools.inspect_capture`` works in a bare interpreter with only Pillow.
    """
    code = (
        "import sys; import atlas.tools.inspect_capture; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PACKAGE, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "atlas.tools.inspect_capture pulled in torch; it must stay runnable "
        f"before the project is installed.\n{result.stderr}"
    )
