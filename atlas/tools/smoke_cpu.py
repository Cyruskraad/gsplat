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

"""``make smoke-cpu``: the whole pipeline, on a machine with no GPU.

Generates a capture, trains on it through the reference renderer, evaluates
both held-out sets, runs the gate and writes a run directory -- every stage the
real pipeline has, at a size that finishes in under a minute.

What it asserts is deliberately modest: that the loss **falls**. Twenty-odd
steps from a random scatter cannot pass the gate and should not claim to; the
point is that every seam between the stages holds, which is the failure this
catches and unit tests do not.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m atlas.tools.smoke_cpu")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--keep", type=Path, help="write here instead of a temp dir")
    args = parser.parse_args(argv)

    import torch  # noqa: F401  (imported here so --help works without it)

    from ..config import Config, from_dict
    from ..data.synthetic import SyntheticConfig, generate_capture
    from ..train import train

    root = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="atlas-smoke-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[1/3] generating a synthetic capture in {root}")
        generate_capture(
            root / "cap",
            SyntheticConfig(
                num_views=4,
                num_lights=4,
                num_primitives=32,
                width=16,
                height=16,
                num_atoms=6,
                specular=0.0,
            ),
        )

        print(f"[2/3] training for {args.steps} steps through the reference renderer")
        config = from_dict(
            Config,
            {
                "atoms": {"count": 6},
                "data": {
                    "capture_dir": str(root / "cap"),
                    "num_val_views": 1,
                    "num_test_views": 1,
                    "num_val_lights": 1,
                    "num_test_lights": 1,
                },
                "model": {"init_count": 128},
                "optim": {
                    "max_steps": args.steps,
                    "eval_every": 0,
                    "save_every": 0,
                    "ssim_weight": 0.0,
                    "warmup_steps": 2,
                },
                "runtime": {"backend": "reference"},
                "run_root": str(root / "runs"),
            },
        )
        run, report = train(config, ledger=root / "ledger.jsonl")

        print("[3/3] checking the seams")
        import json

        rows = [
            json.loads(line)
            for line in (run.path / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        losses = [r["loss"] for r in rows if "loss" in r]
        if len(losses) < 2:
            print(f"FAIL: only {len(losses)} loss records; nothing to compare")
            return 1
        if losses[-1] >= losses[0]:
            print(f"FAIL: the loss did not fall ({losses[0]:.5f} -> {losses[-1]:.5f})")
            return 1

        print()
        print(report.format_table())
        print()
        print(f"loss {losses[0]:.5f} -> {losses[-1]:.5f} over {args.steps} steps")
        print(f"run  {run.path}")
        print(
            "\nOK. The gate is expected to fail here: a few dozen steps from a "
            "random scatter has not reconstructed anything, and the gate now "
            "says so rather than passing on a small gap between two equally "
            "bad numbers."
        )
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
