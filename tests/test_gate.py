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

"""The gate, shown to work in both directions.

A gate that has never failed on purpose is not a gate, and a gate that has
never passed is not evidence either. These train a real model on synthetic data
with a known answer and check that the verdict tracks the truth.

The experiment is deliberately narrow: the model is handed the **true geometry**
and only the transport is fitted. That turns the question into the linear one
the gate is actually about -- did this learn transport, or memorise
illuminations -- instead of confounding it with whether the geometry converged.

What these measured, which is the point of running them:

======  =========  ==================  ============  ========
lights  training   held-out light to   held-out-view   gap
        lights     nearest trained     psnr/mu        psnr/mu
======  =========  ==================  ============  ========
5       3          89 deg              15.6           +8.3
14      12         53 deg              15.5           -4.6
30      28         37 deg              15.4           -0.04
======  =========  ==================  ============  ========

The gap is driven by **angular light coverage**, not by the atom count: at three
training lights the gap is +8 dB whether the model has 4 atoms or 24. With the
held-out light 89 degrees from anything trained the model is extrapolating
across the sphere, and no basis rescues that.
"""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from atlas.config import Config, from_dict  # noqa: E402
from atlas.data.loader import load_capture  # noqa: E402
from atlas.data.synthetic import SyntheticConfig, generate_capture  # noqa: E402
from atlas.run import RunDirectory  # noqa: E402
from atlas.train import Trainer  # noqa: E402

#: Small enough to finish, large enough that the effect is not noise.
SCENE = SyntheticConfig(
    num_views=4, num_primitives=48, width=18, height=18, num_atoms=24, specular=0.0
)

#: Measured to converge; 0.05 diverges and 0.1 blows the transport up by 10x.
TRANSPORT_LR = 0.02


def _fit(tmp_path, tag, *, num_lights, atoms, steps=500):
    """Fit the transport from the true geometry, and report."""
    generate_capture(tmp_path / tag, dataclasses.replace(SCENE, num_lights=num_lights))
    config = from_dict(
        Config,
        {
            "atoms": {"count": atoms},
            "data": {
                "capture_dir": str(tmp_path / tag),
                "num_val_views": 1,
                "num_test_views": 1,
                "num_val_lights": 1,
                "num_test_lights": 1,
            },
            "model": {
                "init_ground_truth": str(tmp_path / tag / "ground_truth.pt"),
                "transport_only": True,
            },
            "optim": {
                "max_steps": steps,
                "eval_every": 0,
                "save_every": 0,
                "ssim_weight": 0.0,
                "warmup_steps": 20,
                "transport_lr": TRANSPORT_LR,
                "batch_size": 2,
            },
            "runtime": {"backend": "reference"},
            "run_root": str(tmp_path / "runs"),
        },
    )
    capture = load_capture(config.data.capture_dir)
    trainer = Trainer(config, capture, RunDirectory.create(config, name=tag))
    order = list(trainer.split.train)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        picks = torch.randint(len(order), (2,), generator=generator)
        trainer.step([order[int(i)] for i in picks])
    return capture, trainer, trainer.evaluate()


def _coverage_degrees(capture, split) -> float:
    """How far the held-out lights are from the nearest one trained on."""
    directions = capture.light_directions()
    trained = sorted({capture.frames[i].light_index for i in split.train})
    held = sorted({capture.frames[i].light_index for i in split.held_out_light})
    worst = 0.0
    for index in held:
        cosine = (directions[index] @ directions[torch.tensor(trained)].T).clamp(-1, 1)
        worst = max(worst, math.degrees(math.acos(float(cosine.max()))))
    return worst


@pytest.mark.slow
def test_the_gate_passes_when_the_lights_cover_the_sphere(tmp_path):
    """The positive case. 28 training lights, worst held-out light 37 degrees
    from a trained one: held-out view and held-out light agree to 0.04 dB."""
    capture, trainer, report = _fit(tmp_path, "covered", num_lights=30, atoms=12)

    coverage = _coverage_degrees(capture, trainer.split)
    assert coverage < 45.0, coverage  # the premise: this is interpolation

    view = report.held_out_view["psnr/mu"]
    assert view > report.baseline_psnr + 2.0, (view, report.baseline_psnr)
    verdict = report.gate()
    assert verdict.passed, (verdict.gap_db, verdict.reasons)
    assert abs(verdict.gap_db) < 1.0


@pytest.mark.slow
def test_the_gate_fails_when_the_held_out_light_is_across_the_sphere(tmp_path):
    """The negative case, and the one that decides the capture protocol.

    Three training lights leave the held-out one 89 degrees from anything the
    model saw. It reconstructs the training illuminations well -- it is above
    the constant-image baseline and scores 15.6 dB on a held-out *view* -- and
    then fails completely on a held-out *light*. That is precisely the failure
    the gate exists to catch, and it fails on the gap rather than on the floor.
    """
    capture, trainer, report = _fit(tmp_path, "sparse", num_lights=5, atoms=12)

    coverage = _coverage_degrees(capture, trainer.split)
    assert coverage > 70.0, coverage  # the premise: this is extrapolation

    view = report.held_out_view["psnr/mu"]
    assert view > report.baseline_psnr, "the model must be reconstructing"
    verdict = report.gate()
    assert not verdict.passed
    assert verdict.gap_db > 4.0, verdict.gap_db
    assert "not transport" in " ".join(verdict.reasons)


@pytest.mark.slow
def test_the_gap_follows_the_light_coverage_and_not_the_atom_count(tmp_path):
    """The finding, stated as the comparison that produced it.

    With three training lights the gap is 8 dB or worse whether the model has
    four atoms or twenty-four. Adding atoms to a capture that does not cover the
    sphere does nothing; adding lights does everything.
    """
    _, _, few_atoms = _fit(tmp_path, "sparse-4", num_lights=5, atoms=4, steps=400)
    _, _, many_atoms = _fit(tmp_path, "sparse-24", num_lights=5, atoms=24, steps=400)
    assert few_atoms.gaps["psnr/mu"] > 4.0
    assert many_atoms.gaps["psnr/mu"] > 4.0

    _, trainer, covered = _fit(tmp_path, "covered-4", num_lights=30, atoms=4, steps=400)
    assert covered.gaps["psnr/mu"] < few_atoms.gaps["psnr/mu"] - 4.0
