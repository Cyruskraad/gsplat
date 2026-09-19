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

"""The trainer.

The first test in this file is the one that matters most, and it exists because
the first version of this trainer did not train. ``nn.Parameter(t)`` shares
storage with ``t`` but is a **new leaf**, so the model rendered from one set of
tensors while the optimiser held another: every gradient landed on the model's
copies, ``params[...].grad`` stayed ``None``, and ``optimizer.step()`` was a
no-op. The loss curve was flat and nothing said why.

Everything here runs through the CPU reference renderer on a scene small enough
to be quick, which is the only reason a trainer test can exist at all before
the GPU runner does.
"""

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from atlas.config import Config, from_dict  # noqa: E402
from atlas.data.synthetic import SyntheticConfig, generate_capture  # noqa: E402
from atlas.run import read_ledger  # noqa: E402
from atlas.train import (  # noqa: E402
    Trainer,
    TrainingDiverged,
    build_model,
    cosine_schedule,
    photometric_loss,
    train,
)

TINY = SyntheticConfig(
    num_views=4,
    num_lights=4,
    num_primitives=24,
    width=14,
    height=14,
    num_atoms=6,
    specular=0.0,
)


def _config(tmp_path, **overrides):
    generate_capture(tmp_path / "cap", TINY)
    payload = {
        "atoms": {"count": 6},
        "data": {
            "capture_dir": str(tmp_path / "cap"),
            "num_val_views": 1,
            "num_test_views": 1,
            "num_val_lights": 1,
            "num_test_lights": 1,
        },
        "model": {"init_count": 96},
        "optim": {
            "max_steps": 8,
            "eval_every": 0,
            "save_every": 0,
            "ssim_weight": 0.0,
            "warmup_steps": 2,
        },
        "runtime": {"backend": "reference"},
        "run_root": str(tmp_path / "runs"),
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            payload.setdefault(section, {})[field] = value
        else:
            payload[section] = value
    return from_dict(Config, payload)


def _trainer(tmp_path, **overrides):
    from atlas.data.loader import load_capture
    from atlas.run import RunDirectory

    config = _config(tmp_path, **overrides)
    capture = load_capture(config.data.capture_dir)
    run = RunDirectory.create(config, name="test")
    return Trainer(config, capture, run)


# --- the bug that made the first trainer not train -------------------------


def test_the_model_renders_from_the_tensors_the_optimiser_holds(tmp_path):
    """``nn.Parameter(t)`` is a new leaf sharing ``t``'s storage. Render from
    one and optimise the other and the loss curve is flat with no error."""
    trainer = _trainer(tmp_path)
    for name, parameter in trainer.params.items():
        assert getattr(trainer.model, name) is parameter, name


def test_a_step_produces_a_gradient_that_is_not_zero(tmp_path):
    trainer = _trainer(tmp_path)
    metrics = trainer.step(list(trainer.split.train[:1]))
    assert metrics["grad_norm"] > 0.0
    assert trainer.params["transport"].grad is not None
    assert float(trainer.params["transport"].grad.abs().max()) > 0.0


def test_the_loss_falls_over_a_handful_of_steps(tmp_path):
    trainer = _trainer(tmp_path, **{"optim.max_steps": 24})
    frames = list(trainer.split.train[:2])
    first = trainer.step(frames)["loss"]
    for _ in range(12):
        last = trainer.step(frames)["loss"]
    assert last < first, (first, last)


def test_the_parameters_actually_move(tmp_path):
    trainer = _trainer(tmp_path)
    before = trainer.params["transport"].detach().clone()
    trainer.step(list(trainer.split.train[:1]))
    assert not torch.equal(before, trainer.params["transport"].detach())


# --- the objective ----------------------------------------------------------


def test_the_loss_is_computed_in_the_log1p_domain():
    """On linear radiance a highlight two orders of magnitude above the diffuse
    surface is very nearly the whole gradient."""
    reference = torch.full((8, 8, 3), 0.3)
    reference[4, 4] = 300.0
    lost_diffuse = reference.clone()
    lost_diffuse[reference < 1.0] = 0.0
    bad_highlight = reference.clone()
    bad_highlight[4, 4] = 330.0

    diffuse_loss, _ = photometric_loss(lost_diffuse, reference, ssim_weight=0.0)
    highlight_loss, _ = photometric_loss(bad_highlight, reference, ssim_weight=0.0)
    # Losing every diffuse pixel must cost more than a 10% error on one highlight.
    assert float(diffuse_loss) > float(highlight_loss)


def test_an_exact_prediction_has_no_loss():
    image = torch.rand(8, 8, 3)
    loss, parts = photometric_loss(image, image, ssim_weight=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    assert parts["l1"] == pytest.approx(0.0, abs=1e-12)


def test_a_mask_excludes_what_it_excludes_from_the_loss():
    reference = torch.rand(8, 8, 3)
    prediction = reference.clone()
    prediction[:4] = 9.0
    mask = torch.zeros(8, 8)
    mask[4:] = 1.0
    loss, _ = photometric_loss(prediction, reference, mask=mask, ssim_weight=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_the_structural_term_is_added_when_it_is_weighted():
    reference = torch.rand(16, 16, 3)
    prediction = torch.rand(16, 16, 3)
    plain, _ = photometric_loss(prediction, reference, ssim_weight=0.0)
    with_ssim, parts = photometric_loss(prediction, reference, ssim_weight=0.5)
    assert float(with_ssim) > float(plain)
    assert "ssim" in parts


# --- the schedule -----------------------------------------------------------


def test_the_schedule_warms_up_then_decays():
    kwargs = dict(max_steps=100, warmup=10, min_scale=0.01)
    assert cosine_schedule(0, **kwargs) == pytest.approx(0.1)
    assert cosine_schedule(9, **kwargs) == pytest.approx(1.0)
    middle = cosine_schedule(55, **kwargs)
    end = cosine_schedule(99, **kwargs)
    assert 0.01 < end < middle < 1.0


def test_the_schedule_never_drops_below_its_floor():
    assert cosine_schedule(10_000, max_steps=100, warmup=10, min_scale=0.05) >= 0.05


def test_a_schedule_with_no_warmup_starts_at_full_rate():
    assert cosine_schedule(0, max_steps=100, warmup=0, min_scale=0.01) == pytest.approx(
        1.0
    )


# --- divergence -------------------------------------------------------------


def test_a_diverged_parameter_names_the_step_and_the_frames(tmp_path):
    """A diverged relighting model renders black and every metric after it is a
    measurement of nothing, so the run stops at the step that caused it."""
    trainer = _trainer(tmp_path)
    frames = list(trainer.split.train[:1])
    trainer.step(frames)
    with torch.no_grad():
        trainer.params["transport"][3, 1, 2] = float("nan")
    with pytest.raises(TrainingDiverged) as info:
        trainer.step(frames)
    # Caught by the gradient-norm check before the parameter check gets to it,
    # which is the earlier and better of the two. Both name the step and the
    # frames, which is what a diagnosis needs.
    message = str(info.value)
    assert "step 1" in message and "frames [0]" in message
    assert "diverged" in message


def test_the_divergence_message_names_the_parameter_block(tmp_path):
    """Tested directly, because the gradient-norm check almost always fires
    first. This path is for the case it cannot catch: a step that writes a
    non-finite value into a parameter from finite gradients."""
    trainer = _trainer(tmp_path)
    with torch.no_grad():
        trainer.params["means"][0, 0] = float("inf")
    with pytest.raises(TrainingDiverged, match="means has 1 non-finite"):
        trainer._assert_finite([0])


# --- checkpoints ------------------------------------------------------------


def test_a_resumed_run_continues_rather_than_restarting(tmp_path):
    trainer = _trainer(tmp_path)
    frames = list(trainer.split.train[:1])
    for _ in range(3):
        trainer.step(frames)
    path = trainer.save_checkpoint("mid")
    transport = trainer.params["transport"].detach().clone()

    fresh = _trainer(tmp_path / "second")
    fresh.config = trainer.config
    fresh.load_checkpoint(path)
    assert fresh.state.step == 3
    assert torch.allclose(fresh.params["transport"].detach(), transport)


def test_a_checkpoint_carries_the_optimiser_state(tmp_path):
    trainer = _trainer(tmp_path)
    frames = list(trainer.split.train[:1])
    trainer.step(frames)
    blob = torch.load(trainer.save_checkpoint("one"), weights_only=False)
    assert set(blob["optimizers"]) >= {"transport", "means"}
    assert blob["optimizers"]["transport"]["state"], "Adam moments were not saved"


def test_resuming_across_a_config_change_is_refused(tmp_path):
    trainer = _trainer(tmp_path)
    path = trainer.save_checkpoint("one")
    other = _trainer(tmp_path / "other", **{"atoms.count": 8})
    with pytest.raises(ValueError, match="different configuration"):
        other.load_checkpoint(path)


# --- learned atoms ----------------------------------------------------------


def test_the_basis_is_not_optimised_unless_it_is_asked_for(tmp_path):
    trainer = _trainer(tmp_path)
    assert "atoms" not in trainer.optimizers
    assert trainer.model.atom_axes.requires_grad is False


def test_the_basis_moves_and_stays_legal_when_it_is_learned(tmp_path):
    trainer = _trainer(tmp_path, **{"optim.learn_atoms": True, "optim.atoms_lr": 0.05})
    assert "atoms" in trainer.optimizers
    before = trainer.model.atom_axes.detach().clone()
    for _ in range(3):
        trainer.step(list(trainer.split.train[:1]))
    after = trainer.model.atom_axes.detach()
    assert not torch.allclose(before, after)
    assert float((after.norm(dim=-1) - 1).abs().max()) < 1e-6
    assert float(trainer.model.atom_sharpness.detach().min()) > 0.0


# --- the whole run ----------------------------------------------------------


def test_a_smoke_run_leaves_a_complete_run_directory_and_a_ledger_row(tmp_path):
    config = _config(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    run, report = train(config, smoke=True, ledger=ledger)

    assert run.status == "COMPLETED"
    for name in ("config.yaml", "provenance.json", "metrics.jsonl", "log.jsonl"):
        assert (run.path / name).is_file(), name
    assert (run.ckpts / "final.pt").is_file()

    (row,) = read_ledger(ledger)
    assert row["run"] == run.path.name
    assert row["smoke"] is True
    assert "heldout_view/psnr/mu" in row and "heldout_light/psnr/mu" in row
    assert "gate/baseline_psnr" in row


def test_an_untrained_model_does_not_pass_the_gate(tmp_path):
    """The hole the constant-image baseline closes.

    A model that has learned nothing scores about the same on both held-out
    sets -- badly -- so the *gap* between them is near zero. Without a floor
    that reads as a pass, and a gate that a random scatter can satisfy is not
    measuring anything. Evaluated at step zero so the premise is certain rather
    than dependent on how far a handful of steps happened to get.
    """
    trainer = _trainer(tmp_path)
    report = trainer.evaluate()
    verdict = report.gate()

    assert report.baseline_psnr is not None
    assert report.held_out_view["psnr/mu"] < report.baseline_psnr  # the premise
    assert abs(verdict.gap_db) < 3.0, "the gap alone looks unremarkable"
    assert not verdict.passed
    assert "not reconstructing" in " ".join(verdict.reasons)


def test_the_environment_is_recorded_in_the_log(tmp_path):
    config = _config(tmp_path)
    run, _ = train(config, smoke=True)
    events = [
        json.loads(line)
        for line in (run.path / "log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    environment = [e for e in events if e.get("event") == "environment"]
    assert environment, "no environment record"
    record = environment[0]
    assert "device" in record and "precision" in record and "memory_plan" in record
    assert record["precision"]["allow_tf32"] is False


def test_evaluation_reports_all_four_sets(tmp_path):
    trainer = _trainer(tmp_path)
    report = trainer.evaluate()
    assert report.held_out_view.count > 0
    assert report.held_out_light.count > 0
    assert report.train is not None and report.train.count > 0
    assert report.baseline_psnr is not None


def test_the_model_scatters_primitives_when_there_is_no_ply(tmp_path):
    from atlas.data.loader import load_capture

    config = _config(tmp_path, **{"model.init_count": 128})
    capture = load_capture(config.data.capture_dir)
    model = build_model(config, capture, device=torch.device("cpu"))
    assert model.num_primitives == 128
    assert model.num_atoms == config.atoms.count
    assert torch.isfinite(model.means).all()
