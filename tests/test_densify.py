# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gsplat's MCMC densification driving an ATLAS model.

The strategy is gsplat's, unchanged. What is tested here is the part that is
ours: the transport ``[N, 3, B]`` follows every relocated and added Gaussian,
the model is rebound to the tensors gsplat replaced, and the cap holds.

gsplat's relocation kernel is CUDA-only, so on CPU it is replaced by the
closed form it implements (Kheradmand et al. 2024, eq. 9), in PyTorch. That
substitutes the kernel, not the strategy.
"""

import math

import pytest
import torch

pytest.importorskip("gsplat.strategy")

from test_train import _trainer  # noqa: E402


def _relocation(opacities, scales, ratios, binoms, min_opacity):
    """New opacity ``1 - (1 - o)^(1/N)`` and the scale that preserves coverage."""
    ratios = ratios.to(opacities.dtype)
    new_opacities = 1.0 - (1.0 - opacities) ** (1.0 / ratios)
    new_opacities = new_opacities.clamp(min_opacity, 1.0 - 1e-7)
    denominators = torch.zeros_like(opacities)
    for n in range(opacities.shape[0]):
        count = int(ratios[n])
        total = 0.0
        for i in range(1, count + 1):
            for k in range(i):
                total += (
                    math.comb(i - 1, k)
                    * ((-1) ** k)
                    / math.sqrt(k + 1)
                    * float(new_opacities[n]) ** (k + 1)
                )
        denominators[n] = total
    new_scales = (opacities / denominators).unsqueeze(-1) * scales
    return new_opacities, new_scales


@pytest.fixture
def cpu_gsplat(monkeypatch):
    import gsplat.strategy.ops as ops
    from gsplat.cuda._math import _quat_scale_to_covar_preci

    monkeypatch.setattr(ops, "compute_relocation", _relocation)
    monkeypatch.setattr(ops, "quat_scale_to_covar_preci", _quat_scale_to_covar_preci)
    return ops


def _mcmc_trainer(tmp_path, **extra):
    overrides = {
        "densify": {
            "strategy": "mcmc",
            "cap_max": 120,
            "refine_start": 0,
            "refine_stop": 100,
            "refine_every": 2,
        }
    }
    overrides["densify"].update(extra)
    return _trainer(tmp_path, **overrides)


def test_the_relocation_formula_preserves_a_single_gaussian():
    o = torch.tensor([0.3, 0.7])
    s = torch.ones(2, 3)
    new_o, new_s = _relocation(o, s, torch.ones(2, dtype=torch.int32), None, 0.0)
    assert torch.allclose(new_o, o) and torch.allclose(new_s, s, atol=1e-6)


def test_mcmc_grows_to_the_cap_and_the_transport_comes_along(tmp_path, cpu_gsplat):
    trainer = _mcmc_trainer(tmp_path)
    start = trainer.model.num_primitives
    assert start == 96
    for _ in range(6):
        trainer.step([trainer.split["train"][0]])
    count = trainer.model.num_primitives
    assert start < count <= 120
    for name in ("means", "quats", "scales", "opacities", "transport"):
        assert trainer.params[name].shape[0] == count, name
    assert trainer.model.transport.shape == (count, 3, trainer.model.num_atoms)
    # The model renders from the optimiser's tensors, not stale copies.
    assert trainer.model.transport is trainer.params["transport"]
    assert trainer.model.means is trainer.params["means"]


def test_training_continues_after_densification(tmp_path, cpu_gsplat):
    trainer = _mcmc_trainer(tmp_path)
    for _ in range(4):
        trainer.step([trainer.split["train"][0]])
    before = trainer.params["transport"].detach().clone()
    parts = trainer.step([trainer.split["train"][0]])
    assert parts["primitives"] == trainer.model.num_primitives
    assert "opacity_reg" in parts and "scale_reg" in parts
    after = trainer.params["transport"].detach()
    if after.shape == before.shape:
        assert not torch.equal(before, after)


def test_a_densified_checkpoint_resumes(tmp_path, cpu_gsplat):
    trainer = _mcmc_trainer(tmp_path)
    for _ in range(4):
        trainer.step([trainer.split["train"][0]])
    path = trainer.save_checkpoint("densified")
    count = trainer.model.num_primitives
    resumed = _mcmc_trainer(tmp_path / "again")
    resumed.config = trainer.config
    resumed.load_checkpoint(path)
    assert resumed.model.num_primitives == count
    resumed.step([resumed.split["train"][0]])


def test_densifying_frozen_geometry_is_refused(tmp_path):
    with pytest.raises(ValueError, match="transport_only"):
        _trainer(
            tmp_path,
            densify={"strategy": "mcmc"},
            model={"init_count": 96, "transport_only": True},
        )


def test_an_unknown_strategy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="densify.strategy"):
        _trainer(tmp_path, densify={"strategy": "adc"})
