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

"""Device selection and the preflight that refuses a doomed run.

The CUDA branch is exercised here with a faked device rather than left
untested until the runner exists. That is not as good as running on a GPU and
it is far better than shipping a code path nobody has executed: the interesting
failures -- an explicit ``--device cuda`` that silently becomes CPU, a memory
plan that says yes when it should say no -- are all logic, not hardware.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from atlas.device import (  # noqa: E402
    ADAM_STATE_MULTIPLIER,
    DEFAULT_HEADROOM,
    MemoryPlan,
    PreflightError,
    autotune_chunks,
    configure_precision,
    describe_device,
    plan_memory,
    preflight,
    seed_everything,
    select_device,
)


@pytest.fixture
def fake_cuda(monkeypatch):
    """A plausible 24 GiB card, so the CUDA branch is not dead code on CPU CI."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda index=0: (20 * 2**30, 24 * 2**30)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Fake RTX 4090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    return None


# --- selection --------------------------------------------------------------


def test_auto_falls_back_to_cpu_and_says_so():
    report = select_device("auto")
    if torch.cuda.is_available():  # pragma: no cover - depends on the machine
        pytest.skip("this machine has CUDA; the fall-back path is tested separately")
    assert report.device.type == "cpu"
    assert "no CUDA device" in report.reason
    assert report.cuda_available is False


def test_asking_for_cuda_without_cuda_raises_rather_than_falling_back(monkeypatch):
    """The guard this module exists for.

    A silent fall back turns a missing driver into a run that is a hundred times
    slower and still produces numbers, which then get compared against numbers
    from a GPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreflightError, match="Refusing to fall back to CPU"):
        select_device("cuda")


def test_cpu_is_always_available_when_asked_for():
    assert select_device("cpu").device.type == "cpu"
    assert "explicitly" in select_device("cpu").reason


def test_an_unknown_device_string_lists_the_forms_it_accepts():
    with pytest.raises(PreflightError, match="'auto', 'cpu', 'cuda' or 'cuda:N'"):
        select_device("tpu")


def test_cuda_is_selected_and_described_when_present(fake_cuda):
    report = select_device("cuda")
    assert report.is_cuda and report.device.index == 0
    assert report.name == "Fake RTX 4090"
    assert report.capability == (8, 9)
    assert report.total_bytes == 24 * 2**30
    assert report.free_bytes == 20 * 2**30
    assert report.bf16_supported is True
    assert report.device_count == 2


def test_auto_prefers_cuda_when_it_is_there(fake_cuda):
    report = select_device("auto")
    assert report.is_cuda
    assert "auto-detected" in report.reason


def test_an_index_beyond_the_visible_devices_is_refused(fake_cuda):
    with pytest.raises(PreflightError, match="only 2 CUDA devices"):
        select_device("cuda:5")


def test_a_valid_second_device_is_accepted(fake_cuda):
    assert select_device("cuda:1").device.index == 1


def test_the_report_serialises_flat_for_provenance(fake_cuda):
    payload = select_device("cuda").to_dict()
    json.dumps(payload)  # must not raise
    assert payload["device"] == "cuda:0"
    assert payload["capability"] == [8, 9]
    assert all(not isinstance(v, (dict, tuple)) for v in payload.values())


def test_the_report_prints_the_card_and_its_free_memory(fake_cuda):
    text = str(select_device("cuda"))
    assert "Fake RTX 4090" in text and "sm_89" in text and "GiB free" in text


def test_a_cpu_report_prints_its_reason():
    assert "requested explicitly" in str(select_device("cpu"))


def test_describe_does_not_select(fake_cuda):
    """It reports on whatever it is handed, so a caller can describe a device it
    is not going to use."""
    report = describe_device(torch.device("cpu"), "for the record")
    assert report.device.type == "cpu" and report.cuda_available is True


# --- precision --------------------------------------------------------------


def test_tf32_is_off_by_default_because_the_exactness_gate_lives_in_float32():
    """TF32 keeps ten mantissa bits against float32's twenty-four. A contraction
    through it misses the 1e-5 Path A / Path B gate by a margin that reads as a
    broken method rather than as a rounding policy."""
    assert configure_precision()["allow_tf32"] is False


def test_tf32_can_be_turned_on_deliberately_and_says_that_it_was():
    settings = configure_precision(allow_tf32=True)
    assert settings["allow_tf32"] is True
    configure_precision()  # put it back


def test_the_precision_setting_is_applied_not_merely_reported(fake_cuda):
    configure_precision(allow_tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    configure_precision(allow_tf32=False)
    assert torch.backends.cuda.matmul.allow_tf32 is False


# --- seeding ----------------------------------------------------------------


def test_seeding_makes_two_draws_identical():
    seed_everything(11)
    first = torch.randn(64)
    seed_everything(11)
    assert torch.equal(first, torch.randn(64))


def test_a_different_seed_draws_differently():
    seed_everything(11)
    first = torch.randn(64)
    seed_everything(12)
    assert not torch.equal(first, torch.randn(64))


def test_seeding_reports_what_it_pinned():
    report = seed_everything(3, deterministic=True)
    assert report["seed"] == 3
    assert report["deterministic"] is True
    assert report["cudnn_benchmark"] is False


# --- the memory plan --------------------------------------------------------


def test_the_plan_matches_arithmetic_done_by_hand():
    """N = 1000, B = 16, float32. Per primitive: means 3 + quats 4 + scales 3 +
    opacity 1 + transport 3*16 = 59 floats = 236 bytes. Plus the atoms, 16 * 4
    floats."""
    plan = plan_memory(1000, 16, per_primitive_light=False, width=8, height=8)
    assert plan.parameters == 1000 * 59 * 4 + 16 * 4 * 4
    assert plan.optimiser == plan.parameters * ADAM_STATE_MULTIPLIER


def test_the_plan_fits_when_there_is_room_and_does_not_when_there_is_not():
    small = plan_memory(1000, 16, available_bytes=8 * 2**30)
    huge = plan_memory(4_000_000, 128, available_bytes=2 * 2**30)
    assert small.fits is True
    assert huge.fits is False


def test_the_budget_is_a_fraction_of_free_memory_not_all_of_it():
    """Planning to 100% of free memory is planning to fail: the allocator
    fragments, the context is resident, cuBLAS wants a workspace."""
    assert DEFAULT_HEADROOM < 1.0
    plan = plan_memory(1000, 16, available_bytes=1024)
    assert plan.headroom == DEFAULT_HEADROOM


def test_a_near_field_light_costs_more_than_a_shared_one():
    shared = plan_memory(10_000, 32, per_primitive_light=False)
    near = plan_memory(10_000, 32, per_primitive_light=True)
    assert near.total > shared.total


def test_chunking_alone_does_not_shrink_a_stored_light():
    """Only the product temporary shrinks. A near-field light held as
    ``[N, 3, B]`` is still 879 MB at N = 600k, B = 128, chunked or not -- which
    is why the estimate keeps the two apart."""
    unchunked = plan_memory(600_000, 128)
    chunked = plan_memory(600_000, 128, chunk_size=1 << 14)
    assert chunked.transport_temporary < unchunked.transport_temporary
    assert chunked.transport_temporary > unchunked.transport_temporary / 2


def test_generating_the_light_per_chunk_is_what_actually_shrinks_it():
    """The callable form of ``contract_chunked``: the [N, 3, B] light never
    exists. Measured here as the estimate it produces."""
    stored = plan_memory(600_000, 128, chunk_size=1 << 14)
    generated = plan_memory(600_000, 128, chunk_size=1 << 14, light_is_generated=True)
    assert generated.transport_temporary < stored.transport_temporary / 10
    assert generated.total < stored.total


def test_the_plan_prints_every_block_so_a_refusal_can_be_argued_with():
    text = str(plan_memory(600_000, 128, available_bytes=24 * 2**30))
    for label in ("parameters", "optimiser state", "transport temporary", "budget"):
        assert label in text
    assert "600,000" in text


def test_a_degenerate_plan_is_refused():
    with pytest.raises(ValueError, match="at least one primitive"):
        plan_memory(0, 16)


def test_the_plan_is_json_serialisable_for_the_ledger():
    json.dumps(plan_memory(1000, 16).to_dict())


# --- preflight --------------------------------------------------------------


def test_preflight_refuses_a_run_that_cannot_fit_and_shows_the_arithmetic(
    fake_cuda, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda index=0: (2 * 2**30, 24 * 2**30)
    )
    with pytest.raises(PreflightError) as info:
        preflight(num_primitives=4_000_000, num_atoms=128)
    message = str(info.value)
    assert "GiB but only" in message
    assert "parameters" in message and "optimiser state" in message
    assert "Reduce atoms.count" in message


def test_preflight_passes_a_run_that_fits(fake_cuda):
    plan, warnings = preflight(num_primitives=600_000, num_atoms=32)
    assert plan.fits
    assert not any("CPU" in w for w in warnings)


def test_preflight_warns_loudly_when_the_gate_cannot_mean_anything(fake_cuda):
    """A bracket capture trains perfectly well. Believing the gate afterwards is
    the mistake, so the warning is at preflight and not at the end."""
    _, warnings = preflight(
        num_primitives=1000, num_atoms=16, splits_are_independent=False
    )
    assert any("not a measurement" in w for w in warnings)


def test_preflight_refuses_a_split_that_leaves_nothing_to_train_on(fake_cuda):
    with pytest.raises(PreflightError, match="no training frames"):
        preflight(num_primitives=1000, num_atoms=16, trainable_frames=0)


def test_preflight_refuses_an_empty_capture(fake_cuda):
    with pytest.raises(PreflightError, match="no frames"):
        preflight(num_primitives=1000, num_atoms=16, capture_frames=0)


def test_preflight_refuses_when_the_disk_is_too_small(fake_cuda, tmp_path):
    with pytest.raises(PreflightError, match="of checkpoints and"):
        preflight(
            num_primitives=1000,
            num_atoms=16,
            run_root=tmp_path / "runs",
            required_disk_bytes=1 << 60,
        )


def test_preflight_says_a_cpu_run_is_a_smoke_test():
    _, warnings = preflight(
        num_primitives=1000, num_atoms=16, report=select_device("cpu")
    )
    assert any("smoke tests, not runs" in w for w in warnings)


def test_preflight_notices_cuda_without_gsplat(fake_cuda, monkeypatch):
    monkeypatch.setattr("atlas.device._gsplat_version", lambda: None)
    _, warnings = preflight(num_primitives=1000, num_atoms=16)
    assert any("gsplat did not import" in w for w in warnings)


# --- chunk autotuning -------------------------------------------------------


def test_chunks_scale_with_the_memory_that_is_actually_free(fake_cuda):
    big = autotune_chunks(select_device("cuda"))["chunk_bytes"]
    assert big > 64 << 20  # larger than the fixed default on a 20 GiB card


def test_chunks_are_clamped_at_both_ends(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (64 << 20, 1 << 30))
    assert autotune_chunks(select_device("cuda"))["chunk_bytes"] == 8 << 20
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (1 << 40, 1 << 40))
    assert autotune_chunks(select_device("cuda"))["chunk_bytes"] == 1 << 30


def test_a_cpu_report_gets_the_fixed_default():
    from atlas.functional.transport import DEFAULT_CHUNK_BYTES

    assert autotune_chunks(select_device("cpu"))["chunk_bytes"] == DEFAULT_CHUNK_BYTES


def test_an_impossible_fraction_is_refused(fake_cuda):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        autotune_chunks(select_device("cuda"), fraction=1.5)
