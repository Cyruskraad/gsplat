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

"""The benchmark has to be honest before it is fast.

A benchmark nobody checks is a benchmark that eventually measures the wrong
thing, and in this project the number it produces is the one that decides
whether the atom count can be chosen after training rather than before. So:
the timing is warm rather than cold, the memory figure is either real or
absent, and every row it writes carries the commit it was measured on.

Everything here runs at toy sizes. The sizes that matter need a GPU, and the
point of writing this now is that the GPU runner can produce them on a push.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from atlas.bench import (  # noqa: E402
    benchmark_contraction,
    benchmark_screen,
    format_rows,
    main,
    sweep,
    time_call,
)
from atlas.run import read_ledger  # noqa: E402

CPU = torch.device("cpu")


# --- timing -----------------------------------------------------------------


def test_the_first_call_is_a_warm_up_and_is_not_timed():
    """Otherwise every measurement includes allocator warm-up and, on CUDA,
    kernel autotuning -- which is a real cost, but not the one being reported.
    """
    calls = []
    time_call(lambda: calls.append(1), device=CPU, repeats=3)
    assert len(calls) == 4, "expected one warm-up plus three timed calls"


def test_the_reported_time_is_the_median_not_the_mean():
    """One scheduler hiccup should not become the headline."""
    import time

    durations = iter([0.0, 0.01, 0.01, 1.0])  # warm-up, then three samples

    def call():
        end = time.perf_counter() + next(durations)
        while time.perf_counter() < end:
            pass

    result = time_call(call, device=CPU, repeats=3)
    assert result["seconds"] < 0.5, result
    assert result["seconds_max"] > 0.9


def test_repeats_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        time_call(lambda: None, device=CPU, repeats=0)


def test_cpu_reports_no_allocator_peak_rather_than_a_misleading_zero():
    """``ru_maxrss`` never falls, so a delta across the timed region reads zero
    once the warm-up has touched the peak. Reporting that as a memory figure
    would be worse than reporting nothing."""
    result = time_call(lambda: torch.randn(256, 256), device=CPU, repeats=2)
    assert result["peak_allocated_bytes"] is None
    assert result["peak_rss_bytes"] > 0


# --- what a row says --------------------------------------------------------


def test_a_contraction_row_carries_everything_needed_to_compare_it():
    row = benchmark_contraction(2000, 16, repeats=2)
    for key in (
        "benchmark",
        "num_primitives",
        "num_atoms",
        "chunk_size",
        "per_primitive_light",
        "device",
        "dtype",
        "seconds",
        "primitives_per_second",
        "estimated_bytes",
    ):
        assert key in row, key
    assert row["benchmark"] == "contraction"
    assert row["primitives_per_second"] == pytest.approx(2000 / row["seconds"])


def test_a_screen_row_reports_the_frame_time_the_claim_is_about():
    row = benchmark_screen(32, 48, 16, repeats=2)
    assert row["benchmark"] == "screen"
    assert row["milliseconds_per_frame"] == pytest.approx(row["seconds"] * 1e3)
    assert row["megapixels_per_second"] == pytest.approx(32 * 48 / row["seconds"] / 1e6)


def test_the_near_field_case_is_measured_separately_because_it_is_the_expensive_one():
    far = benchmark_contraction(2000, 16, per_primitive_light=False, repeats=2)
    near = benchmark_contraction(2000, 16, per_primitive_light=True, repeats=2)
    assert far["per_primitive_light"] is False
    assert near["per_primitive_light"] is True
    # The premise, stated as the quantity it actually is: the near-field light
    # costs one extra copy of the transport. The totals do not double, because
    # the temporary and the output are the same in both cases -- which is worth
    # knowing, since "near-field doubles memory" is the easy thing to assume.
    transport_bytes = 2000 * 3 * 16 * 4
    difference = near["estimated_bytes"] - far["estimated_bytes"]
    assert difference == pytest.approx(transport_bytes, rel=0.01), difference


def test_a_chunked_row_records_the_chunk_size_it_used():
    assert (
        benchmark_contraction(2000, 16, chunk_size=256, repeats=2)["chunk_size"] == 256
    )


# --- the sweep and the ledger -----------------------------------------------


def test_the_sweep_covers_both_light_forms_and_the_screen_at_every_atom_count():
    rows = sweep(num_primitives=500, atom_counts=(4, 8), resolution=(16, 24), repeats=1)
    assert len(rows) == 6
    for atoms in (4, 8):
        at = [r for r in rows if r["num_atoms"] == atoms]
        assert {r["benchmark"] for r in at} == {"contraction", "screen"}
        assert {r.get("per_primitive_light") for r in at} == {False, True, None}


def test_the_report_lands_in_the_ledger_stamped_with_the_commit(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    code = main(
        [
            "--report",
            str(ledger),
            "--device",
            "cpu",
            "--primitives",
            "500",
            "--atoms",
            "4",
            "--resolution",
            "16",
            "24",
            "--repeats",
            "1",
        ]
    )
    assert code == 0
    rows = read_ledger(ledger)
    assert len(rows) == 3
    for row in rows:
        assert "git_commit" in row and "torch" in row and "hostname" in row
        assert all(not isinstance(v, (dict, list)) for v in row.values())


def test_rows_are_also_written_as_json_when_asked(tmp_path):
    destination = tmp_path / "nested" / "bench.json"
    main(
        [
            "--json",
            str(destination),
            "--device",
            "cpu",
            "--primitives",
            "500",
            "--atoms",
            "4",
            "--resolution",
            "16",
            "24",
            "--repeats",
            "1",
        ]
    )
    assert len(json.loads(destination.read_text())) == 3


def test_asking_for_cuda_without_cuda_fails_immediately(monkeypatch):
    """Rather than silently benchmarking the CPU and writing a row that claims
    to be a GPU measurement."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        main(["--device", "cuda"])


def test_the_table_prints_a_dash_where_there_is_no_allocator_figure():
    text = format_rows(
        sweep(num_primitives=200, atom_counts=(4,), resolution=(8, 8), repeats=1)
    )
    assert "alloc MB" in text
    assert "Mprim/s" in text and "Mpx/s" in text
    assert "-" in text
