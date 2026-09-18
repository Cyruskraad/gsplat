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

"""Chunked contraction: same answer, bounded temporaries.

A chunked reduction that is off by a rounding step at every boundary would pass
a casual eye and quietly change every number in the ledger, so equality is
checked at chunk sizes that do and do not divide the input, on all three light
forms, and in both precisions.

The memory claim is measured rather than asserted from the shape arithmetic:
each configuration runs in its own process and reports ``ru_maxrss``. The
alternative -- reasoning about what ``einsum`` allocates -- is exactly how the
module came to claim a saving on the shared-light path that measurement then
showed did not exist.
"""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from atlas.functional import (  # noqa: E402
    DEFAULT_CHUNK_BYTES,
    auto_chunk,
    check_finite,
    contract,
    contract_chunked,
    contract_screen,
    contract_screen_chunked,
    contraction_bytes,
)

ROOT = str(Path(__file__).resolve().parent.parent)


def _transport(num=97, atoms=13, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num, 3, atoms, generator=generator, dtype=dtype)


# --- the same answer --------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 32, 97, 200])
def test_a_shared_light_contracts_identically_however_it_is_chunked(chunk):
    """Chunk sizes that divide 97 (only 1 and 97 do), that do not, and one
    larger than the input, so the tail path is exercised rather than assumed."""
    transport, ell = _transport(), torch.randn(3, 13, dtype=torch.float64)
    reference = contract(transport, ell)
    chunked = contract_chunked(transport, ell, chunk_size=chunk)
    assert torch.allclose(reference, chunked, atol=1e-15), float(
        (reference - chunked).abs().max()
    )


@pytest.mark.parametrize("chunk", [1, 7, 32, 97, 200])
def test_a_per_primitive_light_contracts_identically_however_it_is_chunked(chunk):
    transport = _transport()
    ell = _transport(seed=1)
    reference = contract(transport, ell)
    chunked = contract_chunked(transport, ell, chunk_size=chunk)
    # Per-primitive contraction is elementwise within a row, so slicing cannot
    # change the accumulation order at all: this one is exact, not merely close.
    assert torch.equal(reference, chunked)


def test_a_generated_light_matches_a_stored_one():
    """The form that makes near-field training at B = 128 possible: the
    [N, 3, B] light is never allocated, only produced a chunk at a time."""
    transport = _transport()
    stored = _transport(seed=2)
    calls = []

    def generate(start, stop):
        calls.append((start, stop))
        return stored[start:stop]

    chunked = contract_chunked(transport, generate, chunk_size=20)
    assert torch.equal(contract(transport, stored), chunked)
    assert calls == [(0, 20), (20, 40), (40, 60), (60, 80), (80, 97)]


def test_a_generated_light_may_be_shared_across_the_chunk():
    transport = _transport()
    ell = torch.randn(3, 13, dtype=torch.float64)
    chunked = contract_chunked(transport, lambda a, b: ell, chunk_size=20)
    assert torch.allclose(contract(transport, ell), chunked, atol=1e-15)


@pytest.mark.parametrize("chunk", [1, 3, 8, 40])
def test_screen_space_contraction_is_unchanged_by_chunking(chunk):
    transport = torch.randn(11, 9, 3, 13, dtype=torch.float64)
    ell = torch.randn(3, 13, dtype=torch.float64)
    assert torch.equal(
        contract_screen(transport, ell),
        contract_screen_chunked(transport, ell, chunk_size=chunk),
    )


def test_float32_agrees_to_the_precision_it_has():
    """Measured across 20 seeds and four chunk sizes at N = 4000, B = 64: the
    worst disagreement is 2.2e-7 of the output's own scale, which is float32
    rounding in the matmul and not a boundary error.

    Scaled by the largest output rather than element by element, because an
    element-wise relative error is unbounded wherever the true radiance is near
    zero -- which, on a contraction of signed transport against a signed light,
    is somewhere in every batch.
    """
    generator = torch.Generator().manual_seed(11)
    transport = torch.randn(4000, 3, 64, generator=generator)
    ell = torch.randn(3, 64, generator=generator)
    reference = contract(transport, ell)
    for chunk in (1, 7, 512, 4000):
        chunked = contract_chunked(transport, ell, chunk_size=chunk)
        scaled = (reference - chunked).abs().max() / reference.abs().max()
        assert float(scaled) < 1e-6, (chunk, float(scaled))


def test_the_default_chunk_size_gives_the_same_answer_as_an_explicit_one():
    transport = _transport(num=5000, atoms=32)
    ell = torch.randn(3, 32, dtype=torch.float64)
    assert torch.allclose(
        contract_chunked(transport, ell),
        contract_chunked(transport, ell, chunk_size=64),
        atol=1e-15,
    )


# --- gradients --------------------------------------------------------------


def test_gradients_match_the_unchunked_path():
    """The trainer back-propagates through this. A chunked forward that drops a
    chunk's gradient would still produce a plausible loss curve."""
    transport = _transport(num=40, atoms=7).requires_grad_(True)
    ell = torch.randn(3, 7, dtype=torch.float64)
    weight = torch.randn(40, 3, dtype=torch.float64)

    (contract(transport, ell) * weight).sum().backward()
    reference = transport.grad.clone()

    transport.grad = None
    (contract_chunked(transport, ell, chunk_size=6) * weight).sum().backward()
    assert torch.allclose(reference, transport.grad, atol=1e-14)
    assert float(reference.abs().max()) > 0.1  # the premise: gradients exist


def test_gradients_reach_a_generated_light_too():
    transport = _transport(num=30, atoms=5).requires_grad_(True)
    stored = _transport(num=30, atoms=5, seed=3).requires_grad_(True)
    out = contract_chunked(transport, lambda a, b: stored[a:b], chunk_size=8)
    out.sum().backward()
    assert stored.grad is not None
    assert float(stored.grad.abs().sum()) > 0.0


# --- memory, measured -------------------------------------------------------


def _peak_delta_mb(body: str) -> float:
    """Peak RSS above the allocation of the inputs, in its own process."""
    script = f"""
import resource, torch
from atlas.functional import (contract, contract_chunked, contract_screen,
                              contract_screen_chunked)
def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
torch.manual_seed(0)
{body}
print(peak() - base)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": ROOT, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip().splitlines()[-1])


PER_PRIMITIVE_SETUP = """
transport = torch.randn(200_000, 3, 64)
ell = torch.randn(200_000, 3, 64)
base = peak()
"""

SCREEN_SETUP = """
transport = torch.randn(540, 960, 3, 32)
ell = torch.randn(3, 32)
base = peak()
"""


@pytest.mark.slow
def test_chunking_a_per_primitive_contraction_lowers_peak_memory():
    """Measured: 150 MB unchunked against 11 MB chunked, for a [200k, 3, 64]
    product. The bar is set at 4x rather than at the measured 13x so that an
    allocator change is tolerated and losing the saving is not."""
    unchunked = _peak_delta_mb(PER_PRIMITIVE_SETUP + "contract(transport, ell)")
    chunked = _peak_delta_mb(
        PER_PRIMITIVE_SETUP + "contract_chunked(transport, ell, chunk_size=8192)"
    )
    # Assert the premise: the unchunked path really does allocate the product.
    assert unchunked > 100.0, f"expected a ~150 MB temporary, saw {unchunked:.0f} MB"
    assert chunked * 4 < unchunked, (chunked, unchunked)


@pytest.mark.slow
def test_chunking_a_screen_space_contraction_lowers_peak_memory():
    """Measured: 198 MB unchunked against 21 MB chunked at 540x960 with 32
    atoms. At 1080p with 128 atoms the unchunked buffer alone is 3 GB, which is
    what makes this the difference between Path B running and not."""
    unchunked = _peak_delta_mb(SCREEN_SETUP + "contract_screen(transport, ell)")
    chunked = _peak_delta_mb(
        SCREEN_SETUP + "contract_screen_chunked(transport, ell, chunk_size=32)"
    )
    assert unchunked > 150.0, f"expected a ~198 MB temporary, saw {unchunked:.0f} MB"
    assert chunked * 4 < unchunked, (chunked, unchunked)


@pytest.mark.slow
def test_the_shared_light_path_has_no_temporary_to_remove():
    """Recorded because the module used to claim otherwise.

    ``einsum("ncb,cb->nc", ...)`` reduces through a strided matmul without ever
    forming the product, so chunking it saves nothing. Stating that is worth
    more than a comment asserting a saving that is not there.
    """
    setup = """
transport = torch.randn(200_000, 3, 64)
ell = torch.randn(3, 64)
base = peak()
"""
    assert _peak_delta_mb(setup + "contract(transport, ell)") < 30.0


# --- non-finite transport ---------------------------------------------------


def test_check_finite_passes_a_clean_tensor_without_comment():
    assert check_finite(torch.randn(10, 3, 4), "transport") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_check_finite_names_the_count_and_the_first_index(bad):
    tensor = torch.zeros(4, 3, 2)
    tensor[2, 1, 0] = bad
    with pytest.raises(ValueError, match=r"1 non-finite value of 24.*\(2, 1, 0\)"):
        check_finite(tensor, "transport")


def test_check_finite_counts_them_all():
    tensor = torch.full((3, 4), float("nan"))
    with pytest.raises(ValueError, match="12 non-finite values of 12"):
        check_finite(tensor, "transport")


def test_validation_names_the_chunk_a_diverged_primitive_is_in():
    """Without this a diverged run produces a black image and no clue which of
    600,000 primitives went wrong."""
    transport = _transport(num=97, atoms=13, dtype=torch.float32)
    transport[64, 2, 5] = float("nan")
    ell = torch.randn(3, 13)
    with pytest.raises(ValueError, match=r"transport\[60:80\]"):
        contract_chunked(transport, ell, chunk_size=20, validate=True)


def test_validation_is_off_by_default_because_it_synchronises():
    transport = _transport(num=20, atoms=4, dtype=torch.float32)
    transport[3] = float("nan")
    out = contract_chunked(transport, torch.randn(3, 4), chunk_size=8)
    assert bool(torch.isnan(out[3]).any())


def test_validation_also_catches_a_bad_light():
    transport = _transport(num=20, atoms=4, dtype=torch.float32)
    ell = torch.randn(20, 3, 4)
    ell[7] = float("inf")
    with pytest.raises(ValueError, match=r"ell\[0:20\]"):
        contract_chunked(transport, ell, chunk_size=20, validate=True)


# --- the preflight estimate -------------------------------------------------


def test_the_estimate_matches_arithmetic_done_by_hand():
    """600k primitives, 128 atoms, float32: 600000 * 3 * 128 * 4 bytes."""
    estimate = contraction_bytes(600_000, 128)
    assert estimate["transport"] == 600_000 * 3 * 128 * 4
    assert estimate["light"] == 3 * 128 * 4
    assert estimate["output"] == 600_000 * 3 * 4
    assert estimate["total"] == sum(
        estimate[k] for k in ("transport", "light", "temporary", "output")
    )


def test_chunking_lowers_the_estimated_total():
    unchunked = contraction_bytes(600_000, 128)["total"]
    chunked = contraction_bytes(600_000, 128, chunk_size=1 << 14)["total"]
    assert chunked < unchunked / 1.9, (chunked, unchunked)


def test_a_per_primitive_light_is_counted_as_the_tensor_it_is():
    shared = contraction_bytes(1000, 32)
    per_primitive = contraction_bytes(1000, 32, per_primitive_light=True)
    assert per_primitive["light"] == shared["transport"]
    assert per_primitive["total"] > shared["total"]


def test_the_estimate_follows_the_dtype():
    single = contraction_bytes(1000, 32, dtype=torch.float32)["transport"]
    double = contraction_bytes(1000, 32, dtype=torch.float64)["transport"]
    assert double == 2 * single


def test_auto_chunk_clamps_at_both_ends():
    assert auto_chunk(1000, 1 << 30) == 1  # one row already exceeds the budget
    assert auto_chunk(10, 1) == 10  # the whole input fits
    assert auto_chunk(0, 100) == 1  # no rows: never return zero
    assert 1 <= auto_chunk(10**9, 1024) <= 10**9
    assert auto_chunk(10**9, 1024) == DEFAULT_CHUNK_BYTES // 1024


# --- guards -----------------------------------------------------------------


def test_a_negative_chunk_size_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        contract_chunked(
            _transport(), torch.randn(3, 13, dtype=torch.float64), chunk_size=-1
        )


def test_an_out_buffer_of_the_wrong_shape_names_the_shape_it_wanted():
    with pytest.raises(ValueError, match=r"out must be \[97, 3\]"):
        contract_chunked(
            _transport(),
            torch.randn(3, 13, dtype=torch.float64),
            out=torch.empty(97, 4, dtype=torch.float64),
        )


def test_an_out_buffer_is_written_in_place_and_returned():
    transport = _transport()
    ell = torch.randn(3, 13, dtype=torch.float64)
    buffer = torch.empty(97, 3, dtype=torch.float64)
    returned = contract_chunked(transport, ell, chunk_size=10, out=buffer)
    assert returned is buffer
    assert torch.allclose(buffer, contract(transport, ell), atol=1e-15)


def test_a_per_primitive_light_with_the_wrong_row_count_is_refused():
    with pytest.raises(ValueError, match="has 50 rows but transport has 97"):
        contract_chunked(_transport(), _transport(num=50))


def test_a_callable_that_returns_the_wrong_number_of_rows_is_refused():
    """Silently broadcasting a short chunk would corrupt the tail of every
    contraction and nothing downstream would object."""
    with pytest.raises(ValueError, match=r"returned 5 rows for chunk \[0, 20\)"):
        contract_chunked(_transport(), lambda a, b: _transport(num=5), chunk_size=20)


def test_screen_chunking_refuses_a_light_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"ell must be \[3, B\]"):
        contract_screen_chunked(
            torch.randn(4, 4, 3, 8, dtype=torch.float64),
            torch.randn(3, 7, dtype=torch.float64),
        )


def test_screen_chunking_refuses_an_out_buffer_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"out must be \(4, 4, 3\)"):
        contract_screen_chunked(
            torch.randn(4, 4, 3, 8, dtype=torch.float64),
            torch.randn(3, 8, dtype=torch.float64),
            out=torch.empty(4, 4, 8, dtype=torch.float64),
        )
