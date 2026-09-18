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

"""Throughput and peak memory of the contraction, into the results ledger.

This exists to answer one question with a measurement instead of a preference:

    Is training over-complete at ``B = 128`` and compressing afterwards
    affordable, or does the atom count have to be chosen up front?

``docs/relighting-atlas.md`` assumes the former. The contraction is where ``N``
and ``B`` multiply, so it is where the assumption is cheapest to test, and a
number that lands in the ledger next to a commit is a number that can be
compared with the next one.

Run it with ``--report runs/ledger.jsonl`` and every row carries the device,
the dtype, the sizes, the wall time, the primitives per second and the peak
bytes, so a regression is visible rather than remembered.

Timing notes, because a benchmark that is wrong is worse than none:

* CUDA is asynchronous. Every timed region is bracketed by
  ``torch.cuda.synchronize()``, or the numbers are for queueing work rather
  than doing it.
* The first call pays for allocator warm-up and, on CUDA, for kernel
  autotuning. One untimed warm-up runs before every measurement.
* The reported time is the **median** of the repeats, not the mean. A single
  scheduler hiccup should not become the headline.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .functional.transport import (
    contract,
    contract_chunked,
    contract_screen,
    contract_screen_chunked,
    contraction_bytes,
)
from .run import append_ledger, collect_provenance

__all__ = [
    "time_call",
    "benchmark_contraction",
    "benchmark_screen",
    "sweep",
    "main",
]


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_allocated(device: torch.device) -> Optional[int]:
    """Peak bytes held by torch's allocator, or ``None`` where there is no such
    number.

    On CUDA this is the figure that decides whether a run fits, and it is
    resettable, so it can be scoped to the timed region. On CPU torch does not
    expose one. The tempting substitute is ``ru_maxrss``, but that is a process
    high-water mark that never falls: after a warm-up call has already touched
    the peak, the delta across the timed region is zero. Reporting that zero as
    a memory figure would be worse than reporting nothing, so this returns
    ``None`` and the RSS goes in its own field under its own name.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _peak_rss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def time_call(
    call: Callable[[], Any], *, device: torch.device, repeats: int = 5
) -> Dict[str, float]:
    """Median wall time of ``call`` over ``repeats``, after one warm-up.

    Returns the median, the spread, the allocator peak during the timed region
    where the device has one, and the process RSS high-water mark, which is an
    absolute rather than a delta and is labelled as such.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    call()  # warm-up: allocator, autotuning, lazy module loads
    _synchronise(device)

    _reset_peak(device)
    samples: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        _synchronise(device)
        samples.append(time.perf_counter() - start)

    return {
        "seconds": statistics.median(samples),
        "seconds_min": min(samples),
        "seconds_max": max(samples),
        "repeats": repeats,
        "peak_allocated_bytes": _peak_allocated(device),
        "peak_rss_bytes": _peak_rss(),
    }


def benchmark_contraction(
    num_primitives: int,
    num_atoms: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 0,
    per_primitive_light: bool = False,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Time Path A's contraction at one ``(N, B)``.

    ``per_primitive_light`` is the near-field case and the expensive one: the
    light is as large as the transport, and it is the configuration that
    decides whether ``B = 128`` is affordable.
    """
    device = torch.device(device)
    transport = torch.randn(num_primitives, 3, num_atoms, dtype=dtype, device=device)
    if per_primitive_light:
        ell = torch.randn(num_primitives, 3, num_atoms, dtype=dtype, device=device)
    else:
        ell = torch.randn(3, num_atoms, dtype=dtype, device=device)

    if chunk_size == 0:
        call = lambda: contract(transport, ell)  # noqa: E731
    else:
        call = lambda: contract_chunked(  # noqa: E731
            transport, ell, chunk_size=chunk_size
        )

    with torch.no_grad():
        result = time_call(call, device=device, repeats=repeats)

    result.update(
        {
            "benchmark": "contraction",
            "num_primitives": num_primitives,
            "num_atoms": num_atoms,
            "chunk_size": chunk_size,
            "per_primitive_light": per_primitive_light,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "primitives_per_second": num_primitives / result["seconds"],
            "estimated_bytes": contraction_bytes(
                num_primitives,
                num_atoms,
                dtype=dtype,
                per_primitive_light=per_primitive_light,
                chunk_size=chunk_size,
            )["total"],
        }
    )
    return result


def benchmark_screen(
    height: int,
    width: int,
    num_atoms: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 0,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Time Path B's screen-space contraction: the cost of changing the light.

    This is the number the claim rests on. It does not depend on the number of
    primitives, and it does not depend on how many lights the illumination
    contains -- which is the whole difference from a method that is fast *per
    light*.
    """
    device = torch.device(device)
    transport = torch.randn(height, width, 3, num_atoms, dtype=dtype, device=device)
    ell = torch.randn(3, num_atoms, dtype=dtype, device=device)

    if chunk_size == 0:
        call = lambda: contract_screen(transport, ell)  # noqa: E731
    else:
        call = lambda: contract_screen_chunked(  # noqa: E731
            transport, ell, chunk_size=chunk_size
        )

    with torch.no_grad():
        result = time_call(call, device=device, repeats=repeats)

    pixels = height * width
    result.update(
        {
            "benchmark": "screen",
            "height": height,
            "width": width,
            "num_atoms": num_atoms,
            "chunk_size": chunk_size,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "megapixels_per_second": pixels / result["seconds"] / 1e6,
            "milliseconds_per_frame": result["seconds"] * 1e3,
        }
    )
    return result


def sweep(
    *,
    device: torch.device | str = "cpu",
    num_primitives: int = 600_000,
    atom_counts: Sequence[int] = (32, 64, 128),
    resolution: Sequence[int] = (1080, 1920),
    repeats: int = 5,
    chunk_size: int = 0,
    screen_chunk: int = 0,
) -> List[Dict[str, Any]]:
    """The sweep that sizes ``B``: Path A both ways, then Path B."""
    device = torch.device(device)
    rows: List[Dict[str, Any]] = []
    for num_atoms in atom_counts:
        for per_primitive in (False, True):
            rows.append(
                benchmark_contraction(
                    num_primitives,
                    num_atoms,
                    device=device,
                    chunk_size=chunk_size,
                    per_primitive_light=per_primitive,
                    repeats=repeats,
                )
            )
        rows.append(
            benchmark_screen(
                resolution[0],
                resolution[1],
                num_atoms,
                device=device,
                chunk_size=screen_chunk,
                repeats=repeats,
            )
        )
    return rows


def format_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """A table, for the log. The ledger is the contract; this is a courtesy."""
    lines = []
    header = (
        f"{'benchmark':<12}{'B':>5}{'shape':>16}{'ms':>10}{'alloc MB':>10}{'rate':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        if row["benchmark"] == "contraction":
            shape = f"{row['num_primitives']}"
            shape += " near" if row["per_primitive_light"] else " far"
            rate = f"{row['primitives_per_second'] / 1e6:.1f} Mprim/s"
        else:
            shape = f"{row['height']}x{row['width']}"
            rate = f"{row['megapixels_per_second']:.1f} Mpx/s"
        allocated = row.get("peak_allocated_bytes")
        peak = "-" if allocated is None else f"{allocated / 2**20:.0f}"
        lines.append(
            f"{row['benchmark']:<12}{row['num_atoms']:>5}{shape:>16}"
            f"{row['seconds'] * 1e3:>10.2f}{peak:>10}{rate:>16}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atlas.bench",
        description="Contraction throughput and peak memory, into the ledger.",
    )
    parser.add_argument("--report", type=Path, help="ledger to append to (JSONL)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--primitives", type=int, default=600_000)
    parser.add_argument("--atoms", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--resolution", type=int, nargs=2, default=[1080, 1920])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--screen-chunk", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device cuda was asked for but torch reports no CUDA device")

    rows = sweep(
        device=args.device,
        num_primitives=args.primitives,
        atom_counts=args.atoms,
        resolution=tuple(args.resolution),
        repeats=args.repeats,
        chunk_size=args.chunk_size,
        screen_chunk=args.screen_chunk,
    )
    print(format_rows(rows))

    provenance = collect_provenance()
    stamped = [
        {
            **row,
            "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
            "torch": provenance.get("torch"),
            "cuda_device": provenance.get("cuda_device"),
            "hostname": provenance.get("hostname"),
            "machine": platform.machine(),
        }
        for row in rows
    ]
    if args.report:
        for row in stamped:
            append_ledger(args.report, row)
        print(f"\n{len(stamped)} rows appended to {args.report}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stamped, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
