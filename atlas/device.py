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

"""What am I running on, can this run, and will the answer be reproducible.

Three jobs that are usually scattered across a trainer and are here instead,
because each of them is a thing people get wrong in the same way.

**Detection says what it chose and why.** A pipeline that silently falls back to
CPU turns a missing driver into a run that is a hundred times slower and still
produces numbers -- which then get compared against GPU numbers. ``select_device``
returns a :class:`DeviceReport` naming the device, the reason, and everything
about the machine that could change a result. An explicit request for CUDA that
cannot be honoured **raises**; only ``"auto"`` is allowed to fall back, and it
says so.

**Preflight refuses a run that cannot finish.** The expensive failure in this
project is not a crash, it is an out-of-memory at step 40,000 after three hours.
The footprint is computable in advance from the primitive count and the atom
count -- :func:`~atlas.functional.transport.contraction_bytes` and
``RelightSplats.parameter_bytes`` already do it -- so it is computed in the
first second instead.

**Precision is a policy, not a default.** The project's claim is that the
renderer is an exact linear operator in the illumination, gated at
``max|Path A - Path B| < 1e-5`` in float32. TF32 keeps ten mantissa bits, so a
contraction run through it fails that gate by a margin that looks exactly like a
bug in the method rather than a bug in the arithmetic. :func:`configure_precision`
turns TF32 **off** for matmul by default and records what it did. Speed in this
project comes from chunking and from Path B, not from spending mantissa bits
underneath a correctness claim.
"""

from __future__ import annotations

import os
import platform
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .functional.transport import DEFAULT_CHUNK_BYTES, contraction_bytes

__all__ = [
    "DeviceReport",
    "MemoryPlan",
    "PreflightError",
    "select_device",
    "describe_device",
    "configure_precision",
    "seed_everything",
    "plan_memory",
    "preflight",
    "autotune_chunks",
]


class PreflightError(RuntimeError):
    """A run that would fail later, refused now, with the arithmetic attached."""


# --- what am I running on ---------------------------------------------------


@dataclass(frozen=True)
class DeviceReport:
    """The device, why it was chosen, and what about it could change a result."""

    device: torch.device
    reason: str
    cuda_available: bool
    name: Optional[str] = None
    capability: Optional[Tuple[int, int]] = None
    total_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    cuda_version: Optional[str] = None
    driver: Optional[str] = None
    device_count: int = 0
    torch_version: str = ""
    platform: str = ""
    gsplat_version: Optional[str] = None
    bf16_supported: bool = False
    tf32_allowed: Optional[bool] = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Flat and JSON-safe, for ``provenance.json`` and the ledger."""
        payload = asdict(self)
        payload["device"] = str(self.device)
        payload["capability"] = list(self.capability) if self.capability else None
        return payload

    def __str__(self) -> str:
        if not self.is_cuda:
            return f"{self.device} ({self.reason})"
        gigabytes = (self.free_bytes or 0) / 2**30, (self.total_bytes or 0) / 2**30
        capability = (
            f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "?"
        )
        return (
            f"{self.device} {self.name} [{capability}] "
            f"{gigabytes[0]:.1f}/{gigabytes[1]:.1f} GiB free ({self.reason})"
        )


def _gsplat_version() -> Optional[str]:
    try:
        import gsplat
    except Exception:
        return None
    return getattr(gsplat, "__version__", "unknown")


def describe_device(device: torch.device, reason: str) -> DeviceReport:
    """Everything worth recording about ``device``, without selecting anything."""
    available = torch.cuda.is_available()
    common: Dict[str, Any] = {
        "device": device,
        "reason": reason,
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "gsplat_version": _gsplat_version(),
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32)
        if available
        else None,
    }
    if device.type != "cuda":
        return DeviceReport(**common)

    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - old torch builds
        bf16 = False
    return DeviceReport(
        name=torch.cuda.get_device_name(index),
        capability=tuple(torch.cuda.get_device_capability(index)),
        total_bytes=int(total),
        free_bytes=int(free),
        cuda_version=torch.version.cuda,
        driver=getattr(torch.version, "cuda", None),
        bf16_supported=bf16,
        **common,
    )


def select_device(prefer: str = "auto") -> DeviceReport:
    """Choose a device and say why.

    Args:
        prefer: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Returns:
        A :class:`DeviceReport`.

    Raises:
        PreflightError: If CUDA was asked for explicitly and is not usable. A
            silent fall back to CPU is a hundred-times-slower run that still
            produces numbers, and those numbers then get compared against GPU
            ones. Only ``"auto"`` may fall back, and it records that it did.
    """
    prefer = (prefer or "auto").strip().lower()
    available = torch.cuda.is_available()

    if prefer == "cpu":
        return describe_device(torch.device("cpu"), "requested explicitly")

    if prefer == "auto":
        if available:
            return describe_device(
                torch.device("cuda", torch.cuda.current_device()),
                "auto-detected: CUDA is available",
            )
        return describe_device(
            torch.device("cpu"),
            "auto-detected: no CUDA device is visible to torch",
        )

    if not prefer.startswith("cuda"):
        raise PreflightError(
            f"unknown device {prefer!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'"
        )

    if not available:
        raise PreflightError(
            f"{prefer!r} was requested but torch reports no CUDA device. "
            f"torch {torch.__version__}, built for CUDA "
            f"{torch.version.cuda or 'nothing'}. Refusing to fall back to CPU "
            f"silently: pass --device auto if a CPU run is acceptable."
        )

    index = 0 if prefer == "cuda" else int(prefer.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise PreflightError(
            f"{prefer!r} was requested but only {count} CUDA "
            f"device{'s' if count != 1 else ''} "
            f"{'are' if count != 1 else 'is'} visible"
        )
    return describe_device(torch.device("cuda", index), "requested explicitly")


# --- precision --------------------------------------------------------------


def configure_precision(*, allow_tf32: bool = False) -> Dict[str, Any]:
    """Set the matmul precision policy and return what was set.

    TF32 is **off** by default, and that is a deliberate cost. It keeps ten
    mantissa bits against float32's twenty-four, so a contraction run through it
    disagrees with the reference by far more than the ``1e-5`` the Path A / Path
    B exactness gate allows -- and a failure there would read as a broken method
    rather than as a rounding policy. Enabling it is a measurement, not a
    default: turn it on, run ``pytest tests/gpu``, and put the number in the
    ledger before trusting it.
    """
    settings: Dict[str, Any] = {"allow_tf32": bool(allow_tf32)}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        settings["matmul_precision"] = "high" if allow_tf32 else "highest"
        torch.set_float32_matmul_precision(settings["matmul_precision"])
    else:
        settings["matmul_precision"] = "cpu: not applicable"
    return settings


def seed_everything(seed: int, *, deterministic: bool = True) -> Dict[str, Any]:
    """Seed every generator a run touches, and say what was pinned.

    ``deterministic`` also pins cuDNN's algorithm choice. That costs throughput
    on some convolutions and buys a run that can be reproduced from its config
    hash, which is the whole point of the run directory.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
        seeded_numpy = True
    except Exception:
        seeded_numpy = False

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS needs this to make reductions reproducible across launches.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "numpy_seeded": seeded_numpy,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


# --- will this run fit ------------------------------------------------------

#: Adam keeps a first and second moment per parameter, so optimiser state is
#: twice the parameters. Named rather than inlined because swapping optimiser
#: changes it and the arithmetic should not be hidden in a multiplication.
ADAM_STATE_MULTIPLIER = 2

#: Fraction of free memory a run is allowed to plan for. The rest absorbs
#: allocator fragmentation, the CUDA context, cuBLAS workspaces and whatever
#: else is on the card. Planning to 100% of free memory is planning to fail.
DEFAULT_HEADROOM = 0.80


@dataclass(frozen=True)
class MemoryPlan:
    """What a run would need, block by block, so a refusal can be argued with."""

    num_primitives: int
    num_atoms: int
    parameters: int
    optimiser: int
    transport_temporary: int
    image: int
    total: int
    available: Optional[int]
    headroom: float
    fits: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        def gib(value: Optional[int]) -> str:
            return "?" if value is None else f"{value / 2**30:.2f} GiB"

        lines = [
            f"N = {self.num_primitives:,}  B = {self.num_atoms}",
            f"  parameters          {gib(self.parameters)}",
            f"  optimiser state     {gib(self.optimiser)}",
            f"  transport temporary {gib(self.transport_temporary)}",
            f"  image buffers       {gib(self.image)}",
            f"  total               {gib(self.total)}",
        ]
        if self.available is not None:
            lines.append(
                f"  budget              {gib(int(self.available * self.headroom))} "
                f"({self.headroom:.0%} of {gib(self.available)} free)"
            )
        lines.append(f"  fits                {'yes' if self.fits else 'NO'}")
        return "\n".join(lines)


def plan_memory(
    num_primitives: int,
    num_atoms: int,
    *,
    width: int = 1920,
    height: int = 1080,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    per_primitive_light: bool = True,
    light_is_generated: bool = False,
    chunk_size: int = 0,
    available_bytes: Optional[int] = None,
    headroom: float = DEFAULT_HEADROOM,
    optimiser_multiplier: int = ADAM_STATE_MULTIPLIER,
) -> MemoryPlan:
    """Estimate a training step's footprint.

    Deliberately an **over**-estimate, for the same reason
    :func:`~atlas.functional.transport.contraction_bytes` is: a plan that
    under-estimates lets a run start and die in hour three, which is the failure
    this exists to prevent. A run refused for want of headroom is the cheaper
    mistake.

    ``per_primitive_light`` defaults to ``True`` because near-field training is
    the expensive case and the one that decides whether ``B = 128`` is
    affordable.

    ``light_is_generated`` is the difference the callable form of
    :func:`~atlas.functional.transport.contract_chunked` makes. A near-field
    light stored as ``[N, 3, B]`` is 879 MB at ``N = 600k, B = 128``; generated
    a chunk at a time it is a few megabytes and never exists in full. Chunking
    alone does **not** shrink a stored light -- only the product temporary --
    so the two flags are separate and both belong in the estimate.
    """
    if num_primitives < 1 or num_atoms < 1:
        raise ValueError(
            f"need at least one primitive and one atom, got "
            f"{num_primitives} and {num_atoms}"
        )
    itemsize = torch.empty((), dtype=dtype).element_size()

    # means[N,3] quats[N,4] scales[N,3] opacities[N] transport[N,3,B], atoms[B,4]
    per_primitive = (3 + 4 + 3 + 1 + 3 * num_atoms) * itemsize
    parameters = num_primitives * per_primitive + num_atoms * 4 * itemsize
    optimiser = parameters * optimiser_multiplier

    contraction = contraction_bytes(
        num_primitives,
        num_atoms,
        dtype=dtype,
        per_primitive_light=False,  # the light is accounted for below
        chunk_size=chunk_size,
    )
    rows = chunk_size if chunk_size > 0 else num_primitives
    if not per_primitive_light:
        light = 3 * num_atoms * itemsize
    elif light_is_generated:
        light = rows * 3 * num_atoms * itemsize
    else:
        light = num_primitives * 3 * num_atoms * itemsize

    # The parameters are already counted above; take only what the contraction
    # adds on top of them.
    transport_temporary = light + contraction["temporary"] + contraction["output"]

    # Prediction, reference, mask and one gradient of the prediction.
    image = batch_size * width * height * (3 + 3 + 1 + 3) * itemsize

    total = parameters + optimiser + transport_temporary + image
    budget = None if available_bytes is None else int(available_bytes * headroom)
    return MemoryPlan(
        num_primitives=num_primitives,
        num_atoms=num_atoms,
        parameters=parameters,
        optimiser=optimiser,
        transport_temporary=transport_temporary,
        image=image,
        total=total,
        available=available_bytes,
        headroom=headroom,
        fits=budget is None or total <= budget,
    )


def autotune_chunks(report: DeviceReport, *, fraction: float = 0.05) -> Dict[str, int]:
    """Chunk sizes scaled to the memory actually free, in bytes.

    ``DEFAULT_CHUNK_BYTES`` is a fixed 64 MiB, which is timid on a 48 GiB card
    and too much on a 6 GiB one that is already holding a model. Both chunked
    contractions take a byte budget, so the only decision here is how much of
    the free memory one chunk may occupy.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if report.free_bytes is None:
        return {"chunk_bytes": DEFAULT_CHUNK_BYTES}
    budget = int(report.free_bytes * fraction)
    # Never below 8 MiB (the Python loop starts to dominate) and never above
    # 1 GiB (past that the temporary is the problem chunking was solving).
    return {"chunk_bytes": max(8 << 20, min(budget, 1 << 30))}


def preflight(
    *,
    num_primitives: int,
    num_atoms: int,
    report: Optional[DeviceReport] = None,
    run_root: Optional[Path | str] = None,
    required_disk_bytes: int = 0,
    capture_frames: Optional[int] = None,
    trainable_frames: Optional[int] = None,
    splits_are_independent: Optional[bool] = None,
    **plan_kwargs: Any,
) -> Tuple[MemoryPlan, List[str]]:
    """Refuse now what would fail later. Returns the plan and any warnings.

    Checks, in the order they are cheapest to fix:

    1. **The split can answer the question.** A capture whose light is a fixed
       function of the camera cannot support the held-out-light gate at all --
       see ``atlas/data/synthetic.py``. Training on it is not wrong, but
       believing the gate afterwards is, so it warns loudly.
    2. **There are frames to train on.**
    3. **The memory fits**, with headroom.
    4. **The disk fits** the checkpoints and renders.

    Raises:
        PreflightError: On anything that makes the run pointless or impossible,
            with the arithmetic in the message.
    """
    report = report or select_device()
    warnings: List[str] = []

    if splits_are_independent is False:
        warnings.append(
            "this capture's light is a fixed function of its camera, so the "
            "held-out-light and held-out-view sets contain the same kind of "
            "frame and their difference is not a measurement. Training is fine; "
            "the gate will not mean anything."
        )
    if capture_frames is not None and capture_frames < 1:
        raise PreflightError("the capture has no frames")
    if trainable_frames is not None and trainable_frames < 1:
        raise PreflightError(
            "the split leaves no training frames. Reduce num_val_* / num_test_*"
        )

    plan = plan_memory(
        num_primitives,
        num_atoms,
        available_bytes=report.free_bytes,
        **plan_kwargs,
    )
    if not plan.fits:
        budget = int((plan.available or 0) * plan.headroom)
        raise PreflightError(
            f"this run needs about {plan.total / 2**30:.2f} GiB but only "
            f"{budget / 2**30:.2f} GiB is budgeted on {report.name or report.device} "
            f"({plan.headroom:.0%} of {(plan.available or 0) / 2**30:.2f} GiB free).\n"
            f"{plan}\n"
            f"Reduce atoms.count, cap the primitive count, or lower the "
            f"render resolution. Failing here costs a second; failing at step "
            f"40,000 costs the run."
        )

    if run_root is not None and required_disk_bytes > 0:
        root = Path(run_root)
        probe = root if root.exists() else root.parent
        probe.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(probe).free
        if free < required_disk_bytes:
            raise PreflightError(
                f"{required_disk_bytes / 2**30:.2f} GiB of checkpoints and "
                f"renders are planned but only {free / 2**30:.2f} GiB is free "
                f"at {probe}"
            )

    if report.device.type == "cpu":
        warnings.append(
            "running on CPU. The reference renderer is roughly a thousand times "
            "slower than the CUDA kernel, so this is for smoke tests, not runs."
        )
    if report.is_cuda and report.gsplat_version is None:
        warnings.append(
            "CUDA is available but gsplat did not import, so rendering will "
            'fall back to nothing. Install it with: pip install -e ".[gpu]"'
        )
    return plan, warnings
