# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Transport contraction: the two render paths, and why they are the same one.

A primitive's outgoing radiance is linear in the illumination by construction:

.. math::
    L_i^c = \\sum_k M_i^{c,k} \\, \\ell^{c,k}

Alpha compositing is a weighted sum whose weights depend only on geometry and
opacity -- never on the light. So for a pixel ``p`` covered by primitives with
compositing weights ``w_i``,

.. math::
    C(p)^c = \\sum_i w_i L_i^c
           = \\sum_i w_i \\sum_k M_i^{c,k} \\ell^{c,k}
           = \\sum_k \\Big( \\sum_i w_i M_i^{c,k} \\Big) \\ell^{c,k}
           = \\sum_k \\mathbb{T}(p)^{c,k} \\ell^{c,k}

The two groupings of that sum are the two render paths:

**Path A, contract then splat.** Evaluate ``L_i = M_i . ell`` per primitive
(:func:`contract`), then rasterise three channels. The contraction does not
depend on the camera, so one contraction serves any number of views -- this is
the path that makes multi-view relighting cheap.

**Path B, splat then contract.** Rasterise the ``3B`` channels of ``M_i``
(:func:`pack_transport`), then contract per pixel (:func:`contract_screen`).
The splat does not depend on the light, so one splat serves any number of
illuminations -- this is the path that makes light editing and inverse lighting
interactive.

This is *not* the usual deferred-shading approximation. Deferred shading
composites normals and roughnesses and then evaluates a non-linear BRDF on the
blend, which is only correct where a pixel is covered by a single opaque
surface. Here the shading operator is linear, so the interchange above is an
identity. :func:`composite` and the tests in ``tests/relight`` exist to hold
that claim to a number rather than to an argument.

Near-field training breaks the *shared* ``ell`` but not the linearity: when
every primitive sees its own light direction, ``ell`` acquires a leading ``N``
and Path A still applies. Path B does not, because there is no single ``ell``
to contract against in screen space. That asymmetry is the honest statement of
what the fast path buys and when.
"""

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "contract",
    "contract_chunked",
    "contract_screen",
    "contract_screen_chunked",
    "auto_chunk",
    "check_finite",
    "contraction_bytes",
    "DEFAULT_CHUNK_BYTES",
    "pack_transport",
    "unpack_transport",
    "composite",
    "compositing_weights",
]


def _check_transport(transport: Tensor) -> Tuple[int, int]:
    if transport.ndim < 3 or transport.shape[-2] != 3:
        raise ValueError(f"transport must be [..., 3, B], got {tuple(transport.shape)}")
    return transport.shape[-2], transport.shape[-1]


def contract(transport: Tensor, ell: Tensor) -> Tensor:
    """Path A: contract per-primitive transport against the light. ``M . ell``.

    Args:
        transport: ``[N, 3, B]`` per-primitive transport.
        ell: ``[3, B]`` for a shared (far-field) illumination, or ``[N, 3, B]``
            when every primitive sees its own light, which is what near-field
            point-light training produces.

    Returns:
        ``[N, 3]`` outgoing radiance.
    """
    _check_transport(transport)
    if transport.ndim != 3:
        raise ValueError(f"transport must be [N, 3, B], got {tuple(transport.shape)}")
    if ell.shape[-2:] != transport.shape[-2:]:
        raise ValueError(
            f"ell trailing dims {tuple(ell.shape[-2:])} must match transport "
            f"{tuple(transport.shape[-2:])}"
        )
    if ell.ndim == 2:
        return torch.einsum("ncb,cb->nc", transport, ell)
    if ell.ndim == 3:
        if ell.shape[0] != transport.shape[0]:
            raise ValueError(
                f"per-primitive ell has {ell.shape[0]} rows but transport has "
                f"{transport.shape[0]}"
            )
        return (transport * ell).sum(dim=-1)
    raise ValueError(f"ell must be [3, B] or [N, 3, B], got {tuple(ell.shape)}")


def contract_screen(transport: Tensor, ell: Tensor) -> Tensor:
    """Path B: contract composited screen-space transport against the light.

    This is the whole cost of changing the illumination once the transport has
    been splatted: ``3B`` multiply-adds per pixel, independent of the number of
    primitives and of how many lights the illumination contains.

    Args:
        transport: ``[..., 3, B]`` composited transport, e.g. ``[H, W, 3, B]``.
        ell: ``[3, B]`` shared light coefficients.

    Returns:
        ``[..., 3]`` radiance.
    """
    _check_transport(transport)
    if ell.ndim != 2 or ell.shape != transport.shape[-2:]:
        raise ValueError(
            f"ell must be [3, B] matching transport trailing dims "
            f"{tuple(transport.shape[-2:])}, got {tuple(ell.shape)}"
        )
    return (transport * ell).sum(dim=-1)


def pack_transport(transport: Tensor) -> Tensor:
    """Flatten ``[..., 3, B]`` transport into ``[..., 3*B]`` splat channels.

    Channel order is channel-major (``c * B + k``), so the first ``B`` entries
    are the red atoms. :func:`unpack_transport` is its exact inverse.
    """
    _check_transport(transport)
    return transport.reshape(*transport.shape[:-2], -1)


def unpack_transport(packed: Tensor, num_atoms: int) -> Tensor:
    """Inverse of :func:`pack_transport`."""
    if num_atoms < 1:
        raise ValueError(f"num_atoms must be >= 1, got {num_atoms}")
    if packed.shape[-1] != 3 * num_atoms:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} is not 3 * num_atoms "
            f"({3 * num_atoms})"
        )
    return packed.reshape(*packed.shape[:-1], 3, num_atoms)


def compositing_weights(alphas: Tensor) -> Tensor:
    """Front-to-back alpha compositing weights ``w_i = a_i * prod_{j<i}(1-a_j)``.

    Args:
        alphas: ``[..., P]`` per-primitive alphas, already sorted front to back.

    Returns:
        ``[..., P]`` weights. They sum to ``1 - prod_i (1 - a_i) <= 1``; the
        shortfall is the background transmittance.
    """
    if alphas.shape[-1] < 1:
        raise ValueError("alphas must have at least one primitive")
    one_minus = 1.0 - alphas
    # Exclusive cumulative product, built by shifting rather than by dividing
    # the inclusive product. Dividing is the obvious trick and it is wrong
    # exactly where it matters: a fully opaque primitive makes the divisor zero,
    # and every weight in front of it -- including its own -- collapses to zero.
    inclusive = torch.cumprod(one_minus, dim=-1)
    transmittance = torch.cat(
        [torch.ones_like(inclusive[..., :1]), inclusive[..., :-1]], dim=-1
    )
    return alphas * transmittance


def composite(alphas: Tensor, features: Tensor) -> Tensor:
    """Alpha-composite arbitrary per-primitive features, front to back.

    Deliberately free of any rasteriser: the exactness claim is a property of
    compositing itself, so proving it must not depend on CUDA, on a tile
    schedule, or on anything that only runs on a workstation.

    Args:
        alphas: ``[..., P]`` sorted front to back.
        features: ``[..., P, C]`` per-primitive features.

    Returns:
        ``[..., C]`` composited features.
    """
    if features.shape[:-1] != alphas.shape:
        raise ValueError(
            f"features leading dims {tuple(features.shape[:-1])} must match "
            f"alphas {tuple(alphas.shape)}"
        )
    weights = compositing_weights(alphas)
    return (weights.unsqueeze(-1) * features).sum(dim=-2)


# --- contracting without materialising the whole thing ----------------------
#
# The contraction is where ``N`` and ``B`` multiply, and it is the reason the
# design's "train over-complete at 128 atoms, then compress" is a question
# about memory rather than about taste.
#
# Three tensors matter, at N = 600k primitives and B = 128 atoms in float32:
#
#   transport [N, 3, B]              879 MB   a parameter; unavoidable
#   per-primitive ell [N, 3, B]      879 MB   near-field training only
#   screen transport [H, W, 3, B]   3038 MB   Path B at 1080p
#
# What chunking removes is a *fourth*: the elementwise product that
# ``(transport * ell).sum(-1)`` builds before reducing it, which is another
# copy of the largest tensor in play. Measured with ru_maxrss, over the
# allocation of the inputs:
#
#                                          unchunked   chunked
#   per-primitive ell, N = 200k, B = 64      150 MB     11 MB
#   screen space, 540x960, B = 32            198 MB     21 MB
#
# The shared far-field case is different and worth stating, because the obvious
# assumption about it is wrong: ``einsum("ncb,cb->nc", ...)`` was measured to
# allocate **nothing** beyond its output. It reduces through a strided matmul
# without ever forming the product, so chunking that path buys no memory at
# all. It is still offered, because a caller should not have to know which of
# its three light forms happens to be the cheap one, and because the callable
# form below is what the shared path exists to sit beside.
#
# The callable form is the one that changes what is possible rather than what
# is comfortable. Near-field training needs a different light per primitive,
# and generating it a chunk at a time means the [N, 3, B] light tensor is never
# allocated -- 879 MB at N = 600k, B = 128, which is the difference between
# training over-complete and not.

#: Working-set budget for one chunk when no ``chunk_size`` is given. Small
#: enough to be irrelevant beside the parameters, large enough that the Python
#: loop is not the cost.
DEFAULT_CHUNK_BYTES = 64 << 20


def auto_chunk(
    num_rows: int, bytes_per_row: int, *, budget_bytes: int = DEFAULT_CHUNK_BYTES
) -> int:
    """How many rows fit in ``budget_bytes``, clamped to ``[1, num_rows]``."""
    if num_rows <= 0:
        return 1
    if bytes_per_row <= 0:
        return num_rows
    return max(1, min(num_rows, budget_bytes // bytes_per_row))


def check_finite(tensor: Tensor, name: str = "tensor") -> None:
    """Raise if ``tensor`` holds a NaN or an infinity, saying where.

    Training a linear operator diverges quietly: one non-finite transport row
    contaminates every pixel that primitive touches and nothing else in the
    pipeline objects. The index is included because the interesting question is
    always whether it is one primitive or all of them.
    """
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero()
    count = int(bad.shape[0])
    raise ValueError(
        f"{name} has {count} non-finite value{'s' if count != 1 else ''} of "
        f"{tensor.numel()}; first at index {tuple(int(i) for i in bad[0])}"
    )


def contraction_bytes(
    num_primitives: int,
    num_atoms: int,
    *,
    dtype: torch.dtype = torch.float32,
    per_primitive_light: bool = False,
    chunk_size: int = 0,
) -> dict:
    """An **upper bound** on the footprint of a contraction, for a preflight.

    Upper bound deliberately: a preflight that under-estimates lets a run start
    and die in hour three, which is the failure it exists to prevent. The
    shared far-field path in particular was measured to allocate no temporary
    at all, so this over-states it -- and a run refused for having too little
    headroom is a cheaper mistake than one killed at step 40,000.

    Args:
        num_primitives: ``N``.
        num_atoms: ``B``.
        dtype: Element type of the transport.
        per_primitive_light: Whether ``ell`` is ``[N, 3, B]`` rather than
            ``[3, B]``. A callable light is not stored, so pass ``False``.
        chunk_size: Rows per chunk; ``0`` means the unchunked path, whose
            temporary is a full ``[N, 3, B]``.

    Returns:
        ``{"transport", "light", "temporary", "output", "total"}``, in bytes.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    row = 3 * num_atoms * itemsize
    transport = num_primitives * row
    light = num_primitives * row if per_primitive_light else row
    temporary = (chunk_size if chunk_size > 0 else num_primitives) * row
    output = num_primitives * 3 * itemsize
    return {
        "transport": transport,
        "light": light,
        "temporary": temporary,
        "output": output,
        "total": transport + light + temporary + output,
    }


def _ell_for(ell, start: int, stop: int, transport_chunk: Tensor) -> Tensor:
    """The light for one chunk, whether it is shared, stored, or generated."""
    if callable(ell):
        produced = ell(start, stop)
        if produced.ndim == 3 and produced.shape[0] != stop - start:
            raise ValueError(
                f"the light callable returned {produced.shape[0]} rows for "
                f"chunk [{start}, {stop}); expected {stop - start}"
            )
        return produced
    if ell.ndim == 3:
        return ell[start:stop]
    return ell


def contract_chunked(
    transport: Tensor,
    ell,
    *,
    chunk_size: int = 0,
    validate: bool = False,
    out: Tensor = None,
) -> Tensor:
    """Path A over primitive chunks. Identical result, bounded temporaries.

    Args:
        transport: ``[N, 3, B]``.
        ell: ``[3, B]`` shared, ``[N, 3, B]`` per primitive, or a callable
            ``(start, stop) -> [stop - start, 3, B]`` (or ``[3, B]``). The
            callable form is the one that matters: near-field training needs a
            different light per primitive, and generating it per chunk means the
            full ``[N, 3, B]`` never exists. At ``N = 600k`` and ``B = 128``
            that is 879 MB that is never allocated. With a shared ``[3, B]``
            light there is no memory to save -- see the note above -- and this
            is simply :func:`contract` in a loop.
        chunk_size: Rows per chunk. ``0`` picks one from
            :data:`DEFAULT_CHUNK_BYTES`.
        validate: Check each chunk for non-finite values as it is produced, so
            a diverged run names the primitive rather than producing a black
            image. Off by default: it is a device synchronisation per chunk.
        out: Optional ``[N, 3]`` destination, so a caller that already holds a
            buffer does not allocate another.

    Returns:
        ``[N, 3]``, equal to :func:`contract` on the same inputs.
    """
    _check_transport(transport)
    if transport.ndim != 3:
        raise ValueError(f"transport must be [N, 3, B], got {tuple(transport.shape)}")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")

    num, _, num_atoms = transport.shape
    if not callable(ell) and ell.ndim == 3 and ell.shape[0] != num:
        raise ValueError(
            f"per-primitive ell has {ell.shape[0]} rows but transport has {num}"
        )

    if out is None:
        out = torch.empty(num, 3, dtype=transport.dtype, device=transport.device)
    elif out.shape != (num, 3):
        raise ValueError(f"out must be [{num}, 3], got {tuple(out.shape)}")

    if chunk_size == 0:
        chunk_size = auto_chunk(
            num, 3 * num_atoms * transport.element_size() * 2  # transport + light
        )

    for start in range(0, num, chunk_size):
        stop = min(start + chunk_size, num)
        block = transport[start:stop]
        light = _ell_for(ell, start, stop, block)
        if validate:
            check_finite(block, f"transport[{start}:{stop}]")
            check_finite(light, f"ell[{start}:{stop}]")
        out[start:stop] = contract(block, light)
    return out


def contract_screen_chunked(
    transport: Tensor,
    ell: Tensor,
    *,
    chunk_size: int = 0,
    out: Tensor = None,
) -> Tensor:
    """Path B over rows of the screen-space transport buffer.

    The buffer this reduces is the largest tensor in the system -- 3.2 GB at
    1080p with 128 atoms -- so reducing it a band at a time is not an
    optimisation, it is what makes the path run at all on a card that has to
    hold the model as well.

    Args:
        transport: ``[..., 3, B]`` composited transport, typically
            ``[H, W, 3, B]``. Chunking is over the leading dimension.
        ell: ``[3, B]``.
        chunk_size: Rows of the leading dimension per chunk; ``0`` picks one.
        out: Optional destination of shape ``transport.shape[:-2] + (3,)``.

    Returns:
        ``[..., 3]``, equal to :func:`contract_screen` on the same inputs.
    """
    _check_transport(transport)
    if ell.ndim != 2 or ell.shape != transport.shape[-2:]:
        raise ValueError(
            f"ell must be [3, B] matching transport trailing dims "
            f"{tuple(transport.shape[-2:])}, got {tuple(ell.shape)}"
        )
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")

    target_shape = transport.shape[:-1]
    if out is None:
        out = torch.empty(target_shape, dtype=transport.dtype, device=transport.device)
    elif out.shape != target_shape:
        raise ValueError(f"out must be {tuple(target_shape)}, got {tuple(out.shape)}")

    rows = transport.shape[0]
    if chunk_size == 0:
        per_row = transport[0].numel() * transport.element_size()
        chunk_size = auto_chunk(rows, per_row)

    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        out[start:stop] = contract_screen(transport[start:stop], ell)
    return out
