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

"""Held-out splits over views *and* over lights.

A relighting model has two ways to fail and standard practice only tests one of
them. Holding out views measures novel-view synthesis. Holding out *lights*
measures whether the model learned transport or merely memorised the
illuminations it was shown, and it is the gate that decides whether anything
downstream is worth building.

Both splits are farthest-point rather than random. A random held-out light can
land next to a training light, and the model then scores well for a reason that
has nothing to do with generalisation. Farthest-point selection puts the
held-out samples where the training set is sparsest, which is the honest place
to be measured.

Selection is deterministic with no seed. The first pick is the sample farthest
from the centroid, and every subsequent pick is the one farthest from what has
already been chosen. The same capture always yields the same split, so a result
cannot quietly move when the split does.
"""

from typing import NamedTuple, Tuple

import torch
from torch import Tensor

__all__ = ["Split", "farthest_point_indices", "split_views", "split_lights"]


class Split(NamedTuple):
    """Disjoint index sets over a capture."""

    train: Tensor
    val: Tensor
    test: Tensor


def farthest_point_indices(points: Tensor, count: int) -> Tensor:
    """Deterministically pick ``count`` well-separated samples.

    Args:
        points: ``[N, D]`` sample coordinates -- camera centres for views, unit
            directions for lights.
        count: How many to pick, in ``[0, N]``.

    Returns:
        ``[count]`` indices, in selection order.
    """
    if points.ndim != 2:
        raise ValueError(f"points must be [N, D], got {tuple(points.shape)}")
    num = points.shape[0]
    if count < 0 or count > num:
        raise ValueError(f"count must be in [0, {num}], got {count}")
    if count == 0:
        return torch.zeros(0, dtype=torch.long, device=points.device)

    centroid = points.mean(dim=0, keepdim=True)
    first = int(torch.argmax(torch.linalg.norm(points - centroid, dim=-1)))
    chosen = [first]
    distances = torch.linalg.norm(points - points[first], dim=-1)
    for _ in range(count - 1):
        nxt = int(torch.argmax(distances))
        chosen.append(nxt)
        distances = torch.minimum(
            distances, torch.linalg.norm(points - points[nxt], dim=-1)
        )
    return torch.tensor(chosen, dtype=torch.long, device=points.device)


def _split(points: Tensor, num_val: int, num_test: int) -> Split:
    num = points.shape[0]
    if num_val < 0 or num_test < 0:
        raise ValueError("num_val and num_test must be non-negative")
    if num_val + num_test > num:
        raise ValueError(
            f"num_val + num_test ({num_val + num_test}) exceeds the {num} "
            f"available samples"
        )
    # Choose validation and test together so that they are mutually separated
    # rather than merely each separated from the training set, then deal them
    # out alternately in selection order. Giving one set a contiguous prefix of
    # the farthest-point order and the other the suffix would make the second
    # set systematically more clustered than the first, and the two would not
    # be comparable.
    held = farthest_point_indices(points, num_val + num_test)
    val_ids, test_ids = [], []
    for rank, index in enumerate(held.tolist()):
        if rank % 2 == 0 and len(test_ids) < num_test:
            test_ids.append(index)
        elif len(val_ids) < num_val:
            val_ids.append(index)
        else:
            test_ids.append(index)

    held_set = set(val_ids) | set(test_ids)

    def to_tensor(ids) -> Tensor:
        return torch.tensor(sorted(ids), dtype=torch.long, device=points.device)

    return Split(
        train=to_tensor(i for i in range(num) if i not in held_set),
        val=to_tensor(val_ids),
        test=to_tensor(test_ids),
    )


def split_views(camera_positions: Tensor, num_val: int, num_test: int) -> Split:
    """Hold out views by camera position.

    Args:
        camera_positions: ``[V, 3]`` camera centres in world space.
        num_val: Validation views.
        num_test: Sealed-test views.

    Returns:
        A :class:`Split` over view indices.
    """
    if camera_positions.ndim != 2 or camera_positions.shape[-1] != 3:
        raise ValueError(
            f"camera_positions must be [V, 3], got {tuple(camera_positions.shape)}"
        )
    return _split(camera_positions, num_val, num_test)


def split_lights(light_directions: Tensor, num_val: int, num_test: int) -> Split:
    """Hold out lights by direction on the sphere.

    Directions rather than positions: two flashes a metre apart along the same
    bearing are nearly the same measurement once the near-field factor has been
    divided out, and splitting on position would treat them as well separated.

    Args:
        light_directions: ``[L, 3]`` unit directions from the subject to each
            light.
        num_val: Validation lights.
        num_test: Sealed-test lights.

    Returns:
        A :class:`Split` over light indices.
    """
    if light_directions.ndim != 2 or light_directions.shape[-1] != 3:
        raise ValueError(
            f"light_directions must be [L, 3], got {tuple(light_directions.shape)}"
        )
    norms = torch.linalg.norm(light_directions, dim=-1)
    if float((norms - 1.0).abs().max()) > 1e-3:
        raise ValueError("light_directions must be unit vectors")
    return _split(light_directions, num_val, num_test)


def split_arc(directions: Tensor, fraction: float = 0.2, *, start: int = 0) -> Split:
    """Hold out a **contiguous** angular sector, for extrapolation.

    The farthest-point splits above hold out samples that are *surrounded* by
    training samples, which measures interpolation. That is the right question
    for a capture where views and lights vary independently, and the wrong one
    for a co-located capture, where holding out a light holds out its view too
    and the two numbers become one number.

    What a co-located capture can still answer is the question relighting
    actually asks: **can the transport predict directions it never saw?**
    Removing a contiguous arc leaves the held-out directions *outside* the
    trained region rather than between trained samples, so the score is an
    extrapolation score. Reported next to a farthest-point interpolation score,
    the difference is the same claim the view/light gap made, on splits this
    data supports.

    Selection is deterministic and takes no seed. Samples are ordered by
    azimuth about the axis of least variance -- the natural traversal order of
    an orbit, recovered from the samples rather than assumed from a filename --
    and a contiguous run of that order becomes the test set.

    Args:
        directions: ``[N, D]`` sample coordinates: camera centres, light
            directions, or anything else with an angular order.
        fraction: Share of the samples to hold out, in ``(0, 1)``.
        start: Which position in the azimuthal order the arc begins at, so a
            second disjoint arc can be taken from the same capture.

    Returns:
        A :class:`Split` whose ``val`` is empty: an arc is one contiguous
        region, and carving a validation set out of it would either break the
        contiguity that makes it an extrapolation test or sit inside the test
        arc and leak.
    """
    if directions.ndim != 2:
        raise ValueError(f"directions must be [N, D], got {tuple(directions.shape)}")
    num = directions.shape[0]
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    count = max(1, int(round(num * fraction)))
    if count >= num:
        raise ValueError(
            f"a fraction of {fraction} would hold out every one of {num} samples"
        )

    order = azimuthal_order(directions)
    held = [int(order[(start + i) % num]) for i in range(count)]
    held_set = set(held)

    def to_tensor(ids) -> Tensor:
        return torch.tensor(sorted(ids), dtype=torch.long, device=directions.device)

    return Split(
        train=to_tensor(i for i in range(num) if i not in held_set),
        val=to_tensor([]),
        test=to_tensor(held),
    )


def azimuthal_order(points: Tensor) -> Tensor:
    """Indices ordering ``points`` by angle about their own axis of least spread.

    An orbit is planar to within its elevation jitter, so the direction the
    samples vary *least* along is the orbit's axis. Recovering it from the data
    rather than assuming world ``+z`` means a capture shot around a horizontal
    axis, or tilted, still orders correctly -- and a handheld pass is never
    exactly level.
    """
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError(
            f"need at least two points of shape [N, D], got {tuple(points.shape)}"
        )
    centred = (points - points.mean(dim=0)).to(torch.float64)
    # Right singular vectors are ordered by decreasing spread, so the last is
    # the axis the samples vary least along.
    _, _, basis = torch.linalg.svd(centred, full_matrices=False)
    axis = basis[-1]
    first, second = basis[0], basis[1]
    angles = torch.atan2(centred @ second, centred @ first)
    del axis  # named for the reader; the plane is what the ordering needs
    return torch.argsort(angles)
