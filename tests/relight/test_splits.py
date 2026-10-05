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

"""Held-out splits: disjoint, deterministic, and actually spread out.

The held-out-*light* split is the one that decides whether the model learned
transport or memorised illuminations, so "the held-out lights are far from the
training lights" has to be a measured property rather than an intention.
"""

import pytest
import torch

from gsplat.relight.functional import (
    farthest_point_indices,
    fibonacci_sphere,
    split_lights,
    split_views,
)

DTYPE = torch.float64


def _camera_ring(count=60, radius=1.5):
    angle = torch.linspace(0.0, 2.0 * torch.pi, count + 1, dtype=DTYPE)[:count]
    return torch.stack(
        [
            radius * torch.cos(angle),
            radius * torch.sin(angle),
            torch.full_like(angle, 0.4),
        ],
        dim=-1,
    )


def test_splits_are_disjoint_and_cover_everything():
    views = _camera_ring(60)
    split = split_views(views, num_val=8, num_test=6)
    assert len(split.train) == 46 and len(split.val) == 8 and len(split.test) == 6
    combined = torch.cat([split.train, split.val, split.test])
    assert len(set(combined.tolist())) == 60
    assert sorted(combined.tolist()) == list(range(60))


def test_splits_are_deterministic_with_no_seed():
    views = _camera_ring(60)
    first = split_views(views, num_val=8, num_test=6)
    second = split_views(views, num_val=8, num_test=6)
    assert torch.equal(first.train, second.train)
    assert torch.equal(first.val, second.val)
    assert torch.equal(first.test, second.test)


def test_indices_are_returned_sorted():
    """Sorted output keeps a split manifest diffable between runs."""
    split = split_views(_camera_ring(40), num_val=5, num_test=5)
    for part in (split.train, split.val, split.test):
        assert torch.equal(part, torch.sort(part).values)


def test_held_out_lights_are_further_apart_than_a_contiguous_block():
    """Farthest-point selection must beat the obvious alternative.

    Taking the first ``k`` lights is what a naive split does, and on any capture
    shot in a sweep those are neighbours -- so a held-out light sits next to a
    training light and the gate measures nothing.
    """
    directions = fibonacci_sphere(120, dtype=DTYPE)
    split = split_lights(directions, num_val=0, num_test=10)

    def min_pairwise(indices):
        picked = directions[indices]
        gram = picked @ picked.transpose(0, 1)
        gram.fill_diagonal_(-1.0)
        return float(torch.arccos(torch.clamp(gram.max(), -1.0, 1.0)))

    contiguous = torch.arange(10)
    assert min_pairwise(split.test) > 3.0 * min_pairwise(contiguous)


def test_validation_and_test_are_comparably_spread():
    """Dealing alternately, rather than prefix/suffix, keeps the two comparable.

    If validation were the farthest-point prefix and test the remainder, the
    test set would be systematically more clustered and the two numbers could
    not be read against each other.
    """
    directions = fibonacci_sphere(150, dtype=DTYPE)
    split = split_lights(directions, num_val=12, num_test=12)

    def mean_nearest_neighbour_angle(indices):
        picked = directions[indices]
        gram = picked @ picked.transpose(0, 1)
        gram.fill_diagonal_(-1.0)
        return float(
            torch.arccos(torch.clamp(gram.max(dim=-1).values, -1.0, 1.0)).mean()
        )

    val_spread = mean_nearest_neighbour_angle(split.val)
    test_spread = mean_nearest_neighbour_angle(split.test)
    assert abs(val_spread - test_spread) < 0.4 * max(val_spread, test_spread)


def test_farthest_point_starts_from_the_extreme_sample():
    """Determinism comes from the first pick, so the first pick is pinned."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [5.0, 5.0, 5.0]],
        dtype=DTYPE,
    )
    assert int(farthest_point_indices(points, 1)[0]) == 3


def test_farthest_point_returns_distinct_indices():
    points = _camera_ring(50)
    picked = farthest_point_indices(points, 20)
    assert len(set(picked.tolist())) == 20


def test_requesting_every_sample_leaves_an_empty_training_set():
    views = _camera_ring(10)
    split = split_views(views, num_val=4, num_test=6)
    assert len(split.train) == 0


def test_requesting_nothing_holds_out_nothing():
    views = _camera_ring(10)
    split = split_views(views, num_val=0, num_test=0)
    assert len(split.train) == 10
    assert len(split.val) == 0 and len(split.test) == 0


# --- guards -----------------------------------------------------------------


def test_split_views_rejects_positions_that_are_not_three_vectors():
    with pytest.raises(ValueError, match=r"camera_positions must be \[V, 3\]"):
        split_views(torch.zeros(10, 2, dtype=DTYPE), 1, 1)


def test_split_lights_rejects_directions_that_are_not_normalised():
    with pytest.raises(ValueError, match="must be unit vectors"):
        split_lights(torch.full((10, 3), 0.5, dtype=DTYPE), 1, 1)


def test_split_rejects_holding_out_more_than_exists():
    with pytest.raises(ValueError, match="exceeds the 10 available"):
        split_views(_camera_ring(10), num_val=6, num_test=6)


def test_split_rejects_a_negative_count():
    with pytest.raises(ValueError, match="must be non-negative"):
        split_views(_camera_ring(10), num_val=-1, num_test=2)


def test_farthest_point_rejects_a_count_above_the_population():
    with pytest.raises(ValueError, match=r"count must be in \[0, 5\]"):
        farthest_point_indices(torch.zeros(5, 3, dtype=DTYPE), 6)


def test_farthest_point_rejects_points_of_the_wrong_rank():
    with pytest.raises(ValueError, match=r"points must be \[N, D\]"):
        farthest_point_indices(torch.zeros(5, dtype=DTYPE), 2)
