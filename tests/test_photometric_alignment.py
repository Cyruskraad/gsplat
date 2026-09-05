# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Photometric camera refinement: does fixing the registration fix texturing?

The package's headline tradeoff -- blending retains 59% of ground-truth contrast
but wins on pointwise L1, view selection retains ~106% but loses on L1 -- is a
*symptom* of misregistration, not a law. These tests measure whether removing
the cause removes the symptom.

**Views are rendered by ray-casting the same mesh the refinement is given.**
That matters more than it looks. ``_SphereDataset`` ray-traces the *analytic*
sphere while ``_unit_sphere_mesh`` is a polyhedron inscribed in it, and at
resolution 10 the sagitta is ~9% of this pattern's wavelength -- so the
photometric optimum is genuinely not the true pose, and refinement spends its
freedom compensating for the geometry instead. Measured on that fixture,
refinement moved *correct* cameras by 15' and made the bake 3.9% worse on L1,
which reads as "the method harms good poses" when it is really "the method was
handed the wrong surface". Rendering from the mesh removes the confound.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deps():
    pytest.importorskip("open3d", reason="open3d not installed")
    pytest.importorskip("torch", reason="torch not installed")
    for path in (str(REPO_ROOT / "examples"), str(REPO_ROOT / "tests")):
        if path not in sys.path:
            sys.path.insert(0, path)


class _MeshViews:
    """Multi-view-consistent renders of a mesh, with a perturbable reported pose.

    The duck-typed dataset contract the whole package reads is just ``__len__``
    and ``__getitem__ -> {"camtoworld", "K", "image"}``, so this drops into
    every baker and into the refinement unchanged.

    Reuses ``examples/make_synthetic_capture.py``'s renderer and pose helpers,
    which also pins that example: if its rendering or its pose convention
    breaks, these tests fail rather than the example silently rotting.
    """

    def __init__(self, mesh, num_views=8, width=96, pose_error_arcmin=0.0, seed=0):
        import torch

        from make_synthetic_capture import (
            _camera_poses,
            _perturb_rotation,
            render_views,
        )
        from test_texturing import _high_frequency_pattern

        rng = np.random.default_rng(seed)
        focal = 1.35 * width
        K = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, width / 2.0], [0.0, 0.0, 1.0]]
        )
        # Full-sphere coverage, so almost every surface point is seen by
        # several views and the consensus the refinement solves for exists.
        true_poses = _camera_poses(num_views, 3.5, elevation_limit=1.0)
        images = render_views(
            mesh,
            true_poses,
            K,
            width,
            width,
            pattern=_high_frequency_pattern,
            seed=seed,
        )
        reported = true_poses.copy()
        if pose_error_arcmin:
            for i in range(num_views):
                reported[i, :3, :3] = _perturb_rotation(
                    reported[i, :3, :3], pose_error_arcmin, rng
                )
        self.true_poses = true_poses
        self._items = [
            {
                "camtoworld": torch.from_numpy(np.ascontiguousarray(reported[i])),
                "K": torch.from_numpy(K),
                "image": torch.from_numpy(images[i].astype(np.float64)),
            }
            for i in range(num_views)
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def _relative_pose_error_arcmin(true_poses, estimated):
    """Mean pairwise relative rotation error, in arcminutes.

    Deliberately *relative* rather than absolute. Photometric alignment fixes
    the cameras only up to a global rigid motion -- rotating every camera
    together and the surface with them costs nothing -- so it is anchored to
    one camera, whose own error it can never see. The absolute error therefore
    floors at the anchor's. What actually degrades a texture is the cameras
    disagreeing *with each other*, and that is what this measures; it is
    gauge-free by construction.
    """
    errors = []
    n = len(true_poses)
    for i in range(n):
        for j in range(i + 1, n):
            true_rel = true_poses[i][:3, :3].T @ true_poses[j][:3, :3]
            est_rel = estimated[i][:3, :3].T @ estimated[j][:3, :3]
            delta = true_rel.T @ est_rel
            cos = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
            errors.append(np.degrees(np.arccos(cos)) * 60.0)
    return float(np.mean(errors))


def _refine(mesh, dataset, **overrides):
    from gsplat.photogrammetry.photometric_alignment import (
        refine_camera_poses_photometric,
    )

    kwargs = dict(
        num_levels=3, iterations=60, alternations=3, num_points=4000, lr=3e-4, seed=0
    )
    kwargs.update(overrides)
    return refine_camera_poses_photometric(mesh, dataset, **kwargs)


@pytest.fixture(scope="module")
def sphere():
    _deps()
    from test_mesh_extraction import _unit_sphere_mesh

    return _unit_sphere_mesh(resolution=12)


@pytest.fixture(scope="module")
def recovered(sphere):
    """One 45' refinement, shared -- it is the expensive part of this file."""
    _deps()
    dataset = _MeshViews(sphere, pose_error_arcmin=45.0)
    camtoworlds, stats = _refine(sphere, dataset)
    return dataset, camtoworlds, stats


# ---------------------------------------------------------------------------
# 1. Recovery
# ---------------------------------------------------------------------------


def test_refinement_recovers_injected_pose_error(recovered):
    """45' of simulated residual SfM error, largely undone."""
    dataset, camtoworlds, stats = recovered
    reported = [dataset[i]["camtoworld"].numpy() for i in range(len(dataset))]

    # Premise, measured: the injected error must actually be there, or every
    # assertion below is satisfied by a no-op. 45' about a random axis per
    # camera shows up as ~56' of *pairwise* disagreement.
    before = _relative_pose_error_arcmin(dataset.true_poses, reported)
    assert before > 40.0, f"nothing to recover: {before:.2f} arcmin"

    # Measured: 55.92' -> 24.75', a 2.26x reduction.
    after = _relative_pose_error_arcmin(dataset.true_poses, camtoworlds)
    assert after < before / 1.8, (before, after)

    # Measured: 0.1449 -> 0.0414, a 3.5x reduction.
    residual_before = stats["mean_photometric_residual_before"]
    residual_after = stats["mean_photometric_residual_after"]
    assert residual_before > 0.05, residual_before
    assert residual_after < residual_before / 2.5, (residual_before, residual_after)

    # The anchor camera is held exactly fixed: that is what fixes the gauge.
    anchor = stats["anchor_image_idx"]
    assert np.allclose(camtoworlds[anchor], reported[anchor], atol=1e-9)


# ---------------------------------------------------------------------------
# 2. The headline: the tradeoff is a symptom
# ---------------------------------------------------------------------------


def test_refinement_collapses_the_blending_versus_view_selection_tradeoff(sphere):
    """ISSUES.md § 4.1's tradeoff, measured before and after refinement.

    Before: blending low-passes the detail away (59% of ground-truth contrast)
    but wins pointwise, because single-view sampling *displaces* detail and a
    displaced-but-sharp texture scores worse on L1 than a blurred one.

    After: blending recovers to the ceiling perfectly-registered cameras reach,
    and **the L1 ranking flips to match that ceiling's** -- view selection wins,
    as it does with exact poses. So the tradeoff was never a property of the two
    methods; it was misregistration, and this measures it going away.

    Note the prediction this *falsifies*. "Blending retains >= 95% after
    refinement" is unreachable: with exact poses it retains ~93% on this
    fixture, so 95% is above the ceiling, and a bake reading much over 100%
    is showing added noise rather than recovered fidelity. The assertion is
    therefore phrased against the measured ceiling, not against a round number.
    """
    _deps()
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.photometric_alignment import _PosedPyramidDataset
    from gsplat.photogrammetry.texturing import (
        bake_texture_atlas,
        bake_texture_atlas_view_selected,
    )
    from test_texturing import _ground_truth_atlas, _high_frequency_pattern

    size = 128
    noisy = _MeshViews(sphere, pose_error_arcmin=45.0)
    perfect = _MeshViews(sphere, pose_error_arcmin=0.0)

    # Bake FIRST, so the mesh acquires one UV layout every later bake reuses --
    # `compute_uvatlas` is non-deterministic, so a ground-truth atlas built on
    # its own unwrap is compared against a *different* layout and every number
    # below becomes noise. (This is ISSUES.md #10, and it bit while writing
    # this test: the L1 figures were unrepeatable until the order was fixed.)
    _, blend_before = bake_texture_atlas(sphere, noisy, texture_size=size)
    _, select_before, _ = bake_texture_atlas_view_selected(
        sphere, noisy, texture_size=size
    )
    camtoworlds, _ = _refine(sphere, noisy)
    refined = _PosedPyramidDataset(noisy, camtoworlds, levels=0)
    _, blend_after = bake_texture_atlas(sphere, refined, texture_size=size)
    _, select_after, _ = bake_texture_atlas_view_selected(
        sphere, refined, texture_size=size
    )
    _, blend_perfect = bake_texture_atlas(sphere, perfect, texture_size=size)
    # Ground truth LAST, on that same layout.
    truth, covered = _ground_truth_atlas(sphere, size, _high_frequency_pattern)

    truth_grad = atlas_sharpness(truth, covered)["mean_gradient"]
    truth_flat = truth[covered] / 255.0

    def contrast(atlas):
        return atlas_sharpness(atlas, covered)["mean_gradient"] / truth_grad

    def l1(atlas):
        return float(np.abs(atlas[covered] / 255.0 - truth_flat).mean())

    # Premise: the documented tradeoff must reproduce here, or there is nothing
    # to collapse. Measured on this fixture: blending 63.3% of the truth's
    # contrast, view selection 92.3%, and blending ahead on L1
    # (0.2114 vs 0.2424) -- the same shape as ISSUES.md § 4.1's 59% / 106% and
    # 0.171 / 0.199, at this fixture's smaller scale.
    assert contrast(blend_before) < 0.70, contrast(blend_before)
    assert contrast(select_before) > 0.88, contrast(select_before)
    assert l1(blend_before) < l1(select_before), (
        l1(blend_before),
        l1(select_before),
    )

    # After refinement blending reaches the ceiling exact poses achieve:
    # measured 83.7% against a ceiling of 84.2%, i.e. 99.3% of it.
    ceiling = contrast(blend_perfect)
    assert contrast(blend_after) > 0.78, contrast(blend_after)
    assert contrast(blend_after) > 0.95 * ceiling, (contrast(blend_after), ceiling)
    # ... and is far better than it was (63.3% -> 83.7%, a 1.32x gain).
    assert contrast(blend_after) > 1.25 * contrast(blend_before)

    # Both bakes get much closer to the truth: blending 0.2114 -> 0.0911
    # (2.3x), view selection 0.2424 -> 0.0814 (3.0x).
    assert l1(blend_after) < l1(blend_before) / 2.0
    assert l1(select_after) < l1(select_before) / 2.0

    # And the ranking flips: view selection, which lost on L1 before, now wins,
    # which is the ordering exact poses produce. That is the tradeoff gone.
    assert l1(select_after) < l1(blend_after), (l1(select_after), l1(blend_after))


# ---------------------------------------------------------------------------
# 3. Do no harm
# ---------------------------------------------------------------------------


def test_refinement_does_not_degrade_already_registered_cameras(sphere):
    """A broken objective passes the recovery test and fails this one.

    Anything that merely *moves* cameras until the re-baked colour agrees with
    itself can drive the residual down -- a blurrier consensus is easier to
    agree with. The check that separates a working objective from that is
    whether it leaves correct poses alone.
    """
    _deps()
    from gsplat.photogrammetry.metrics import atlas_sharpness
    from gsplat.photogrammetry.photometric_alignment import _PosedPyramidDataset
    from gsplat.photogrammetry.texturing import bake_texture_atlas
    from test_texturing import _ground_truth_atlas, _high_frequency_pattern

    size = 128
    dataset = _MeshViews(sphere, pose_error_arcmin=0.0)
    _, before = bake_texture_atlas(sphere, dataset, texture_size=size)
    camtoworlds, stats = _refine(sphere, dataset)
    refined = _PosedPyramidDataset(dataset, camtoworlds, levels=0)
    _, after = bake_texture_atlas(sphere, refined, texture_size=size)
    truth, covered = _ground_truth_atlas(sphere, size, _high_frequency_pattern)

    truth_flat = truth[covered] / 255.0
    l1_before = float(np.abs(before[covered] / 255.0 - truth_flat).mean())
    l1_after = float(np.abs(after[covered] / 255.0 - truth_flat).mean())
    # Not "no worse at all": the solve is stochastic in its point sampling and
    # the mesh is a polyhedron approximating a sphere, so a little motion is
    # legitimate. It must not *cost* anything measurable.
    assert l1_after < l1_before * 1.05, (l1_before, l1_after)

    grad_before = atlas_sharpness(before, covered)["mean_gradient"]
    grad_after = atlas_sharpness(after, covered)["mean_gradient"]
    assert grad_after > 0.95 * grad_before, (grad_before, grad_after)

    # And it should barely move: correct cameras have nothing to correct. An
    # order of magnitude less motion than the 45' case, which moves ~43'.
    assert stats["mean_pose_correction_arcmin"] < 10.0, stats[
        "mean_pose_correction_arcmin"
    ]


# ---------------------------------------------------------------------------
# 4. The pyramid, pinned at its call site
# ---------------------------------------------------------------------------


def test_the_pyramid_earns_its_place_only_where_the_objective_aliases(sphere):
    """Coarse-to-fine must beat *equal-work* single-scale, or it is decoration.

    This test was wrong on the first attempt, in precisely the way
    ISSUES.md § 5 warns about, and the mutation caught it. It compared
    ``num_levels=1`` against ``num_levels=3`` at the same ``alternations``,
    which is not a comparison of coarse-to-fine against single-scale -- it is a
    comparison of 3 optimisation rounds against 9. Forcing every level to full
    resolution (the mutation) left the suite green, because the extra *rounds*
    were doing the work the pyramid was being credited with. Measured at 45':
    single-scale with 3 rounds fails (65'), single-scale with 9 rounds reaches
    23.6', and the 3-level pyramid with 3 rounds each reaches 24.75'. At equal
    work the pyramid is not better at all.

    **So the received claim -- "a photometric objective has a tiny basin of
    convergence, single-scale will not recover 45'" -- is false on this
    fixture, and the reason is measurable.** The detail here has a wavelength of
    5.2 px in the image; 45' of pose error displaces a projection by 1.70 px,
    which is well inside the half-wavelength (2.6 px) where the objective is
    still unambiguous. The alternation's re-baked target, not the pyramid, is
    what carries that case.

    The pyramid does earn its place -- just further out, exactly where the
    physics says it should. At 90' the displacement is 3.39 px, past the
    half-wavelength, and the objective aliases: equal-work single-scale lands
    at 183' where the pyramid lands at 133'. That band is what this pins.

    Note what is *not* claimed: at 90' neither variant actually recovers
    anything (both end worse than the 112' they started from). The method's
    working range on this fixture is a displacement below half the detail's
    wavelength. The pyramid makes failure gentler there, not absent.
    """
    _deps()
    dataset = _MeshViews(sphere, pose_error_arcmin=90.0)
    reported = [dataset[i]["camtoworld"].numpy() for i in range(len(dataset))]
    before = _relative_pose_error_arcmin(dataset.true_poses, reported)
    assert before > 80.0, before

    # Equal total work: 3 levels x 3 rounds against 1 level x 9 rounds.
    pyramid, _ = _refine(sphere, dataset, num_levels=3, alternations=3)
    single, _ = _refine(sphere, dataset, num_levels=1, alternations=9)
    pyramid_error = _relative_pose_error_arcmin(dataset.true_poses, pyramid)
    single_error = _relative_pose_error_arcmin(dataset.true_poses, single)

    # Measured: 133.3' with the pyramid, 182.7' without.
    assert pyramid_error < single_error / 1.25, (pyramid_error, single_error)


# ---------------------------------------------------------------------------
# 5. The differentiable sampler against the one every bake reads through
# ---------------------------------------------------------------------------


def test_torch_bilinear_matches_the_numpy_sampler():
    """The refinement must optimise the same image the bake reads.

    `_torch_bilinear` is a re-expression of `texturing._bilinear` rather than a
    call, because the numpy original cannot carry a gradient back to the pixel
    coordinate -- and that gradient is the entire photometric signal. A
    re-expression that drifts from its original (a half-pixel offset, wrapping
    instead of clamping) would refine the cameras against a subtly different
    image than the one they are then baked through, so the two are pinned
    together here.
    """
    _deps()
    import torch

    from gsplat.photogrammetry.photometric_alignment import _torch_bilinear
    from gsplat.photogrammetry.texturing import _bilinear

    rng = np.random.default_rng(0)
    image = rng.random((17, 23, 3))
    # Interior, both borders, and well outside the frame in every direction --
    # the clamp is the part most likely to drift.
    uv = np.array(
        [
            [5.5, 7.25],
            [0.0, 0.0],
            [22.999, 16.999],
            [-8.0, 4.0],
            [40.0, 4.0],
            [4.0, -8.0],
            [4.0, 40.0],
            [11.5, 8.5],
        ]
    )
    expected = _bilinear(image, uv)
    got = _torch_bilinear(
        torch.from_numpy(image).unsqueeze(0),
        torch.zeros(len(uv), dtype=torch.long),
        torch.from_numpy(uv),
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-12)

    # And it is differentiable in uv, which is the only reason it exists.
    uv_t = torch.tensor([[5.5, 7.25]], dtype=torch.float64, requires_grad=True)
    out = _torch_bilinear(
        torch.from_numpy(image).unsqueeze(0), torch.zeros(1, dtype=torch.long), uv_t
    )
    out.sum().backward()
    assert uv_t.grad is not None and torch.any(uv_t.grad != 0)
