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

"""What the evaluation harness has to get right.

Three things, in descending order of how badly they would hurt:

1. The metrics are the metrics. A PSNR that is off by a constant still ranks
   runs correctly and is therefore *worse* than one that is obviously wrong,
   because it will be quoted. SSIM is checked against a direct transcription of
   Wang et al. written in this file -- nested loops, no convolution, no
   cleverness -- and PSNR against its own definition computed by hand.
2. The gate is not passable by accident. A diverged model, an empty split and a
   mismatched domain must each fail it.
3. The ledger keeps every row when several processes finish at once.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from atlas.eval import (  # noqa: E402
    DEFAULT_DOMAINS,
    GATE_PSNR_GAP_DB,
    RelightingReport,
    SheetEntry,
    SplitMetrics,
    TONEMAPS,
    align_exposure,
    constant_baseline_psnr,
    append_evaluation,
    colourise,
    comparison_sheet,
    evaluate_image,
    evaluate_set,
    psnr,
    ssim,
    ssim_map,
    tonemap,
)
from atlas.run import read_ledger  # noqa: E402


def _rng(seed: int = 0) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    return generator


def _image(height=24, width=28, channels=3, seed=0, scale=1.0) -> torch.Tensor:
    return torch.rand(height, width, channels, generator=_rng(seed)) * scale


# --- the reference SSIM, written to be obviously correct --------------------


def _reference_ssim_map(x, y, *, window=11, sigma=1.5, data_range=1.0):
    """Wang et al. (2004), transcribed literally.

    One Python loop per window position, the Gaussian weights applied by hand,
    population statistics. Far too slow to use and impossible to get subtly
    wrong, which is the entire point of having it.
    """
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    height, width, channels = x.shape
    offsets = torch.arange(window, dtype=torch.float64) - (window - 1) / 2.0
    weights_1d = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    weights_1d = weights_1d / weights_1d.sum()
    weights = torch.outer(weights_1d, weights_1d)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    out = torch.zeros(
        height - window + 1, width - window + 1, channels, dtype=torch.float64
    )
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for c in range(channels):
                patch_x = x[i : i + window, j : j + window, c]
                patch_y = y[i : i + window, j : j + window, c]
                mean_x = float((weights * patch_x).sum())
                mean_y = float((weights * patch_y).sum())
                var_x = float((weights * patch_x * patch_x).sum()) - mean_x**2
                var_y = float((weights * patch_y * patch_y).sum()) - mean_y**2
                cov = float((weights * patch_x * patch_y).sum()) - mean_x * mean_y
                out[i, j, c] = ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / (
                    (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
                )
    return out


@pytest.mark.parametrize(
    "mix,low,high",
    [(0.1, 0.97, 0.995), (0.5, 0.60, 0.75), (1.0, 0.02, 0.15)],
)
def test_ssim_matches_a_literal_transcription_of_the_paper(mix, low, high):
    """Checked at high, middle and low similarity.

    One pair would not do it: two implementations can agree perfectly on
    near-identical images -- where both return something very close to 1 -- and
    disagree everywhere the metric is actually discriminating. The measured
    values are 0.989, 0.686 and 0.077, and the bands around them are the
    premise: if a change moved the fixture into a corner, the band fails before
    the comparison does.
    """
    reference = _image(20, 22, 3, seed=1)
    predicted = (1 - mix) * reference + mix * _image(20, 22, 3, seed=99)

    slow = _reference_ssim_map(predicted, reference)
    assert low < float(slow.mean()) < high, float(slow.mean())

    fast = ssim_map(predicted.double(), reference.double())
    assert fast.dtype == torch.float64
    assert fast.shape == slow.shape == (10, 12, 3)
    assert torch.allclose(fast, slow, atol=1e-10), float((fast - slow).abs().max())


def test_ssim_returns_the_dtype_it_was_given_however_it_computed():
    """Internally it is float64, because the variance is a difference of two
    nearly equal means and float32 loses several digits doing it."""
    image = _image(16, 16, 3, seed=2)
    assert ssim_map(image, image).dtype == torch.float32
    assert ssim_map(image.double(), image.double()).dtype == torch.float64


def test_ssim_of_an_image_with_itself_is_one():
    image = _image(16, 16, 3, seed=3)
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-9)


def test_ssim_of_two_constant_images_has_the_closed_form_value():
    """With zero variance the contrast and structure terms are exactly 1, so
    SSIM collapses to the luminance term alone and can be written down.

    Float64, because the closed form is exact and float32 resolution at 0.6 is
    6e-8 -- a tolerance that loose would not notice a real error.
    """
    a = torch.full((16, 16, 1), 0.2, dtype=torch.float64)
    b = torch.full((16, 16, 1), 0.6, dtype=torch.float64)
    c1 = (0.01 * 1.0) ** 2
    expected = (2 * 0.2 * 0.6 + c1) / (0.2**2 + 0.6**2 + c1)
    assert ssim(a, b) == pytest.approx(expected, abs=1e-9)


def test_ssim_refuses_an_image_smaller_than_its_window():
    with pytest.raises(ValueError, match="smaller than the 11x11"):
        ssim(torch.zeros(8, 40, 3), torch.zeros(8, 40, 3))


def test_ssim_falls_as_noise_grows():
    reference = _image(24, 24, 3, seed=4)
    scores = [
        ssim(
            (reference + level * torch.rand(24, 24, 3, generator=_rng(5))).clamp(0, 1),
            reference,
        )
        for level in (0.0, 0.05, 0.2, 0.6)
    ]
    assert scores == sorted(scores, reverse=True), scores


# --- PSNR -------------------------------------------------------------------


def test_psnr_equals_its_definition():
    predicted = _image(9, 11, 3, seed=6)
    reference = _image(9, 11, 3, seed=7)
    mse = float(((predicted - reference) ** 2).mean())
    assert psnr(predicted, reference) == pytest.approx(10 * math.log10(1.0 / mse))


def test_psnr_of_a_known_constant_error_is_the_hand_computed_value():
    """A uniform error of 0.1 gives an MSE of exactly 0.01, so PSNR is 20 dB.

    In float64, because 0.5 + 0.1 - 0.5 is not 0.1 in float32 and the test
    would then be measuring the fixture's rounding rather than the metric.
    """
    reference = torch.full((12, 12, 3), 0.5, dtype=torch.float64)
    assert psnr(reference + 0.1, reference) == pytest.approx(20.0, abs=1e-9)


def test_psnr_scales_with_the_declared_data_range():
    reference = torch.full((12, 12, 3), 0.5)
    at_one = psnr(reference + 0.1, reference, data_range=1.0)
    at_two = psnr(reference + 0.1, reference, data_range=2.0)
    assert at_two - at_one == pytest.approx(20 * math.log10(2.0), abs=1e-9)


def test_identical_images_give_infinite_psnr_rather_than_a_capped_number():
    image = _image(8, 8, 3, seed=8)
    assert psnr(image, image) == float("inf")


def test_a_shape_mismatch_names_both_shapes():
    with pytest.raises(ValueError, match=r"\(4, 4, 3\) and \(4, 5, 3\)"):
        psnr(torch.zeros(4, 4, 3), torch.zeros(4, 5, 3))


# --- masks ------------------------------------------------------------------


def test_an_all_ones_mask_gives_the_unmasked_answer():
    """The gate for the whole masking path. If these ever disagree, every
    masked number in the repo is measuring something other than it claims."""
    predicted = _image(24, 26, 3, seed=9)
    reference = _image(24, 26, 3, seed=10)
    ones = torch.ones(24, 26)
    assert psnr(predicted, reference, mask=ones) == pytest.approx(
        psnr(predicted, reference), abs=1e-9
    )
    assert ssim(predicted, reference, mask=ones) == pytest.approx(
        ssim(predicted, reference), abs=1e-9
    )
    masked = evaluate_image(predicted, reference, mask=ones)
    unmasked = evaluate_image(predicted, reference)
    assert set(masked) == set(unmasked)
    for key in unmasked:
        assert masked[key] == pytest.approx(unmasked[key], abs=1e-9), key


def test_a_mask_actually_excludes_what_it_excludes():
    """Corrupting only the masked-out region must not move the score."""
    reference = _image(24, 26, 3, seed=11)
    mask = torch.zeros(24, 26)
    mask[:, :13] = 1.0
    predicted = reference.clone()
    predicted[:, 13:] = 99.0
    assert psnr(predicted, reference, mask=mask) == float("inf")


def test_a_mask_that_selects_nothing_is_refused_rather_than_averaged():
    with pytest.raises(ValueError, match="selects no pixels"):
        psnr(_image(12, 12), _image(12, 12, seed=1), mask=torch.zeros(12, 12))


def test_a_mask_of_the_wrong_shape_names_the_shape_it_wanted():
    with pytest.raises(ValueError, match=r"must be \[12, 12\]"):
        psnr(torch.zeros(12, 12, 3), torch.zeros(12, 12, 3), mask=torch.ones(8, 8))


def test_a_mask_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        psnr(
            torch.zeros(12, 12, 3),
            torch.zeros(12, 12, 3),
            mask=torch.full((12, 12), 2.0),
        )


# --- tonemaps ---------------------------------------------------------------


def test_every_tonemap_is_monotonic_and_fixes_black():
    values = torch.linspace(0.0, 40.0, 400)
    for name, curve in TONEMAPS.items():
        mapped = curve(values)
        assert float(mapped[0]) == pytest.approx(0.0, abs=1e-7), name
        assert bool((mapped[1:] >= mapped[:-1]).all()), name


def test_the_mu_law_maps_white_to_white():
    assert float(tonemap(torch.tensor([1.0]), "mu")) == pytest.approx(1.0, abs=1e-12)


def test_the_mu_law_keeps_highlights_above_white_rather_than_clipping_them():
    """The reason 'mu' is the primary domain: an sRGB PSNR cannot see a
    highlight error at all, because both images clip to the same white. A
    two-stop difference at eight times white is the kind of error a relighting
    model makes, and one of these two domains reports it."""
    bright = torch.tensor([[[8.0]]])
    brighter = torch.tensor([[[16.0]]])
    assert float(tonemap(brighter, "mu")) > float(tonemap(bright, "mu")) + 0.05
    assert float(tonemap(brighter, "srgb")) == float(tonemap(bright, "srgb"))


def test_a_bounded_domain_never_exceeds_the_data_range_it_declares():
    """PSNR divides by it and SSIM squares it into its constants, so a domain
    that overshoots its declared white reports numbers that are slightly wrong
    in a direction nobody would think to check."""
    values = torch.cat([torch.linspace(0.0, 1.0, 512), torch.tensor([2.0, 1e4])])
    for name in ("srgb", "gamma22", "reinhard"):
        curve = TONEMAPS[name]
        assert float(curve(values).max()) <= curve.data_range, name
    assert float(TONEMAPS["srgb"](torch.tensor([1.0]))) == pytest.approx(1.0, abs=1e-7)


def test_a_raw_hdr_psnr_is_dominated_by_the_highlight_and_the_mu_law_is_not():
    """The measured premise for computing metrics in a tonemapped domain.

    Two predictions of the same scene: one loses the diffuse response
    completely, the other is 10% off on a single specular pixel. On linear
    radiance the second scores *worse*, which would rank a broken model first.
    """
    reference = torch.full((16, 16, 3), 0.3)
    reference[8, 8] = 400.0

    lost_diffuse = reference.clone()
    lost_diffuse[reference < 1.0] = 0.0
    bad_highlight = reference.clone()
    bad_highlight[8, 8] = 440.0

    linear = TONEMAPS["linear"]
    assert psnr(linear(lost_diffuse), linear(reference)) > psnr(
        linear(bad_highlight), linear(reference)
    )
    mu = TONEMAPS["mu"]
    assert psnr(mu(lost_diffuse), mu(reference)) < psnr(
        mu(bad_highlight), mu(reference)
    )


def test_an_unknown_domain_lists_the_known_ones():
    with pytest.raises(KeyError, match="reinhard"):
        tonemap(torch.zeros(4), "filmic")


# --- exposure alignment -----------------------------------------------------


def test_alignment_recovers_a_scale_it_was_given():
    reference = _image(16, 16, 3, seed=12) + 0.05
    scaled, factor = align_exposure(reference * 0.37, reference)
    assert float(factor.mean()) == pytest.approx(1.0 / 0.37, rel=1e-6)
    assert torch.allclose(scaled, reference, atol=1e-6)


def test_a_global_alignment_does_not_absorb_a_colour_cast():
    reference = _image(16, 16, 3, seed=13) + 0.05
    cast = reference * torch.tensor([1.4, 1.0, 0.7])
    _, one = align_exposure(cast, reference, per_channel=False)
    _, three = align_exposure(cast, reference, per_channel=True)
    assert float(one.std()) == pytest.approx(0.0, abs=1e-9)
    assert float(three.std()) > 0.1
    globally_scaled, _ = align_exposure(cast, reference)
    per_channel_scaled, _ = align_exposure(cast, reference, per_channel=True)
    assert psnr(per_channel_scaled, reference) > psnr(globally_scaled, reference)


def test_alignment_leaves_an_all_zero_prediction_alone_rather_than_dividing_by_zero():
    reference = _image(8, 8, 3, seed=14) + 0.1
    scaled, factor = align_exposure(torch.zeros(8, 8, 3), reference)
    assert torch.isfinite(factor).all()
    assert float(scaled.abs().max()) == 0.0


def test_the_fitted_scale_is_reported_so_it_cannot_hide():
    reference = _image(16, 16, 3, seed=15) + 0.05
    metrics = evaluate_image(reference * 0.5, reference, exposure_align=True)
    assert metrics["exposure_scale"] == pytest.approx(2.0, rel=1e-5)
    assert metrics["psnr/mu"] > 100.0
    assert "exposure_scale" not in evaluate_image(reference * 0.5, reference)


# --- divergence -------------------------------------------------------------


def test_a_non_finite_prediction_is_counted_rather_than_crashing_the_run():
    reference = _image(12, 12, 3, seed=16) + 0.1
    predicted = reference.clone()
    predicted[0, 0, 0] = float("nan")
    predicted[1, 1, :] = float("inf")
    metrics = evaluate_image(predicted, reference)
    assert metrics["non_finite"] == pytest.approx(4 / (12 * 12 * 3))
    assert math.isfinite(metrics["psnr/mu"])


def test_a_non_finite_reference_is_a_data_fault_and_raises():
    reference = _image(12, 12, 3, seed=17)
    reference[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="data fault"):
        evaluate_image(reference.clone(), reference)


# --- evaluate_set -----------------------------------------------------------


def test_a_set_averages_over_images_and_sums_pixels():
    reference = _image(16, 16, 3, seed=18) + 0.1
    pairs = [(reference * 1.1, reference), (reference * 0.9, reference)]
    result = evaluate_set(pairs, name="held_out_view")
    assert result.count == 2
    singles = [evaluate_image(p, r)["psnr/mu"] for p, r in pairs]
    assert result["psnr/mu"] == pytest.approx(sum(singles) / 2)
    assert result["pixels"] == pytest.approx(2 * 16 * 16 * 3)


def test_an_empty_set_is_refused_by_name():
    with pytest.raises(ValueError, match="'held_out_light' split is empty"):
        evaluate_set([], name="held_out_light")


def test_a_badly_shaped_entry_says_what_it_wanted():
    with pytest.raises(ValueError, match=r"\(pred, ref\) or \(pred, ref, mask\)"):
        evaluate_set([(torch.zeros(12, 12, 3),)])


def test_a_set_accepts_a_per_image_mask():
    reference = _image(16, 16, 3, seed=19) + 0.1
    mask = torch.zeros(16, 16)
    mask[:8] = 1.0
    predicted = reference.clone()
    predicted[8:] = 42.0
    assert evaluate_set([(predicted, reference, mask)])["psnr/mu"] == float("inf")


# --- the report and the gate ------------------------------------------------


def _report(view_psnr: float, light_psnr: float, **kwargs) -> RelightingReport:
    def split(name, value):
        return SplitMetrics(
            name=name,
            count=kwargs.pop(f"{name}_count", 4),
            metrics={
                "psnr/mu": value,
                "ssim/mu": 0.9,
                "lpips/srgb": 0.1,
                "non_finite": kwargs.get(f"{name}_non_finite", 0.0),
            },
        )

    return RelightingReport(
        held_out_view=split("held_out_view", view_psnr),
        held_out_light=split("held_out_light", light_psnr),
    )


def test_the_gate_passes_when_the_two_splits_agree():
    verdict = _report(31.0, 30.6).gate()
    assert verdict.passed and bool(verdict)
    assert verdict.gap_db == pytest.approx(0.4)


def test_the_gate_fails_when_the_light_split_trails_and_says_what_that_means():
    verdict = _report(31.0, 27.0).gate()
    assert not verdict
    assert verdict.gap_db == pytest.approx(4.0)
    assert "not transport" in " ".join(verdict.reasons)


def test_the_gate_is_one_sided_because_a_better_light_split_is_not_a_failure():
    assert _report(27.0, 31.0).gate().passed


def test_the_gate_is_exactly_one_decibel_wide():
    """ "Within 1 dB" includes 1 dB. 31.0 - 30.0 is exactly 1.0 in binary, so
    this is the boundary itself and not a value near it."""
    assert GATE_PSNR_GAP_DB == 1.0
    assert _report(31.0, 30.0).gate().passed
    assert not _report(31.0, 30.0 - 1e-6).gate().passed
    assert not _report(31.0, 29.9).gate().passed


def test_a_diverged_model_cannot_pass_the_gate_by_scoring_its_own_zeros():
    """Non-finite pixels are scored as zero, which against an unlit background
    is a *perfect* score. A gate that only read the metric could be passed by
    diverging, so it reads the divergence fraction too."""
    report = _report(31.0, 30.9, held_out_light_non_finite=1e-6)
    verdict = report.gate()
    assert not verdict
    assert "diverged" in " ".join(verdict.reasons)


def test_an_empty_split_fails_the_gate_rather_than_passing_by_default():
    verdict = _report(31.0, 30.9, held_out_light_count=0).gate()
    assert not verdict
    assert "empty" in " ".join(verdict.reasons)


def test_a_missing_domain_fails_rather_than_silently_comparing_nothing():
    report = RelightingReport(
        held_out_view=SplitMetrics("held_out_view", 3, {"psnr/srgb": 30.0}),
        held_out_light=SplitMetrics("held_out_light", 3, {"psnr/mu": 30.0}),
    )
    verdict = report.gate(domain="mu")
    assert not verdict
    assert "domains do not match" in " ".join(verdict.reasons)


def test_a_positive_gap_always_means_the_light_split_did_worse():
    """Including for LPIPS, where the raw numbers run the other way."""
    report = RelightingReport(
        held_out_view=SplitMetrics("v", 2, {"psnr/mu": 30.0, "lpips/srgb": 0.10}),
        held_out_light=SplitMetrics("l", 2, {"psnr/mu": 28.0, "lpips/srgb": 0.18}),
    )
    assert report.gaps["psnr/mu"] == pytest.approx(2.0)
    assert report.gaps["lpips/srgb"] == pytest.approx(0.08)


def test_informational_fields_are_not_reported_as_gaps():
    assert "non_finite" not in _report(31.0, 30.0).gaps
    assert "pixels" not in _report(31.0, 30.0).gaps


def test_the_table_shows_both_splits_and_the_verdict():
    text = _report(31.0, 27.0).format_table()
    assert "held-out view" in text and "held-out light" in text
    assert "psnr/mu" in text and "FAIL" in text
    assert "positive gap = the held-out-light split did worse" in text


def test_the_table_survives_an_infinite_metric():
    assert "inf" in _report(float("inf"), 30.0).format_table()


# --- the ledger -------------------------------------------------------------


def test_an_evaluation_lands_in_the_ledger_as_one_flat_row(tmp_path):
    ledger = tmp_path / "results.jsonl"
    row = append_evaluation(
        ledger,
        _report(31.0, 30.5),
        run="angel-20260918T120000Z-abcd1234",
        config_hash="f" * 64,
        step=30000,
        provenance={"git_commit": "cafe", "git_dirty": False, "hostname": "dropped"},
        notes="first real run",
    )
    (stored,) = read_ledger(ledger)
    assert stored == json.loads(json.dumps(row, sort_keys=True, default=str))
    assert stored["heldout_view/psnr/mu"] == 31.0
    assert stored["heldout_light/psnr/mu"] == 30.5
    assert stored["gap/psnr/mu"] == pytest.approx(0.5)
    assert stored["gate/passed"] is True
    assert stored["git_commit"] == "cafe" and "hostname" not in stored
    assert stored["notes"] == "first real run"
    assert all(not isinstance(v, (dict, list)) for v in stored.values())


def test_the_ledger_keeps_every_row_when_processes_finish_at_once(tmp_path):
    """The concurrency that actually happens: two runs ending together, from
    separate processes. Threads would not exercise the same write path."""
    ledger = tmp_path / "results.jsonl"
    code = f"""
import sys
from atlas.eval import RelightingReport, SplitMetrics, append_evaluation

def split(name, value):
    return SplitMetrics(name, 4, {{"psnr/mu": value, "non_finite": 0.0}})

append_evaluation(
    {str(ledger)!r},
    RelightingReport(split("v", 30.0), split("l", 29.8)),
    run="r" + sys.argv[1],
)
"""
    root = str(Path(__file__).resolve().parent.parent)
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(i)],
            cwd=root,
            env={"PYTHONPATH": root, "PATH": "/usr/bin:/bin:/usr/local/bin"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(8)
    ]
    for worker in workers:
        _, err = worker.communicate(timeout=180)
        assert worker.returncode == 0, err.decode()

    rows = read_ledger(ledger)
    assert len(rows) == 8, f"lost {8 - len(rows)} rows"
    assert {r["run"] for r in rows} == {f"r{i}" for i in range(8)}


def test_the_ledger_is_append_only(tmp_path):
    ledger = tmp_path / "results.jsonl"
    append_evaluation(ledger, _report(30.0, 29.0), run="first")
    first_bytes = ledger.read_bytes()
    append_evaluation(ledger, _report(31.0, 30.5), run="second")
    assert ledger.read_bytes().startswith(first_bytes)
    assert [r["run"] for r in read_ledger(ledger)] == ["first", "second"]


def test_a_later_run_may_add_a_metric_an_earlier_one_did_not_have(tmp_path):
    ledger = tmp_path / "results.jsonl"
    append_evaluation(ledger, _report(30.0, 29.0), run="first")
    append_evaluation(ledger, _report(30.0, 29.0), run="second", lpips_net="alex")
    rows = read_ledger(ledger)
    assert "lpips_net" not in rows[0] and rows[1]["lpips_net"] == "alex"


# --- the comparison sheet ---------------------------------------------------


def test_a_sheet_is_a_readable_png_with_a_machine_readable_sidecar(tmp_path):
    from atlas.imageio import read_png

    reference = _image(20, 24, 3, seed=20)
    entries = [
        SheetEntry("VIEW 03 / LIGHT 11", reference, reference * 0.8),
        SheetEntry("VIEW 07 / LIGHT 02", reference, reference + 0.2),
    ]
    sidecar = comparison_sheet(tmp_path / "sheet.png", entries)

    image = read_png(tmp_path / "sheet.png")
    assert image.shape[-1] == 3
    assert image.shape[1] >= 3 * 24
    assert sidecar == json.loads((tmp_path / "sheet.json").read_text())
    assert [r["label"] for r in sidecar["rows"]] == [e.label for e in entries]
    assert sidecar["error_scale"] > 0


def test_every_row_of_a_sheet_shares_one_error_scale(tmp_path):
    """Per-row normalisation would make a row that is barely wrong look exactly
    as bad as one that is completely wrong, which is the opposite of what a
    sheet is for. The tile rectangles come from the sidecar, so this measures
    the error column and not a guess at where it landed."""
    from atlas.imageio import read_png

    reference = torch.full((16, 16, 3), 0.5)
    sidecar = comparison_sheet(
        tmp_path / "sheet.png",
        [
            SheetEntry("SMALL ERROR", reference, reference + 0.01),
            SheetEntry("LARGE ERROR", reference, reference + 0.50),
        ],
        error_scale=0.5,
    )
    assert sidecar["error_scale"] == 0.5
    image = read_png(tmp_path / "sheet.png").to(torch.float32) / 255.0

    def tile(row, column):
        (place,) = [
            t for t in sidecar["tiles"] if t["row"] == row and t["column"] == column
        ]
        return image[
            place["y"] : place["y"] + place["height"],
            place["x"] : place["x"] + place["width"],
        ]

    # The reference column is identical in both rows; only the error differs.
    assert torch.allclose(tile(0, "reference"), tile(1, "reference"))
    assert float(tile(1, "error").mean()) > 5 * float(tile(0, "error").mean())
    # 0.01 against a scale of 0.5 is 2% up a ramp that starts at black.
    assert float(tile(0, "error").mean()) < 0.1
    # 0.50 against a scale of 0.5 is the top of the ramp, which is white.
    assert float(tile(1, "error").mean()) > 0.85


def test_a_sheet_tile_holds_the_image_it_says_it_holds(tmp_path):
    """The sidecar is the contract; if the rectangles are wrong, everything
    downstream that crops a tile out of a sheet crops the wrong thing."""
    from atlas.imageio import read_png

    reference = torch.full((12, 12, 3), 0.0)
    prediction = torch.full((12, 12, 3), 1.0)
    sidecar = comparison_sheet(
        tmp_path / "s.png", [SheetEntry("BLACK VS WHITE", reference, prediction)]
    )
    image = read_png(tmp_path / "s.png").to(torch.float32) / 255.0
    spots = {t["column"]: t for t in sidecar["tiles"]}
    for column, expected in (("reference", 0.0), ("prediction", 1.0)):
        place = spots[column]
        patch = image[
            place["y"] : place["y"] + place["height"],
            place["x"] : place["x"] + place["width"],
        ]
        assert float(patch.mean()) == pytest.approx(expected, abs=0.01), column


def test_a_sheet_marks_a_row_whose_prediction_diverged(tmp_path):
    reference = torch.full((16, 16, 3), 0.5)
    predicted = reference.clone()
    predicted[4, 4] = float("nan")
    comparison_sheet(
        tmp_path / "sheet.png",
        [SheetEntry("DIVERGED", reference, predicted)],
    )
    assert (tmp_path / "sheet.png").is_file()


def test_an_empty_sheet_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one entry"):
        comparison_sheet(tmp_path / "sheet.png", [])


def test_the_error_ramp_is_monotonic_in_lightness():
    ramp = colourise(torch.linspace(0.0, 1.0, 64).unsqueeze(0))
    lightness = ramp.mean(dim=-1)[0]
    assert bool((lightness[1:] >= lightness[:-1] - 1e-6).all())
    assert float(lightness[0]) == pytest.approx(0.0, abs=1e-6)


# --- defaults ---------------------------------------------------------------


def test_the_default_domains_are_one_hdr_and_one_display():
    assert DEFAULT_DOMAINS == ("mu", "srgb")
    keys = set(evaluate_image(_image(16, 16, 3), _image(16, 16, 3, seed=1)))
    assert {"psnr/mu", "ssim/mu", "psnr/srgb", "ssim/srgb"} <= keys


def test_no_metric_is_reported_without_its_domain():
    """There is no unnamed PSNR in this module, because a PSNR without its
    domain is not comparable with anything."""
    keys = set(evaluate_image(_image(16, 16, 3), _image(16, 16, 3, seed=1)))
    assert "psnr" not in keys and "ssim" not in keys


# --- the floor under the gate ----------------------------------------------


def test_the_constant_baseline_is_what_a_model_that_learned_nothing_scores():
    """The best constant image: every pixel the reference's own mean, in the
    metric's domain. For a uniform-random reference the variance is known, so
    the PSNR it implies can be written down."""
    reference = torch.rand(64, 64, 3, generator=_rng(0))
    baseline = constant_baseline_psnr([reference], domain="linear")
    variance = float(((reference - reference.mean()) ** 2).mean())
    assert baseline == pytest.approx(10 * math.log10(1.0 / variance), abs=1e-6)


def test_a_constant_reference_leaves_nothing_for_a_model_to_beat():
    """Nothing to predict, so a constant predicts it. Measured at 144 dB rather
    than infinite: the mean goes through a Python float on its way back into a
    float32 tensor and leaves a few ulps behind. Either way nothing clears it.
    """
    assert constant_baseline_psnr([torch.full((16, 16, 3), 0.4)]) > 100.0


def test_the_baseline_averages_over_images():
    a, b = _image(16, 16, 3, seed=1), _image(16, 16, 3, seed=2) * 4.0
    together = constant_baseline_psnr([a, b])
    apart = (constant_baseline_psnr([a]) + constant_baseline_psnr([b])) / 2
    assert together == pytest.approx(apart)


def test_the_baseline_honours_a_mask():
    reference = _image(16, 16, 3, seed=3)
    mask = torch.zeros(16, 16)
    mask[:8] = 1.0
    assert constant_baseline_psnr([reference], masks=[mask]) != constant_baseline_psnr(
        [reference]
    )


def test_no_references_is_refused_rather_than_averaged():
    with pytest.raises(ValueError, match="no baseline"):
        constant_baseline_psnr([])


def test_a_model_below_the_baseline_fails_the_gate_however_small_its_gap():
    """The hole: the gate is a difference, and a difference is satisfied by a
    model that is equally hopeless on both splits."""
    report = RelightingReport(
        held_out_view=SplitMetrics("v", 4, {"psnr/mu": 8.0, "non_finite": 0.0}),
        held_out_light=SplitMetrics("l", 4, {"psnr/mu": 7.95, "non_finite": 0.0}),
        baseline_psnr=11.0,
    )
    verdict = report.gate()
    assert abs(verdict.gap_db) < 0.1  # the gap alone looks excellent
    assert not verdict.passed
    assert "not reconstructing" in " ".join(verdict.reasons)


def test_a_model_above_the_baseline_is_judged_on_its_gap():
    above = RelightingReport(
        held_out_view=SplitMetrics("v", 4, {"psnr/mu": 31.0, "non_finite": 0.0}),
        held_out_light=SplitMetrics("l", 4, {"psnr/mu": 30.6, "non_finite": 0.0}),
        baseline_psnr=11.0,
    )
    assert above.gate().passed
    memorising = RelightingReport(
        held_out_view=SplitMetrics("v", 4, {"psnr/mu": 31.0, "non_finite": 0.0}),
        held_out_light=SplitMetrics("l", 4, {"psnr/mu": 25.0, "non_finite": 0.0}),
        baseline_psnr=11.0,
    )
    assert not memorising.gate().passed


def test_a_report_without_a_baseline_keeps_the_old_behaviour():
    """The floor is opt-in, so an existing caller that has no baseline to give
    is not silently failed."""
    report = RelightingReport(
        held_out_view=SplitMetrics("v", 4, {"psnr/mu": 8.0, "non_finite": 0.0}),
        held_out_light=SplitMetrics("l", 4, {"psnr/mu": 7.95, "non_finite": 0.0}),
    )
    assert report.gate().passed
    assert "gate/baseline_psnr" not in report.as_row()


def test_the_baseline_reaches_the_ledger_row():
    report = RelightingReport(
        held_out_view=SplitMetrics("v", 4, {"psnr/mu": 31.0, "non_finite": 0.0}),
        held_out_light=SplitMetrics("l", 4, {"psnr/mu": 30.6, "non_finite": 0.0}),
        baseline_psnr=11.25,
    )
    assert report.as_row()["gate/baseline_psnr"] == 11.25
