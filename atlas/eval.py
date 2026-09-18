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

"""Evaluation: the metrics, the two held-out sets, and the gate between them.

Built before the trainer, because a trainer written against an evaluation
harness is a different program from one that has metrics bolted on afterwards.
Four decisions here are load-bearing.

**Metrics are computed in a named tonemapped domain, never on raw HDR.** Mean
squared error on linear radiance is dominated by whatever is brightest. A
specular highlight at 40.0 and the diffuse surface at 0.3 differ by more than
two orders of magnitude, so on a relighting capture -- which is *made of*
moving highlights -- a raw-HDR PSNR is very nearly a measurement of the
highlight alone, and a model can lose the whole diffuse response without the
number moving. Every metric therefore carries its domain in its name:
``psnr/mu``, ``ssim/srgb``. There is no unnamed PSNR in this module.

**Held-out views and held-out lights are reported side by side**, because the
quantity that decides the project is their *difference*. A model that memorises
the illuminations it was shown scores well on a new camera and badly on a new
light; a model that learned transport scores alike on both. One number in
isolation cannot tell those apart, so :class:`RelightingReport` refuses to
report one without the other.

**The gate fails on non-finite output rather than scoring it.** Diverged pixels
are counted and reported as a fraction, and metrics are computed with them set
to zero so that a run still produces a comparison sheet to look at. Zero
against a black background would otherwise *flatter* a diverged model, so any
non-zero fraction fails the gate outright, whatever the metrics say.

**Exposure alignment is off by default and named when it is on.** Fitting a
scalar per image before scoring is standard practice in relighting papers and
it is also an excellent way to hide a systematic brightness error. When it is
enabled the fitted scale is recorded alongside the metrics, so a reader can see
how much was absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .imageio import GLYPH_WIDTH, draw_text, text_size, write_png
from .run import append_ledger

__all__ = [
    "ToneMap",
    "TONEMAPS",
    "DEFAULT_DOMAINS",
    "tonemap",
    "psnr",
    "ssim",
    "ssim_map",
    "lpips",
    "lpips_available",
    "align_exposure",
    "evaluate_image",
    "evaluate_set",
    "SplitMetrics",
    "RelightingReport",
    "GateResult",
    "GATE_PSNR_GAP_DB",
    "append_evaluation",
]


# --- tonemaps ---------------------------------------------------------------


@dataclass(frozen=True)
class ToneMap:
    """A named transfer function from linear radiance to a metric domain.

    ``data_range`` is the value the domain calls white. It sets the ``L`` in
    SSIM's stabilising constants and the numerator of PSNR, so it belongs to
    the domain and not to the caller's taste.
    """

    name: str
    forward: Callable[[Tensor], Tensor]
    data_range: float
    description: str = ""

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def _mu_law(x: Tensor, mu: float = 5000.0) -> Tensor:
    return torch.log1p(mu * x.clamp_min(0.0)) / math.log1p(mu)


def _srgb(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    encoded = torch.where(
        x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055
    )
    # Clamped so that `data_range = 1.0` is a true bound rather than a bound up
    # to rounding. The curve reaches 1.0 only to within float32 epsilon anyway
    # -- 1.055 - 0.055 is 0.99999994 -- which is far below anything a metric
    # can see, but a domain that states its range should not exceed it.
    return encoded.clamp(0.0, 1.0)


def _gamma22(x: Tensor) -> Tensor:
    return x.clamp(0.0, 1.0) ** (1 / 2.2)


def _reinhard(x: Tensor) -> Tensor:
    x = x.clamp_min(0.0)
    return x / (1.0 + x)


TONEMAPS: Dict[str, ToneMap] = {
    "linear": ToneMap(
        "linear",
        lambda x: x,
        1.0,
        "raw radiance; dominated by the brightest few percent, so read it "
        "alongside another domain and never alone",
    ),
    "mu": ToneMap(
        "mu",
        _mu_law,
        1.0,
        "mu-law, log(1 + 5000 x) / log(5001); the HDR convention. Unbounded "
        "above, so highlights still count -- they just stop being everything",
    ),
    "srgb": ToneMap(
        "srgb",
        _srgb,
        1.0,
        "sRGB transfer with clipping at 1.0; comparable with published LDR "
        "numbers, and blind to anything above white",
    ),
    "gamma22": ToneMap("gamma22", _gamma22, 1.0, "plain 2.2 gamma with clipping"),
    "reinhard": ToneMap(
        "reinhard", _reinhard, 1.0, "x / (1 + x); bounded, no clipping"
    ),
}

#: What a run reports unless told otherwise: one unbounded HDR domain and one
#: display domain. The pair is deliberate. Agreement between them means the
#: error is spread across the range; disagreement localises it.
DEFAULT_DOMAINS = ("mu", "srgb")


def tonemap(x: Tensor, domain: str = "mu") -> Tensor:
    """Apply a named tonemap. Raises :class:`KeyError` naming the known ones."""
    try:
        curve = TONEMAPS[domain]
    except KeyError:
        raise KeyError(
            f"unknown tonemap {domain!r}; known: {sorted(TONEMAPS)}"
        ) from None
    return curve(x)


# --- shape and mask plumbing ------------------------------------------------


def _check_pair(pred: Tensor, ref: Tensor) -> None:
    if pred.shape != ref.shape:
        raise ValueError(
            f"prediction and reference must match, got {tuple(pred.shape)} "
            f"and {tuple(ref.shape)}"
        )
    if pred.ndim not in (2, 3):
        raise ValueError(f"images must be [H, W] or [H, W, C], got {tuple(pred.shape)}")


def _as_hwc(image: Tensor) -> Tensor:
    return image.unsqueeze(-1) if image.ndim == 2 else image


def _prepare_mask(mask: Optional[Tensor], shape: Sequence[int]) -> Optional[Tensor]:
    """Normalise a mask to ``[H, W, 1]`` float, or ``None``."""
    if mask is None:
        return None
    height, width = shape[0], shape[1]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(
            f"mask must be [{height}, {width}] or [{height}, {width}, 1], "
            f"got {tuple(mask.shape)}"
        )
    mask = mask.to(torch.float32)
    if float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
        raise ValueError("mask values must lie in [0, 1]")
    return mask.unsqueeze(-1)


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> float:
    if mask is None:
        return float(values.mean())
    weight = mask.expand_as(values)
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("mask selects no pixels")
    return float((values * weight).sum() / total)


# --- exposure ---------------------------------------------------------------


def align_exposure(
    pred: Tensor,
    ref: Tensor,
    *,
    mask: Optional[Tensor] = None,
    per_channel: bool = False,
):
    """Scale ``pred`` by the least-squares factor that best matches ``ref``.

    Absolute radiance is only recoverable when the capture was photometrically
    calibrated, so comparing an uncalibrated reconstruction against a reference
    without this step measures the calibration and not the model. Comparing
    *with* it hides a systematic brightness error completely. Both are
    defensible and they are not the same measurement, which is why the fitted
    scale comes back with the result and ends up in the ledger.

    ``per_channel`` fits three scales, which additionally absorbs a white
    balance error. That is a bigger thing to hide, so it is not the default.

    Args:
        pred: ``[H, W]`` or ``[H, W, C]`` linear radiance.
        ref: Same shape.
        mask: Optional ``[H, W]`` weights; the fit uses only what it selects.
        per_channel: Fit one scale per channel instead of one overall.

    Returns:
        ``(scaled_prediction, scale)``, where ``scale`` is a ``[C]`` tensor.
    """
    _check_pair(pred, ref)
    pred_hwc, ref_hwc = _as_hwc(pred), _as_hwc(ref)
    weight = _prepare_mask(mask, pred_hwc.shape)
    if weight is None:
        weight = torch.ones_like(pred_hwc[..., :1])

    dims = (0, 1)
    numerator = (weight * pred_hwc * ref_hwc).sum(dim=dims)
    denominator = (weight * pred_hwc * pred_hwc).sum(dim=dims)
    if not per_channel:
        numerator = numerator.sum().expand(numerator.shape)
        denominator = denominator.sum().expand(denominator.shape)
    # A prediction that is identically zero has no scale that matches anything;
    # leaving it alone is the honest outcome and scores it as the failure it is.
    scale = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-30),
        torch.ones_like(denominator),
    )
    scaled = pred_hwc * scale
    return (scaled.squeeze(-1) if pred.ndim == 2 else scaled), scale


# --- PSNR -------------------------------------------------------------------


def psnr(
    pred: Tensor,
    ref: Tensor,
    *,
    mask: Optional[Tensor] = None,
    data_range: float = 1.0,
) -> float:
    """Peak signal-to-noise ratio, in decibels, over the masked pixels.

    Args:
        pred: ``[H, W]`` or ``[H, W, C]``, already in the metric domain.
        ref: Same shape.
        mask: Optional ``[H, W]`` weights in ``[0, 1]``.
        data_range: What the domain calls white.

    Returns:
        Decibels. Identical images give ``inf``, which is the true value and is
        loud enough that nobody mistakes it for a good score.
    """
    _check_pair(pred, ref)
    weight = _prepare_mask(mask, _as_hwc(pred).shape)
    error = (_as_hwc(pred) - _as_hwc(ref)) ** 2
    mse = _masked_mean(error, weight)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range * data_range / mse)


# --- SSIM -------------------------------------------------------------------


def _gaussian(size: int, sigma: float, dtype, device) -> Tensor:
    coordinates = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    weights = torch.exp(-(coordinates**2) / (2.0 * sigma * sigma))
    return weights / weights.sum()


def _separable_valid(image: Tensor, kernel: Tensor) -> Tensor:
    """Gaussian-weighted local mean, 'valid' support, channels independent.

    The channels ride in the *batch* dimension with a single-channel kernel,
    rather than in the channel dimension with grouped convolution. Both are
    correct; this one cannot be got subtly wrong, because there is only one
    filter and nothing to line up.
    """
    planes = image.permute(2, 0, 1).unsqueeze(1)  # [C, 1, H, W]
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    out = torch.nn.functional.conv2d(planes, horizontal)
    out = torch.nn.functional.conv2d(out, vertical)
    return out.squeeze(1).permute(1, 2, 0)


def ssim_map(
    pred: Tensor,
    ref: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """The per-pixel SSIM map over the 'valid' region.

    This is Wang et al. (2004) with the Gaussian window they specify: 11x11,
    sigma 1.5, K1 = 0.01, K2 = 0.03, and population rather than sample
    statistics -- the same configuration as
    ``skimage.metrics.structural_similarity(..., gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False)``.

    Support is 'valid': the map is ``[H - w + 1, W - w + 1, C]``. Padding would
    invent statistics at the border out of reflected or zeroed pixels, and the
    border of a masked object render is exactly where a relighting model is
    most likely to be wrong, so an invented number there is worse than none.

    Returns:
        ``[H - w + 1, W - w + 1, C]``, unaggregated, so that a caller can mask
        it, visualise it, or reduce it however it needs.
    """
    _check_pair(pred, ref)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and at least 3, got {window_size}")
    x, y = _as_hwc(pred).to(torch.float64), _as_hwc(ref).to(torch.float64)
    height, width = x.shape[0], x.shape[1]
    if height < window_size or width < window_size:
        raise ValueError(
            f"image {height}x{width} is smaller than the {window_size}x"
            f"{window_size} SSIM window; crop the window or use a larger image"
        )

    kernel = _gaussian(window_size, sigma, x.dtype, x.device)
    mu_x = _separable_valid(x, kernel)
    mu_y = _separable_valid(y, kernel)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = _separable_valid(x * x, kernel) - mu_xx
    sigma_yy = _separable_valid(y * y, kernel) - mu_yy
    sigma_xy = _separable_valid(x * y, kernel) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return (numerator / denominator).to(_as_hwc(pred).dtype)


def ssim(
    pred: Tensor,
    ref: Tensor,
    *,
    mask: Optional[Tensor] = None,
    data_range: float = 1.0,
    window_size: int = 11,
    **kwargs: Any,
) -> float:
    """Mean SSIM over the valid region, weighted by ``mask`` if given.

    The mask is cropped to the valid region rather than eroded, so a window
    centred just inside the object still sees background pixels. That is a
    property of windowed SSIM and not something a mask can fix; it is named
    here so that a masked SSIM is not read as "SSIM of the object alone".
    Compositing both images onto the same background before calling is the way
    to make the boundary term identical for prediction and reference, and is
    what :func:`evaluate_image` does when given a ``background``.
    """
    per_pixel = ssim_map(
        pred, ref, data_range=data_range, window_size=window_size, **kwargs
    )
    weight = _prepare_mask(mask, _as_hwc(pred).shape)
    if weight is not None:
        margin = (window_size - 1) // 2
        weight = weight[
            margin : margin + per_pixel.shape[0], margin : margin + per_pixel.shape[1]
        ]
    return _masked_mean(per_pixel, weight)


# --- LPIPS, behind an optional dependency -----------------------------------

_LPIPS_CACHE: Dict[str, Any] = {}


def lpips_available() -> bool:
    """Whether the optional ``lpips`` package can be imported."""
    try:
        import lpips as _  # noqa: F401
    except ImportError:
        return False
    return True


def lpips(pred: Tensor, ref: Tensor, *, net: str = "alex") -> float:
    """Learned perceptual distance. Lower is better.

    Needs the optional ``lpips`` package and its pretrained weights. Inputs are
    expected to be in a display domain already -- the network was trained on
    ordinary images, so feeding it unbounded HDR is out of distribution -- and
    are clamped to ``[0, 1]`` before use.

    Raises:
        RuntimeError: If ``lpips`` is not installed, naming the extra to add.
    """
    try:
        import lpips as lpips_package
    except ImportError as error:
        raise RuntimeError(
            "LPIPS needs the optional 'lpips' package: pip install 'atlas-relight[eval]'"
        ) from error

    _check_pair(pred, ref)
    if net not in _LPIPS_CACHE:
        _LPIPS_CACHE[net] = lpips_package.LPIPS(net=net, verbose=False).eval()
    model = _LPIPS_CACHE[net]

    def prepare(image: Tensor) -> Tensor:
        image = _as_hwc(image).clamp(0.0, 1.0).to(torch.float32)
        if image.shape[-1] == 1:
            image = image.expand(-1, -1, 3)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        return image.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    with torch.no_grad():
        return float(model(prepare(pred), prepare(ref)).reshape(()))


# --- one image --------------------------------------------------------------

#: Metrics where a smaller number is better. Everything else is treated as
#: higher-is-better when a gap is computed, so that a positive gap always means
#: the same thing: the held-out *light* split did worse.
LOWER_IS_BETTER = ("lpips",)

#: Recorded alongside the metrics but not scored or gapped.
INFORMATIONAL = ("pixels", "images", "non_finite", "exposure_scale")


def _sanitise(pred: Tensor) -> "tuple[Tensor, float]":
    """Replace non-finite predictions with zero and report how many there were.

    Crashing here would throw away the comparison sheet that shows *where* a
    model diverged, which is the one artifact worth having when it does. Zero
    is the replacement because it is what an absent primitive contributes, and
    because it scores badly against anything lit. Against an unlit background
    it scores perfectly, which is why the fraction is reported and why any
    non-zero value fails the gate outright.
    """
    finite = torch.isfinite(pred)
    if bool(finite.all()):
        return pred, 0.0
    fraction = float((~finite).sum()) / max(pred.numel(), 1)
    return torch.where(finite, pred, torch.zeros_like(pred)), fraction


def evaluate_image(
    pred: Tensor,
    ref: Tensor,
    *,
    mask: Optional[Tensor] = None,
    background: Optional[float] = None,
    domains: Sequence[str] = DEFAULT_DOMAINS,
    lpips_net: Optional[str] = None,
    lpips_domain: str = "srgb",
    exposure_align: bool = False,
    per_channel_exposure: bool = False,
) -> Dict[str, float]:
    """Every metric for one prediction, keyed by ``metric/domain``.

    Args:
        pred: ``[H, W]`` or ``[H, W, C]`` **linear** radiance.
        ref: Same shape, linear radiance, and required to be finite -- a
            non-finite reference is a data fault, not a model fault.
        mask: Optional ``[H, W]`` weights in ``[0, 1]``.
        background: If given with a mask, both images are composited onto this
            constant first, so SSIM windows that straddle the silhouette see
            the same thing in each. Without it the boundary term differs and
            masked SSIM quietly measures the background too.
        domains: Tonemaps to report in. Each contributes ``psnr/<domain>`` and
            ``ssim/<domain>``.
        lpips_net: ``"alex"`` or ``"vgg"`` to include LPIPS; ``None`` to skip.
        lpips_domain: Which domain LPIPS is computed in. It is one domain, not
            all of them, because the network expects display-referred input.
        exposure_align: Fit a least-squares scale on linear radiance first.
        per_channel_exposure: Fit three scales rather than one.

    Returns:
        A flat ``{metric: value}`` dict, including ``non_finite`` and
        ``pixels``, and ``exposure_scale`` when alignment was requested.
    """
    _check_pair(pred, ref)
    if not bool(torch.isfinite(ref).all()):
        raise ValueError(
            "reference image contains non-finite values; that is a data fault "
            "and would silently corrupt every metric computed against it"
        )

    pred, non_finite = _sanitise(pred)
    weight = _prepare_mask(mask, _as_hwc(pred).shape)

    results: Dict[str, float] = {}
    if exposure_align:
        pred, scale = align_exposure(
            pred, ref, mask=mask, per_channel=per_channel_exposure
        )
        results["exposure_scale"] = float(scale.mean())

    if background is not None and weight is not None:
        pred_hwc = _as_hwc(pred) * weight + background * (1.0 - weight)
        ref_hwc = _as_hwc(ref) * weight + background * (1.0 - weight)
        pred = pred_hwc.squeeze(-1) if pred.ndim == 2 else pred_hwc
        ref = ref_hwc.squeeze(-1) if ref.ndim == 2 else ref_hwc

    for domain in domains:
        curve = TONEMAPS[domain] if domain in TONEMAPS else None
        if curve is None:
            raise KeyError(f"unknown tonemap {domain!r}; known: {sorted(TONEMAPS)}")
        mapped_pred, mapped_ref = curve(pred), curve(ref)
        results[f"psnr/{domain}"] = psnr(
            mapped_pred, mapped_ref, mask=mask, data_range=curve.data_range
        )
        results[f"ssim/{domain}"] = ssim(
            mapped_pred, mapped_ref, mask=mask, data_range=curve.data_range
        )

    if lpips_net is not None:
        curve = TONEMAPS[lpips_domain]
        results[f"lpips/{lpips_domain}"] = lpips(curve(pred), curve(ref), net=lpips_net)

    results["non_finite"] = non_finite
    results["pixels"] = float(
        weight.sum() * _as_hwc(pred).shape[-1]
        if weight is not None
        else _as_hwc(pred).numel()
    )
    return results


# --- a set of images --------------------------------------------------------


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics averaged over one held-out set."""

    name: str
    count: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.metrics.get(key, default)

    def as_row(self, prefix: Optional[str] = None) -> Dict[str, float]:
        """Flatten for the ledger, under ``<name>/<metric>`` keys."""
        stem = self.name if prefix is None else prefix
        row: Dict[str, Any] = {f"{stem}/images": self.count}
        row.update({f"{stem}/{k}": v for k, v in self.metrics.items()})
        return row


def evaluate_set(
    pairs: Iterable[Sequence[Tensor]], *, name: str = "set", **kwargs: Any
) -> SplitMetrics:
    """Average :func:`evaluate_image` over a set of ``(pred, ref[, mask])``.

    The average is over images, not over pooled pixels: each image counts once
    however large it is, which is what a per-image PSNR table means everywhere
    else and what makes a mean comparable across captures shot at different
    resolutions. ``pixels`` is summed instead, so the two together say how much
    was actually measured.

    An identical pair scores ``inf`` and the mean is then ``inf``. That is
    intentional -- it is visible, whereas a capped value looks like a result.
    """
    totals: Dict[str, float] = {}
    count = 0
    for entry in pairs:
        if len(entry) == 2:
            pred, ref, mask = entry[0], entry[1], None
        elif len(entry) == 3:
            pred, ref, mask = entry[0], entry[1], entry[2]
        else:
            raise ValueError(
                f"each entry must be (pred, ref) or (pred, ref, mask), got "
                f"{len(entry)} items"
            )
        for key, value in evaluate_image(pred, ref, mask=mask, **kwargs).items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

    if count == 0:
        raise ValueError(f"the {name!r} split is empty; there is nothing to report")
    averaged = {
        key: (total if key == "pixels" else total / count)
        for key, total in totals.items()
    }
    return SplitMetrics(name=name, count=count, metrics=averaged)


# --- the two splits, and the gate between them ------------------------------

#: The decision the project rests on. If held-out-light PSNR trails
#: held-out-view PSNR by more than this, the model is reproducing illuminations
#: it was shown rather than transport, and no amount of speed work redeems it.
GATE_PSNR_GAP_DB = 1.0


@dataclass(frozen=True)
class GateResult:
    """Whether a report clears the gate, and if not, exactly why."""

    passed: bool
    gap_db: float
    threshold: float
    domain: str
    reasons: Sequence[str] = ()

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        head = (
            f"{verdict}  psnr/{self.domain} gap = {self.gap_db:+.3f} dB "
            f"(threshold {self.threshold:.3f} dB)"
        )
        return "\n".join([head, *(f"  - {r}" for r in self.reasons)])


def _is_lower_better(key: str) -> bool:
    return key.split("/", 1)[0] in LOWER_IS_BETTER


@dataclass(frozen=True)
class RelightingReport:
    """Held-out views and held-out lights, reported together.

    Together because apart they are misleading. Novel-view quality on its own
    says nothing about whether the model learned transport, and novel-light
    quality on its own cannot distinguish a relighting failure from a
    reconstruction that was never any good. The pair, and specifically their
    difference, is the measurement.
    """

    held_out_view: SplitMetrics
    held_out_light: SplitMetrics
    train: Optional[SplitMetrics] = None

    @property
    def gaps(self) -> Dict[str, float]:
        """``view`` minus ``light``, signed so that positive is always worse.

        For LPIPS, where smaller is better, the subtraction is reversed. The
        invariant a reader can rely on is one-directional: a positive gap means
        the held-out *light* split did worse, whatever the metric.
        """
        out: Dict[str, float] = {}
        for key, view_value in self.held_out_view.metrics.items():
            if key in INFORMATIONAL or key not in self.held_out_light.metrics:
                continue
            light_value = self.held_out_light.metrics[key]
            out[key] = (
                light_value - view_value
                if _is_lower_better(key)
                else view_value - light_value
            )
        return out

    def gate(
        self,
        *,
        domain: str = "mu",
        threshold: float = GATE_PSNR_GAP_DB,
    ) -> GateResult:
        """Apply the held-out-light gate.

        Three ways to fail, and only one of them is about the metric:

        1. The gap exceeds ``threshold``.
        2. Either split produced non-finite pixels. Those are scored as zero,
           which against an unlit background is a *perfect* score, so a gate
           that ignored them could be passed by diverging.
        3. Either split is empty, which would otherwise pass by having nothing
           to disagree about.
        """
        key = f"psnr/{domain}"
        reasons: List[str] = []
        for split in (self.held_out_view, self.held_out_light):
            if split.count == 0:
                reasons.append(f"the {split.name!r} split is empty")
            if key not in split.metrics:
                reasons.append(f"{split.name!r} has no {key}; domains do not match")
            diverged = split.metrics.get("non_finite", 0.0)
            if diverged > 0.0:
                reasons.append(
                    f"{split.name!r} produced non-finite pixels "
                    f"({diverged:.3%} of the tensor); the model diverged"
                )

        gap = self.gaps.get(key, float("nan"))
        if not reasons:
            if math.isnan(gap):
                reasons.append(f"{key} is not available in both splits")
            elif gap > threshold:
                reasons.append(
                    f"held-out-light psnr trails held-out-view by {gap:.3f} dB, "
                    f"over the {threshold:.3f} dB budget: the model is "
                    f"reproducing illuminations it was shown, not transport"
                )
        return GateResult(
            passed=not reasons,
            gap_db=gap,
            threshold=threshold,
            domain=domain,
            reasons=tuple(reasons),
        )

    def as_row(self, **extra: Any) -> Dict[str, Any]:
        """Flatten to one ledger row: both splits, the gaps, and the gate."""
        row: Dict[str, Any] = {}
        row.update(self.held_out_view.as_row("heldout_view"))
        row.update(self.held_out_light.as_row("heldout_light"))
        if self.train is not None:
            row.update(self.train.as_row("train"))
        row.update({f"gap/{k}": v for k, v in self.gaps.items()})
        verdict = self.gate()
        row["gate/passed"] = verdict.passed
        row["gate/psnr_gap_db"] = verdict.gap_db
        row["gate/threshold_db"] = verdict.threshold
        row.update(extra)
        return row

    def format_table(self, *, precision: int = 4) -> str:
        """The side-by-side table, for a log or a terminal."""
        columns = ["held-out view", "held-out light", "gap"]
        if self.train is not None:
            columns.insert(0, "train")
        keys = [k for k in self.held_out_view.metrics if k not in INFORMATIONAL]
        keys += [
            k
            for k in self.held_out_light.metrics
            if k not in INFORMATIONAL and k not in keys
        ]
        gaps = self.gaps

        def cell(value: Optional[float]) -> str:
            if value is None:
                return "-"
            if math.isinf(value):
                return "inf"
            return f"{value:.{precision}f}"

        rows = []
        for key in sorted(keys):
            values = [
                self.held_out_view.get(key),
                self.held_out_light.get(key),
                gaps.get(key),
            ]
            if self.train is not None:
                values.insert(0, self.train.get(key))
            rows.append([key, *(cell(v) for v in values)])

        header = ["metric", *columns]
        widths = [
            max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
            for i in range(len(header))
        ]

        def line(cells: Sequence[str]) -> str:
            body = [cells[0].ljust(widths[0])]
            body += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
            return "  ".join(body)

        counts = (
            f"{self.held_out_view.count} views, " f"{self.held_out_light.count} lights"
        )
        out = [line(header), "  ".join("-" * w for w in widths)]
        out += [line(r) for r in rows]
        out += ["  ".join("-" * w for w in widths), f"({counts})"]
        out.append("positive gap = the held-out-light split did worse")
        out.append(str(self.gate()))
        return "\n".join(out)


# --- the ledger -------------------------------------------------------------


def append_evaluation(
    ledger: Path | str,
    report: RelightingReport,
    *,
    run: Optional[str] = None,
    config_hash: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    step: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Append one evaluation to the results ledger and return the row written.

    Provenance is carried across selectively rather than wholesale: the commit,
    the dirty flag and the dataset hash are what make a row comparable with
    another row, and the rest -- argv, hostname, package versions -- already
    lives in the run directory that ``run`` names.
    """
    row: Dict[str, Any] = {}
    if run is not None:
        row["run"] = run
    if config_hash is not None:
        row["config_hash"] = config_hash
    if step is not None:
        row["step"] = int(step)
    if provenance is not None:
        for key in ("git_commit", "git_dirty", "dataset_hash", "cuda_device"):
            if key in provenance:
                row[key] = provenance[key]
    row.update(report.as_row(**extra))
    append_ledger(ledger, row)
    return row


# --- the comparison sheet ---------------------------------------------------

#: Control points of the error ramp: black, blue, magenta, orange, white. It is
#: not perceptually uniform and does not pretend to be. What it has to do is
#: make "where is the error" answerable across four orders of magnitude at a
#: glance, and a ramp that changes hue as well as lightness does that whether
#: the sheet is viewed on a calibrated display or a phone.
_ERROR_RAMP = (
    (0.00, (0.00, 0.00, 0.00)),
    (0.25, (0.15, 0.10, 0.55)),
    (0.50, (0.72, 0.13, 0.52)),
    (0.75, (0.98, 0.55, 0.05)),
    (1.00, (1.00, 1.00, 0.90)),
)


def colourise(values: Tensor) -> Tensor:
    """Map ``[H, W]`` in ``[0, 1]`` through the error ramp to ``[H, W, 3]``."""
    values = values.clamp(0.0, 1.0).unsqueeze(-1)
    out = torch.zeros(*values.shape[:-1], 3, dtype=torch.float32)
    for (lo, low_colour), (hi, high_colour) in zip(_ERROR_RAMP, _ERROR_RAMP[1:]):
        low = torch.tensor(low_colour)
        high = torch.tensor(high_colour)
        span = max(hi - lo, 1e-9)
        t = ((values - lo) / span).clamp(0.0, 1.0)
        inside = ((values >= lo) & (values <= hi)).to(torch.float32)
        out = out * (1.0 - inside) + inside * (low + (high - low) * t)
    return out


def _fit_text(text: str, available: int) -> "tuple[str, int]":
    """The largest scale at which ``text`` fits, and the text truncated to fit.

    A caption that overruns its column and collides with the next one is worse
    than a short one, and a sheet is read at a glance or not at all.
    """
    for scale in (2, 1):
        if text_size(text, scale=scale)[0] <= available:
            return text, scale
    advance = GLYPH_WIDTH + 1
    return text[: max(available // advance, 1)], 1


@dataclass(frozen=True)
class SheetEntry:
    """One row of a comparison sheet."""

    label: str
    reference: Tensor
    prediction: Tensor
    mask: Optional[Tensor] = None


def comparison_sheet(
    path: Path | str,
    entries: Sequence[SheetEntry],
    *,
    domain: str = "srgb",
    error_scale: Optional[float] = None,
    error_percentile: float = 0.99,
    gutter: int = 6,
    label_height: int = 18,
    background: float = 0.08,
) -> Dict[str, Any]:
    """Write a reference / prediction / error sheet, and its sidecar JSON.

    One row per entry, three columns, labelled. The error column is
    ``|prediction - reference|`` averaged over channels, in **linear** radiance
    and divided by a single scale shared by every row -- so the rows are
    comparable with each other, which is the only reason to put them on one
    sheet. Normalising each row to its own maximum would make every row look
    equally bad and the sheet worthless.

    Args:
        path: Destination PNG. A ``.json`` sidecar is written beside it with
            the labels, the error scale and the domain, because the picture is
            for a person and the sidecar is for everything else.
        entries: Rows, in order.
        domain: Tonemap for the reference and prediction columns.
        error_scale: Linear radiance mapped to the top of the ramp. Defaults to
            the ``error_percentile`` quantile over every entry, which keeps one
            blown-out pixel from flattening the whole sheet.
        error_percentile: Quantile used when ``error_scale`` is ``None``.
        gutter: Pixels between tiles.
        label_height: Pixels reserved above each row for its caption.
        background: Grey level of the sheet behind the tiles.

    Returns:
        The sidecar dictionary, also written to disk.
    """
    if not entries:
        raise ValueError("a comparison sheet needs at least one entry")
    curve = TONEMAPS[domain] if domain in TONEMAPS else None
    if curve is None:
        raise KeyError(f"unknown tonemap {domain!r}; known: {sorted(TONEMAPS)}")

    errors: List[Tensor] = []
    for entry in entries:
        _check_pair(entry.prediction, entry.reference)
        prediction, _ = _sanitise(entry.prediction)
        magnitude = (_as_hwc(prediction) - _as_hwc(entry.reference)).abs().mean(dim=-1)
        if entry.mask is not None:
            magnitude = (
                magnitude
                * _prepare_mask(entry.mask, _as_hwc(entry.reference).shape)[..., 0]
            )
        errors.append(magnitude)

    if error_scale is None:
        pooled = torch.cat([e.reshape(-1) for e in errors]).to(torch.float32)
        # torch.quantile refuses more than about 16 million elements, which a
        # single 4K frame already exceeds. A strided subsample of a quantile
        # estimate is still a quantile estimate, and this one only has to pick
        # a display scale.
        limit = 1 << 22
        if pooled.numel() > limit:
            pooled = pooled[:: pooled.numel() // limit + 1]
        error_scale = max(float(torch.quantile(pooled, error_percentile)), 1e-8)

    heights = [int(_as_hwc(e.reference).shape[0]) for e in entries]
    widths = [int(_as_hwc(e.reference).shape[1]) for e in entries]
    tile_width, tile_height = max(widths), max(heights)
    # The error column says what the top of the ramp is worth in linear
    # radiance, because "bright red" is only meaningful against a number.
    columns = ("REFERENCE", "PREDICTION", f"ERROR 0-{error_scale:.3G}")

    sheet_width = 3 * tile_width + 4 * gutter
    row_height = tile_height + label_height + gutter
    sheet_height = row_height * len(entries) + gutter + label_height
    canvas = torch.full((sheet_height, sheet_width, 3), background, dtype=torch.float32)

    for index, title in enumerate(columns):
        text, scale = _fit_text(title, tile_width - 4)
        draw_text(
            canvas,
            text,
            (gutter, gutter + index * (tile_width + gutter) + 2),
            colour=0.85,
            scale=scale,
        )

    placements: List[Dict[str, Any]] = []
    for row, (entry, magnitude) in enumerate(zip(entries, errors)):
        top = label_height + gutter + row * row_height
        text, scale = _fit_text(entry.label, sheet_width - 2 * gutter - 4)
        draw_text(canvas, text, (top, gutter + 2), colour=0.75, scale=scale)
        prediction, non_finite = _sanitise(entry.prediction)
        tiles = (
            curve(_as_hwc(entry.reference)).clamp(0.0, 1.0),
            curve(_as_hwc(prediction)).clamp(0.0, 1.0),
            colourise(magnitude / error_scale),
        )
        for column, tile in enumerate(tiles):
            if tile.shape[-1] == 1:
                tile = tile.expand(-1, -1, 3)
            y0 = top + label_height
            x0 = gutter + column * (tile_width + gutter)
            canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile.to(
                torch.float32
            )
            placements.append(
                {
                    "row": row,
                    "column": columns[column].split()[0].lower(),
                    "y": int(y0),
                    "x": int(x0),
                    "height": int(tile.shape[0]),
                    "width": int(tile.shape[1]),
                }
            )
        if non_finite > 0.0:
            warning = f"NON-FINITE {non_finite:.2%}"
            text, scale = _fit_text(warning, sheet_width // 2)
            width, _ = text_size(text, scale=scale)
            draw_text(
                canvas,
                text,
                (top, sheet_width - gutter - width),
                colour=(1.0, 0.25, 0.25),
                scale=scale,
            )

    path = Path(path)
    write_png(path, canvas)
    sidecar = {
        "columns": list(columns),
        "domain": domain,
        "error_scale": error_scale,
        "error_ramp": "black-blue-magenta-orange-white, linear in |dI| / scale",
        "rows": [
            {
                "label": e.label,
                "height": int(_as_hwc(e.reference).shape[0]),
                "width": int(_as_hwc(e.reference).shape[1]),
                "masked": e.mask is not None,
            }
            for e in entries
        ],
        "tiles": placements,
        "image": path.name,
        "size": [int(canvas.shape[1]), int(canvas.shape[0])],
    }
    import json

    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar
