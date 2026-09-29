# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Photographs in, linear radiance out.

Everything downstream of this module does arithmetic on radiance, so this is
the one place where a camera's tone curve is undone. The rules:

* **RAW is decoded linearly** with LibRaw (``rawpy``): camera white balance,
  sRGB primaries, *no* gamma, *no* auto-brightening, 16 bits. That is as close
  to scene-referred as a consumer camera gets.
* **JPEG is a fallback**, decoded through the inverse sRGB curve. A camera's
  JPEG engine applies its own tone curve on top of sRGB, so this is an
  approximation and :class:`DecodedImage` says so (``linear_exact=False``); the
  capture report surfaces it rather than letting it pass silently.
* **Resampling happens in linear space**, with area (antialiased) filtering.
  Downsampling a gamma-encoded image darkens edges; the order matters.

Exposure is read from EXIF so shots taken at different ISO or aperture land on
one radiometric scale. For a **flash** exposure the shutter time is not part
of it -- the flash pulse is far shorter than the shutter -- so the relevant
factor is ``ISO / N^2``; ambient light, which does integrate over the shutter,
is what ``shutter`` would scale and is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "RAW_SUFFIXES",
    "IMAGE_SUFFIXES",
    "DecodedImage",
    "ExposureInfo",
    "decode_image",
    "srgb_to_linear",
    "linear_to_srgb",
    "resize_linear",
    "list_images",
    "read_exposure",
]

RAW_SUFFIXES = (".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class ExposureInfo:
    """What EXIF says about how bright a shot was made."""

    iso: Optional[float] = None
    f_number: Optional[float] = None
    shutter: Optional[float] = None
    flash_fired: Optional[bool] = None

    def flash_gain(self) -> Optional[float]:
        """``ISO / N^2``: how strongly a flash-lit pixel was recorded."""
        if self.iso is None or self.f_number is None or self.f_number <= 0:
            return None
        return self.iso / (self.f_number * self.f_number)


@dataclass(frozen=True)
class DecodedImage:
    path: Path
    #: ``[H, W, 3]`` float32 linear radiance, arbitrary scale.
    linear: Tensor
    #: True for RAW; False when a tone curve had to be guessed away.
    linear_exact: bool
    exposure: ExposureInfo


def srgb_to_linear(x: Tensor) -> Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def resize_linear(image: Tensor, max_side: Optional[int]) -> Tensor:
    """Area-downsample ``[H, W, C]`` so its longer side is at most ``max_side``."""
    if max_side is None:
        return image
    height, width = image.shape[:2]
    factor = max(height, width) / float(max_side)
    if factor <= 1.0:
        return image
    size = (max(1, round(height / factor)), max(1, round(width / factor)))
    chw = image.permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.interpolate(
        chw, size=size, mode="bilinear", antialias=True, align_corners=False
    )
    return out.squeeze(0).permute(1, 2, 0).contiguous()


def list_images(directory: Path | str) -> list[Path]:
    """Every decodable image in ``directory``, one per stem, RAW preferred.

    Cameras set to RAW+JPEG write both; the RAW is the measurement and the JPEG
    is a rendering of it, so when both exist only the RAW is kept.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such image directory: {directory}")
    by_stem: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in RAW_SUFFIXES + IMAGE_SUFFIXES or not path.is_file():
            continue
        current = by_stem.get(path.stem)
        if current is None or (
            suffix in RAW_SUFFIXES and current.suffix.lower() not in RAW_SUFFIXES
        ):
            by_stem[path.stem] = path
    if not by_stem:
        raise ValueError(f"no images found in {directory}")
    return [by_stem[stem] for stem in sorted(by_stem)]


def _ratio(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            numerator, denominator = value
            return float(numerator) / float(denominator)
        except Exception:
            return None


def read_exposure(path: Path | str) -> ExposureInfo:
    """ISO, aperture, shutter and flash state from EXIF, when Pillow can see it.

    Pillow reads the EXIF of JPEGs and of the TIFF container most RAW formats
    use. Anything it cannot read comes back as ``None``, never as a guess.
    """
    try:
        from PIL import Image
    except ImportError:
        return ExposureInfo()
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            detail = exif.get_ifd(0x8769) if exif else {}
    except Exception:
        return ExposureInfo()
    iso = detail.get(0x8827)  # ISOSpeedRatings
    if isinstance(iso, Sequence) and not isinstance(iso, (str, bytes)):
        iso = iso[0] if iso else None
    flash = detail.get(0x9209)  # Flash; bit 0 = fired
    return ExposureInfo(
        iso=_ratio(iso),
        f_number=_ratio(detail.get(0x829D)),
        shutter=_ratio(detail.get(0x829A)),
        flash_fired=None if flash is None else bool(int(flash) & 1),
    )


def _decode_raw(path: Path) -> Tensor:
    try:
        import rawpy
    except ImportError as error:
        raise ImportError(
            f"decoding {path.name} needs rawpy: pip install 'atlas-relight[capture]'"
        ) from error
    import numpy as np

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            user_flip=0,
        )
    return torch.from_numpy(rgb.astype(np.float32) / 65535.0)


def decode_image(path: Path | str, *, max_side: Optional[int] = None) -> DecodedImage:
    """Decode one photograph to ``float32`` linear radiance."""
    from ..imageio import read_image

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        linear, exact = _decode_raw(path), True
    elif suffix in IMAGE_SUFFIXES:
        if suffix in (".jpg", ".jpeg") and not _has_opencv():
            linear = _decode_with_pillow(path)
        else:
            linear = read_image(path).to(torch.float32)[..., :3]
        if linear.shape[-1] == 1:
            linear = linear.expand(-1, -1, 3)
        linear, exact = srgb_to_linear(linear), False
    else:
        raise ValueError(f"not an image this pipeline decodes: {path}")
    return DecodedImage(
        path=path,
        linear=resize_linear(linear, max_side),
        linear_exact=exact,
        exposure=read_exposure(path),
    )


def _has_opencv() -> bool:
    from ..imageio import opencv_available

    return opencv_available()


def _decode_with_pillow(path: Path) -> Tensor:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)
