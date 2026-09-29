# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreground masks from an off-the-shelf matting network.

A relighting loss that includes the turntable, the backdrop and the lab is
spending most of its capacity on things that are not the object. ``rembg``
wraps the current dichotomous-segmentation models behind one call; the default
is **BiRefNet** (``birefnet-general``), the strongest of them on object
captures. The masks are soft (matting, not segmentation), which the loss
already accepts as per-pixel weights.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

__all__ = ["Masker", "make_masker", "MASK_BACKENDS"]

#: ``[H, W, 3]`` float sRGB in ``[0, 1]`` -> ``[H, W]`` coverage in ``[0, 1]``.
Masker = Callable[[Tensor], Tensor]

MASK_BACKENDS = ("none", "rembg")


def make_masker(
    backend: str = "none", model: str = "birefnet-general"
) -> Optional[Masker]:
    """A mask function, or ``None`` when masks are not wanted."""
    if backend == "none":
        return None
    if backend != "rembg":
        raise ValueError(
            f"mask backend must be one of {MASK_BACKENDS}, got {backend!r}"
        )
    try:
        import rembg
    except ImportError as error:
        raise ImportError(
            "masks need rembg: pip install 'rembg[gpu]' (or 'rembg' for CPU)"
        ) from error
    import numpy as np
    from PIL import Image

    session = rembg.new_session(model)

    def masker(srgb: Tensor) -> Tensor:
        array = (srgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        out = rembg.remove(Image.fromarray(array), session=session, only_mask=True)
        return torch.from_numpy(np.asarray(out, dtype=np.float32) / 255.0)

    return masker
