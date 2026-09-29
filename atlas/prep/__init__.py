# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture preparation: photographs in, a trainable capture out.

Every stage is an established tool -- LibRaw for RAW decoding, COLMAP with
ALIKED + LightGlue features and GLOMAP global mapping for poses, BiRefNet via
rembg for masks -- connected with the radiometric and metric bookkeeping that
relighting needs. Imports are lazy: ``atlas.prep`` loads without pycolmap,
rawpy or rembg, and each stage says what to install when it needs it.
"""

from .images import decode_image, list_images
from .pipeline import PrepConfig, export_reconstruction, prepare_capture
from .sfm import SfMConfig, learned_features_available, run_sfm

__all__ = [
    "PrepConfig",
    "SfMConfig",
    "prepare_capture",
    "export_reconstruction",
    "run_sfm",
    "learned_features_available",
    "decode_image",
    "list_images",
]
