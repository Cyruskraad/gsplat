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

"""Reading a capture, and cutting it into the four sets the gate needs.

The loader is small because the interesting decisions are all about the split,
and there are two of them.

**A relighting capture is a grid, not a list.** Each shot is a *(view, light)*
pair, so holding out views and holding out lights are different cuts through
the same data and a frame can be held out by either, both, or neither. That
gives four sets, and naming them is most of the job:

===================  =========================  ==========================
set                  view                       light
===================  =========================  ==========================
``train``            seen                       seen
``held_out_view``    **unseen**                 seen
``held_out_light``   seen                       **unseen**
``held_out_both``    **unseen**                 **unseen**
===================  =========================  ==========================

``held_out_view`` and ``held_out_light`` are the pair whose *difference* is the
gate. ``held_out_both`` is the hardest set and is reported separately rather
than folded into either, because averaging it in would flatter neither number
honestly.

**The split is written once and never regenerated.** ``split.json`` records the
indices and the manifest hash they were computed from. A split that silently
recomputes when the data changes is a split that can be tuned, accidentally or
otherwise, and every number measured against it stops being comparable with
every number measured before.

Images are read on demand, never all at once: a 4k capture of a few hundred
shots is tens of gigabytes and the trainer only ever needs one at a time.
Everything comes back as **linear radiance** -- the manifest states its colour
space and a non-linear one is refused rather than silently used, because the
one thing this project never does is arithmetic on gamma-encoded values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..functional.splits import Split, split_lights, split_views
from ..imageio import read_png
from .synthetic import nerf_to_viewmat

__all__ = [
    "Frame",
    "Capture",
    "FrameSplit",
    "load_capture",
    "SET_NAMES",
]

#: The four sets, in the order they are reported.
SET_NAMES = ("train", "held_out_view", "held_out_light", "held_out_both")


@dataclass(frozen=True)
class Frame:
    """One shot: a camera, a light, and where its pixels live."""

    index: int
    image_path: Path
    viewmat: Tensor  #: [4, 4] world-to-camera, OpenCV
    view_index: int
    light_index: int
    light_position: Tensor  #: [3] world space
    light_intensity: Tensor  #: [3] RGB at `reference_distance`
    reference_distance: float
    exposure: float
    mask_path: Optional[Path] = None


@dataclass(frozen=True)
class FrameSplit:
    """Frame indices in each of the four sets, plus the splits they came from."""

    train: Tuple[int, ...]
    held_out_view: Tuple[int, ...]
    held_out_light: Tuple[int, ...]
    held_out_both: Tuple[int, ...]
    views: Split
    lights: Split
    manifest_hash: str

    def __getitem__(self, name: str) -> Tuple[int, ...]:
        if name not in SET_NAMES:
            raise KeyError(f"unknown set {name!r}; the four are {SET_NAMES}")
        return getattr(self, name)

    def counts(self) -> Dict[str, int]:
        return {name: len(self[name]) for name in SET_NAMES}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "frames": {name: list(self[name]) for name in SET_NAMES},
            "views": {
                k: v.tolist() for k, v in zip(("train", "val", "test"), self.views)
            },
            "lights": {
                k: v.tolist() for k, v in zip(("train", "val", "test"), self.lights)
            },
        }


def _hash_manifest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class Capture:
    """A capture on disk. Images are read when asked for, not before."""

    def __init__(self, root: Path, manifest: Dict[str, Any], manifest_text: str):
        self.root = Path(root)
        self.manifest = manifest
        self.manifest_hash = _hash_manifest(manifest_text)

        block = manifest.get("atlas", {})
        colour_space = block.get("colour_space", "unknown")
        if colour_space != "linear":
            raise ValueError(
                f"the manifest says colour_space={colour_space!r}; this loader "
                f"only reads linear radiance, and converting here would hide "
                f"where the conversion happened. Decode to linear on export."
            )
        self.scale = float(block.get("scale", 1.0))
        self.ambient = float(block.get("ambient", 0.0))
        self.flash_mode = block.get("flash_mode", "unknown")
        #: False when the light is a fixed function of the camera, in which case
        #: the two held-out sets select the same shots and their difference is
        #: not a measurement. See `atlas/data/synthetic.py`.
        self.splits_are_independent = bool(block.get("splits_are_independent", True))

        self.width = int(manifest["w"])
        self.height = int(manifest["h"])
        self.intrinsics = torch.tensor(
            [
                [float(manifest["fl_x"]), 0.0, float(manifest["cx"])],
                [0.0, float(manifest["fl_y"]), float(manifest["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        self.frames: List[Frame] = [
            self._frame(index, record)
            for index, record in enumerate(manifest["frames"])
        ]

    def _frame(self, index: int, record: Dict[str, Any]) -> Frame:
        block = record.get("atlas")
        if block is None:
            raise ValueError(
                f"frame {index} ({record.get('file_path')}) has no 'atlas' block; "
                f"a relighting capture needs a light position per shot, and a "
                f"plain NeRF manifest does not carry one"
            )
        mask = block.get("mask_path")
        return Frame(
            index=index,
            image_path=self.root / record["file_path"],
            viewmat=nerf_to_viewmat(
                torch.tensor(record["transform_matrix"], dtype=torch.float64)
            ),
            view_index=int(block["view_index"]),
            light_index=int(block["light_index"]),
            light_position=torch.tensor(block["light_position"], dtype=torch.float64),
            light_intensity=torch.tensor(block["light_intensity"], dtype=torch.float64),
            reference_distance=float(block.get("reference_distance", 1.0)),
            exposure=float(block.get("exposure", 1.0)),
            mask_path=None if mask is None else self.root / mask,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    # -- pixels --

    def image(self, index: int, *, subtract_ambient: bool = True) -> Tensor:
        """``[H, W, 3]`` linear radiance for one frame, read from disk now.

        The stored file holds radiance divided by the manifest's ``scale`` and
        clipped to ``[0, 1]``, so this multiplies it back. Exposure is divided
        out here rather than in the loss: a per-shot exposure is a property of
        the capture, not of the model, and leaving it in would make the model
        learn the photographer.
        """
        frame = self.frames[index]
        pixels = read_png(frame.image_path).to(torch.float64) / 255.0
        if pixels.shape[-1] == 1:
            pixels = pixels.expand(-1, -1, 3)
        radiance = pixels[..., :3] * self.scale
        if subtract_ambient and self.ambient != 0.0:
            radiance = radiance - self.ambient
        return (radiance / frame.exposure).clamp_min(0.0)

    def mask(self, index: int) -> Optional[Tensor]:
        """``[H, W]`` coverage in ``[0, 1]``, or ``None`` if the capture has none."""
        frame = self.frames[index]
        if frame.mask_path is None:
            return None
        return read_png(frame.mask_path).to(torch.float64)[..., 0] / 255.0

    # -- geometry of the grid --

    @property
    def view_indices(self) -> List[int]:
        return sorted({f.view_index for f in self.frames})

    @property
    def light_indices(self) -> List[int]:
        return sorted({f.light_index for f in self.frames})

    def camera_positions(self) -> Tensor:
        """``[V, 3]`` camera centres, one per distinct view, in view order."""
        seen: Dict[int, Tensor] = {}
        for frame in self.frames:
            if frame.view_index not in seen:
                rotation = frame.viewmat[:3, :3]
                seen[frame.view_index] = -rotation.T @ frame.viewmat[:3, 3]
        return torch.stack([seen[v] for v in self.view_indices])

    def light_directions(self) -> Tensor:
        """``[L, 3]`` unit directions to each distinct light, in light order.

        Directions rather than positions, because that is what
        :func:`~atlas.functional.splits.split_lights` separates on: two flashes
        a metre apart along the same bearing are nearly the same measurement
        once the near-field factor is divided out.
        """
        seen: Dict[int, Tensor] = {}
        for frame in self.frames:
            if frame.light_index not in seen:
                position = frame.light_position
                seen[frame.light_index] = position / position.norm().clamp_min(1e-12)
        return torch.stack([seen[l] for l in self.light_indices])

    # -- the split --

    def split(
        self,
        *,
        num_val_views: int = 1,
        num_test_views: int = 1,
        num_val_lights: int = 1,
        num_test_lights: int = 1,
        path: Optional[Path | str] = None,
    ) -> FrameSplit:
        """Cut the capture four ways, reading ``split.json`` if one exists.

        The file is authoritative. Once a split has been written, changing the
        arguments does not change the split -- it raises, because a result
        measured against one split is not comparable with a result measured
        against another, and silently re-cutting is how that stops being
        noticed.
        """
        path = Path(path) if path is not None else self.root / "split.json"
        if path.is_file():
            return self._load_split(path)

        views = split_views(self.camera_positions(), num_val_views, num_test_views)
        lights = split_lights(self.light_directions(), num_val_lights, num_test_lights)
        split = self._assemble(views, lights)
        path.write_text(json.dumps(split.to_dict(), indent=1, sort_keys=True))
        return split

    def _assemble(self, views: Split, lights: Split) -> FrameSplit:
        held_views = set(views.val.tolist()) | set(views.test.tolist())
        held_lights = set(lights.val.tolist()) | set(lights.test.tolist())
        buckets: Dict[str, List[int]] = {name: [] for name in SET_NAMES}
        for frame in self.frames:
            view_held = frame.view_index in held_views
            light_held = frame.light_index in held_lights
            if view_held and light_held:
                buckets["held_out_both"].append(frame.index)
            elif view_held:
                buckets["held_out_view"].append(frame.index)
            elif light_held:
                buckets["held_out_light"].append(frame.index)
            else:
                buckets["train"].append(frame.index)
        return FrameSplit(
            train=tuple(buckets["train"]),
            held_out_view=tuple(buckets["held_out_view"]),
            held_out_light=tuple(buckets["held_out_light"]),
            held_out_both=tuple(buckets["held_out_both"]),
            views=views,
            lights=lights,
            manifest_hash=self.manifest_hash,
        )

    def _load_split(self, path: Path) -> FrameSplit:
        stored = json.loads(path.read_text())
        if stored.get("manifest_hash") != self.manifest_hash:
            raise ValueError(
                f"{path} was computed from a different manifest "
                f"({stored.get('manifest_hash')} vs {self.manifest_hash}). The "
                f"data under it changed. Results measured against the two are "
                f"not comparable, so delete the split deliberately or point at "
                f"the capture it belongs to."
            )
        frames = stored["frames"]

        def to_split(block: Dict[str, Any]) -> Split:
            return Split(
                train=torch.tensor(block["train"], dtype=torch.long),
                val=torch.tensor(block["val"], dtype=torch.long),
                test=torch.tensor(block["test"], dtype=torch.long),
            )

        return FrameSplit(
            train=tuple(frames["train"]),
            held_out_view=tuple(frames["held_out_view"]),
            held_out_light=tuple(frames["held_out_light"]),
            held_out_both=tuple(frames["held_out_both"]),
            views=to_split(stored["views"]),
            lights=to_split(stored["lights"]),
            manifest_hash=stored["manifest_hash"],
        )


def load_capture(root: Path | str) -> Capture:
    """Read ``<root>/transforms.json`` and return the capture it describes."""
    root = Path(root)
    manifest_path = root / "transforms.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no transforms.json in {root}. Run "
            f"`python -m atlas.tools.inspect_capture {root}` to see what is there."
        )
    text = manifest_path.read_text()
    manifest = json.loads(text)
    for key in ("frames", "w", "h", "fl_x", "fl_y", "cx", "cy"):
        if key not in manifest:
            raise ValueError(f"{manifest_path} has no {key!r}")
    if not manifest["frames"]:
        raise ValueError(f"{manifest_path} lists no frames")
    return Capture(root, manifest, text)
