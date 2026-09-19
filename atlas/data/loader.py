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

from ..functional.splits import (
    Split,
    farthest_point_indices,
    split_arc,
    split_lights,
    split_views,
)
from ..imageio import read_png
from .synthetic import nerf_to_viewmat

__all__ = [
    "Frame",
    "Capture",
    "FrameSplit",
    "CouplingReport",
    "load_capture",
    "SET_NAMES",
    "EXTRAPOLATION_SET_NAMES",
    "CO_LOCATED_DEGREES",
    "DECOUPLED_DEGREES",
]

#: The four sets of a capture whose lights and views vary independently.
SET_NAMES = ("train", "held_out_view", "held_out_light", "held_out_both")

#: The sets a co-located capture supports instead. Holding out a light there
#: holds out its view too, so the view/light pair collapses to one number; what
#: survives is interpolation against extrapolation, which is the question
#: relighting actually asks.
EXTRAPOLATION_SET_NAMES = (
    "train",
    "held_out_interpolated",
    "held_out_extrapolated",
)

#: Below this view-to-light angle the flash is effectively on the lens.
#: Measured: a hotshoe 5-20 cm from the lens on an object a metre away gives
#: 3-11 degrees, so there is no decoupling to exploit at all.
CO_LOCATED_DEGREES = 15.0

#: Above this the two are separated enough that a view/light split measures two
#: different things. Between the two the capture is partially decoupled and the
#: report says so rather than rounding to one or the other.
DECOUPLED_DEGREES = 45.0


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
    """Frame indices per set, and which scheme produced them.

    A capture whose lights and views vary independently gets the four-way cut
    (:data:`SET_NAMES`); a co-located one gets interpolation against
    extrapolation (:data:`EXTRAPOLATION_SET_NAMES`). The sets the scheme does
    not use are empty rather than absent, so a caller that iterates
    ``split.names`` works on either.
    """

    train: Tuple[int, ...]
    manifest_hash: str
    scheme: str = "independent"
    held_out_view: Tuple[int, ...] = ()
    held_out_light: Tuple[int, ...] = ()
    held_out_both: Tuple[int, ...] = ()
    held_out_interpolated: Tuple[int, ...] = ()
    held_out_extrapolated: Tuple[int, ...] = ()
    views: Optional[Split] = None
    lights: Optional[Split] = None

    @property
    def names(self) -> Tuple[str, ...]:
        """The sets this scheme actually populates."""
        return SET_NAMES if self.scheme == "independent" else EXTRAPOLATION_SET_NAMES

    def __getitem__(self, name: str) -> Tuple[int, ...]:
        if name not in SET_NAMES + EXTRAPOLATION_SET_NAMES:
            raise KeyError(f"unknown set {name!r}; this split has {self.names}")
        return getattr(self, name)

    def counts(self) -> Dict[str, int]:
        return {name: len(self[name]) for name in self.names}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "manifest_hash": self.manifest_hash,
            "scheme": self.scheme,
            "frames": {name: list(self[name]) for name in self.names},
        }
        for key, value in (("views", self.views), ("lights", self.lights)):
            if value is not None:
                payload[key] = {
                    k: v.tolist() for k, v in zip(("train", "val", "test"), value)
                }
        return payload


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

    # -- what kind of capture is this --

    def coupling_report(self) -> CouplingReport:
        """Measure how far each shot's light sits from its own view direction.

        Both directions are taken at the object's centre -- the centroid of the
        camera positions is a poor proxy, so the origin of the capture's own
        coordinate frame is used, which is where the generator and every SfM
        convention put the subject.

        The implied flash offset is the median distance from a camera to its
        own light. On a rigidly mounted flash that is the bracket arm; on a
        hand-held one it is meaningless, and the verdict already says which.
        """
        views, lights, offsets = [], [], []
        target = torch.zeros(3, dtype=torch.float64)
        for frame in self.frames:
            rotation = frame.viewmat[:3, :3]
            centre = -rotation.T @ frame.viewmat[:3, 3]
            to_view = target - centre
            to_light = target - frame.light_position
            views.append(to_view / to_view.norm().clamp_min(1e-12))
            lights.append(to_light / to_light.norm().clamp_min(1e-12))
            offsets.append(float((frame.light_position - centre).norm()))

        cosines = (torch.stack(views) * torch.stack(lights)).sum(-1).clamp(-1.0, 1.0)
        degrees = torch.rad2deg(torch.acos(cosines))
        median = float(degrees.median())
        if median < CO_LOCATED_DEGREES:
            verdict = "co_located"
        elif median < DECOUPLED_DEGREES:
            verdict = "partially_decoupled"
        else:
            verdict = "decoupled"

        offsets_tensor = torch.tensor(offsets, dtype=torch.float64)
        return CouplingReport(
            verdict=verdict,
            median_degrees=median,
            min_degrees=float(degrees.min()),
            max_degrees=float(degrees.max()),
            implied_offset_metres=float(offsets_tensor.median()),
            num_shots=len(self.frames),
        )

    # -- the split --

    def split(
        self,
        *,
        num_val_views: int = 1,
        num_test_views: int = 1,
        num_val_lights: int = 1,
        num_test_lights: int = 1,
        scheme: str = "auto",
        extrapolation_fraction: float = 0.2,
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

        if scheme == "auto":
            scheme = (
                "independent"
                if self.coupling_report().splits_are_independent
                else "extrapolation"
            )
        if scheme == "extrapolation":
            split = self._assemble_extrapolation(
                fraction=extrapolation_fraction,
                num_interpolated=num_test_views + num_test_lights,
            )
        elif scheme == "independent":
            views = split_views(self.camera_positions(), num_val_views, num_test_views)
            lights = split_lights(
                self.light_directions(), num_val_lights, num_test_lights
            )
            split = self._assemble(views, lights)
        else:
            raise ValueError(
                f"scheme must be 'auto', 'independent' or 'extrapolation', "
                f"got {scheme!r}"
            )
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
            scheme="independent",
            held_out_view=tuple(buckets["held_out_view"]),
            held_out_light=tuple(buckets["held_out_light"]),
            held_out_both=tuple(buckets["held_out_both"]),
            views=views,
            lights=lights,
            manifest_hash=self.manifest_hash,
        )

    def _assemble_extrapolation(
        self, *, fraction: float, num_interpolated: int
    ) -> FrameSplit:
        """Hold out one contiguous arc, and some samples inside what remains.

        The arc is removed first and the interpolation set is chosen from the
        frames that survive, so an "interpolated" frame really is surrounded by
        training frames rather than sitting on the lip of the hole.
        """
        positions = torch.stack(
            [-f.viewmat[:3, :3].T @ f.viewmat[:3, 3] for f in self.frames]
        )
        arc = split_arc(positions, fraction)
        extrapolated = [int(i) for i in arc.test]
        remaining = [int(i) for i in arc.train]

        interpolated: List[int] = []
        if num_interpolated > 0 and len(remaining) > num_interpolated + 1:
            inner = farthest_point_indices(
                positions[torch.tensor(remaining)], num_interpolated
            )
            interpolated = sorted(remaining[int(i)] for i in inner)

        held = set(extrapolated) | set(interpolated)
        return FrameSplit(
            train=tuple(i for i in range(len(self.frames)) if i not in held),
            scheme="extrapolation",
            held_out_interpolated=tuple(interpolated),
            held_out_extrapolated=tuple(extrapolated),
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

        def to_split(block):
            if block is None:
                return None
            return Split(
                train=torch.tensor(block["train"], dtype=torch.long),
                val=torch.tensor(block["val"], dtype=torch.long),
                test=torch.tensor(block["test"], dtype=torch.long),
            )

        return FrameSplit(
            train=tuple(frames["train"]),
            scheme=stored.get("scheme", "independent"),
            held_out_view=tuple(frames.get("held_out_view", ())),
            held_out_light=tuple(frames.get("held_out_light", ())),
            held_out_both=tuple(frames.get("held_out_both", ())),
            held_out_interpolated=tuple(frames.get("held_out_interpolated", ())),
            held_out_extrapolated=tuple(frames.get("held_out_extrapolated", ())),
            views=to_split(stored.get("views")),
            lights=to_split(stored.get("lights")),
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


@dataclass(frozen=True)
class CouplingReport:
    """How independent this capture's lights are from its cameras.

    Measured from the poses rather than read from the manifest, because a real
    capture arrives with no such flag and a synthetic one should be checked
    against its own geometry rather than trusted.
    """

    verdict: str  #: "co_located", "partially_decoupled" or "decoupled"
    median_degrees: float
    min_degrees: float
    max_degrees: float
    implied_offset_metres: Optional[float]
    num_shots: int

    @property
    def splits_are_independent(self) -> bool:
        return self.verdict == "decoupled"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "verdict": self.verdict,
            "median_degrees": self.median_degrees,
            "min_degrees": self.min_degrees,
            "max_degrees": self.max_degrees,
            "implied_offset_metres": self.implied_offset_metres,
            "num_shots": self.num_shots,
            "splits_are_independent": self.splits_are_independent,
        }
        return payload

    def __str__(self) -> str:
        offset = (
            "unknown"
            if self.implied_offset_metres is None
            else f"{self.implied_offset_metres * 100:.0f} cm"
        )
        advice = {
            "co_located": (
                "the flash is effectively on the lens. Holding out a light "
                "holds out its view, so the gate runs on interpolation against "
                "extrapolation instead."
            ),
            "partially_decoupled": (
                "partially separated. The view/light split is weak here; the "
                "extrapolation split is the more honest measurement."
            ),
            "decoupled": (
                "lights and views vary independently, so held-out view and "
                "held-out light measure different things."
            ),
        }[self.verdict]
        return (
            f"{self.verdict}: view-to-light angle "
            f"{self.median_degrees:.1f} deg median "
            f"({self.min_degrees:.1f}-{self.max_degrees:.1f}), implied flash "
            f"offset {offset} over {self.num_shots} shots.\n  {advice}"
        )
