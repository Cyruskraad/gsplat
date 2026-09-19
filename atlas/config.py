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

"""The configuration tree, and the hash that identifies a run.

Two properties matter here and neither is negotiable.

**A typo fails in the first second, not the first hour.** Layered YAML is
convenient and silently forgiving, which is the worst combination: write
``num_atms: 64`` and an unvalidated loader gives you a run that trains for six
hours at the default 32 and tells you nothing. So every key is checked against a
typed tree, and an unknown one raises immediately, naming itself and suggesting
the closest key that does exist.

**The same configuration always produces the same hash.** That hash is what a
run directory is named after, what the results ledger joins on, and what says
whether two experiments differ. It must not depend on key order in the YAML, on
whether a float was written ``1`` or ``1.0``, or on which layer a value came
from. Values are coerced into the typed tree first and hashed from there, so all
three collapse before the hash sees them.

Layering is plain deepest-last merge::

    load_config(["configs/base.yaml", "configs/object.yaml"],
                overrides={"atoms.count": 64})

with dotted overrides applied on top, which is what a command line supplies.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "AtomConfig",
    "RuntimeConfig",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "Config",
    "ConfigError",
    "load_config",
    "from_dict",
    "to_dict",
    "config_hash",
    "apply_overrides",
    "deep_merge",
]


class ConfigError(ValueError):
    """A configuration that cannot be used, explained in terms of the file."""


# --- the tree ---------------------------------------------------------------
#
# Deliberately small. The trainer does not exist yet, and inventing its full
# parameter set now would be guessing at an interface -- the same mistake as
# writing models/ before the loader. What is here is what has already been
# decided; extending it is adding a field to a dataclass.


@dataclass
class AtomConfig:
    """The light basis the transport is expressed in."""

    # 32, not the design's 128: near-field training needs a per-primitive [N, B]
    # intermediate, 77 MB at 600k primitives and B=32 but 307 MB at B=128,
    # before autograd. Raise it once the memory has been measured on hardware.
    count: int = 32
    sharpness: Optional[float] = None


@dataclass
class DataConfig:
    """Where the capture is and how it is split."""

    capture_dir: str = ""
    # Held out by farthest-point selection over camera positions and light
    # directions. Both, because held-out views measure novel-view synthesis and
    # held-out lights measure whether transport was learned at all.
    num_val_views: int = 8
    num_test_views: int = 8
    num_val_lights: int = 6
    num_test_lights: int = 6
    mask_dir: Optional[str] = None
    split_manifest: Optional[str] = None


@dataclass
class ModelConfig:
    """How the model starts."""

    # A fixed-light reconstruction of the same object. The single biggest
    # accelerator available for a first run.
    init_ply: Optional[str] = None
    #: Primitives scattered when there is no PLY to start from. A poor start,
    #: and an honest one; a fixed-light reconstruction is far better.
    init_count: int = 4096
    near_field: bool = True
    profile_exponent: float = 0.0


@dataclass
class OptimConfig:
    max_steps: int = 30_000
    batch_size: int = 1
    #: Steps of gradient accumulation per optimiser step. Raises the effective
    #: batch without raising peak memory, which matters when one frame already
    #: carries [N, 3, B].
    grad_accum: int = 1

    transport_lr: float = 2.5e-3
    means_lr: float = 1.6e-4
    quats_lr: float = 1e-3
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    #: The basis moves slowly or it drags the transport around with it: every
    #: coefficient in the model is expressed in terms of it.
    atoms_lr: float = 1e-4

    #: "cosine" or "constant".
    schedule: str = "cosine"
    warmup_steps: int = 300
    min_lr_scale: float = 0.01
    grad_clip: float = 0.0

    #: The L in ATLAS. Off for a first run so there is a fixed-basis baseline.
    learn_atoms: bool = False

    #: Loss = l1 + ssim_weight * (1 - SSIM), both in the log1p domain.
    ssim_weight: float = 0.2
    #: Evaluations with no improvement before stopping. 0 disables.
    early_stop_patience: int = 0

    eval_every: int = 2_000
    save_every: int = 5_000


@dataclass
class RuntimeConfig:
    """Where the run executes, and at what precision.

    These belong in the config rather than in command-line flags because the
    config hash names the run: a result produced with TF32 on is not the same
    result, and nothing downstream could tell.
    """

    #: "auto", "cpu", "cuda" or "cuda:N". "auto" falls back and says so; an
    #: explicit "cuda" that cannot be honoured raises.
    device: str = "auto"
    #: "auto", "gsplat" or "reference". The reference renderer is roughly a
    #: thousand times slower and is for smoke tests and oracles.
    backend: str = "auto"
    #: Off, deliberately. TF32 keeps ten mantissa bits and the exactness gate
    #: lives at 1e-5 in float32. See atlas/device.py.
    allow_tf32: bool = False
    deterministic: bool = True
    #: Primitives per contraction chunk. 0 sizes it from free memory.
    chunk_size: int = 0
    #: Check every contraction chunk for non-finite values. Costs a device
    #: synchronisation per chunk; worth it while a run is being brought up.
    validate_chunks: bool = False


@dataclass
class Config:
    """The whole configuration. Its hash identifies a run."""

    atoms: AtomConfig = field(default_factory=AtomConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    seed: int = 0
    run_root: str = "runs"
    run_name: Optional[str] = None


# --- loading ----------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``overlay`` onto ``base``, recursing into nested mappings.

    A scalar in the overlay replaces whatever was there. Two mappings merge.
    That is the whole rule; there is no list-merging mode, because "append or
    replace?" is a question no one should have to remember the answer to.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def apply_overrides(
    data: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply dotted-path overrides, e.g. ``{"atoms.count": 64}``.

    Intermediate keys must already exist. Creating them on demand would let
    ``--set atomz.count=64`` silently add a branch nothing reads, which is the
    failure this module exists to prevent.
    """
    result: Dict[str, Any] = json.loads(json.dumps(dict(data), default=str))
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(
                    f"override {dotted!r}: no section {part!r} to set it in"
                )
            cursor = cursor[part]
        if not isinstance(cursor, dict):
            raise ConfigError(f"override {dotted!r}: {parts[-2]!r} is not a section")
        cursor[parts[-1]] = value
    return result


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigError(
            "reading YAML configs needs PyYAML. pip install -e '.[dev]'"
        ) from exc
    try:
        loaded = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path}: top level must be a mapping, got {type(loaded).__name__}"
        )
    return loaded


def load_config(
    paths: Sequence[Path | str] = (),
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load layered YAML, apply dotted overrides, and validate into a `Config`.

    Args:
        paths: Config files, merged deepest-last.
        overrides: Dotted-path values applied after the merge.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: Naming the file, the key, and where possible the key that
            was probably meant.
    """
    merged: Dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        merged = deep_merge(merged, _read_yaml(path))
    if overrides:
        merged = apply_overrides(merged, overrides)
    return from_dict(Config, merged)


# --- typed construction -----------------------------------------------------


@lru_cache(maxsize=None)
def _hints(cls: Any) -> Tuple[Tuple[str, Any], ...]:
    """Resolved annotations for a config dataclass.

    ``dataclasses.fields(cls)[i].type`` is a *string* here, because this module
    uses ``from __future__ import annotations`` -- so reading it directly gives
    ``'AtomConfig'`` rather than the class, and every nested section fails to
    build. ``get_type_hints`` evaluates them against the defining module.
    """
    return tuple(get_type_hints(cls).items())


def _is_optional(annotation: Any) -> Tuple[bool, Any]:
    """Unwrap ``Optional[X]`` into ``(True, X)``."""
    if get_origin(annotation) is None:
        return False, annotation
    args = get_args(annotation)
    if type(None) in args:
        remaining = [a for a in args if a is not type(None)]
        if len(remaining) == 1:
            return True, remaining[0]
    return False, annotation


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    optional, inner = _is_optional(annotation)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where}: null is not allowed here")

    if inner is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true or false, got {value!r}")
    if inner is int:
        # bool is a subclass of int; accepting it would turn `true` into 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if inner is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        result = float(value)
        if not math.isfinite(result):
            # NaN and infinity are not representable in canonical JSON and would
            # make the hash unstable, quite apart from being nonsense here.
            raise ConfigError(f"{where}: must be finite, got {value!r}")
        return result
    if inner is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if is_dataclass(inner):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a section, got {value!r}")
        return from_dict(inner, value, where=where)
    raise ConfigError(f"{where}: unsupported config type {inner!r}")


def from_dict(cls: Any, data: Mapping[str, Any], *, where: str = "") -> Any:
    """Build a config dataclass from a mapping, rejecting anything unexpected.

    Args:
        cls: A config dataclass.
        data: The mapping to read.
        where: Dotted path prefix, for error messages.

    Raises:
        ConfigError: On an unknown key, with a suggestion when one is close.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a config dataclass")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or 'config'}: expected a mapping")

    annotations = dict(_hints(cls))
    known = {f.name: annotations[f.name] for f in fields(cls) if f.name in annotations}
    for key in data:
        if key in known:
            continue
        path = f"{where}.{key}" if where else key
        close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
        suggestion = f" Did you mean {close[0]!r}?" if close else ""
        raise ConfigError(
            f"unknown config key {path!r}.{suggestion} "
            f"Valid keys here: {sorted(known)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, annotation in known.items():
        if name not in data:
            continue
        path = f"{where}.{name}" if where else name
        kwargs[name] = _coerce(data[name], annotation, path)
    return cls(**kwargs)


def to_dict(config: Any) -> Dict[str, Any]:
    """The config as plain data, ready to serialise."""
    if not is_dataclass(config):
        raise ConfigError(f"{config!r} is not a config dataclass instance")
    return dataclasses.asdict(config)


# --- identity ---------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Normalise so that equal configurations serialise identically."""
    if isinstance(value, Mapping):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError("non-finite float in config; cannot hash")
        # An integral float and the integer itself are different values here,
        # and must stay different: atoms.count = 32 is not atoms.count = 32.0.
        # repr gives the shortest round-tripping form, so 1.10 and 1.1 agree.
        return repr(value)
    return value


def config_hash(config: Any) -> str:
    """SHA-256 of the configuration's canonical form.

    Independent of key order in the source YAML, of how a float was written, and
    of which layer a value came from -- because the value is coerced into the
    typed tree before it is hashed.
    """
    payload = _canonical(to_dict(config) if is_dataclass(config) else config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
