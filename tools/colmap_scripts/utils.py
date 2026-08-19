"""General utility helpers used across the CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


from . import constants


class PipelineError(RuntimeError):
    """Base error for local validation or orchestration failures."""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_relative_path(root: Path, target: Path) -> Path:
    root = root.resolve()
    target = target.resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as err:
        raise PipelineError(f"{target} is outside allowed root {root}") from err
    return relative


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_cmd(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
    capture_output: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(a) for a in args)
    print(f"[RUN] {printable}")
    if dry_run:
        return subprocess.CompletedProcess(args=tuple(args), returncode=0, stdout="", stderr="")
    try:
        cp = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=dict(os.environ | dict(env or {})),
            check=False,
            text=True,
            capture_output=capture_output,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"Command timed out after {timeout_seconds}s: {printable}") from exc
    if cp.returncode != 0:
        raise PipelineError(
            f"Command failed with status {cp.returncode}: {printable}\n{cp.stderr}"
        )
    return cp


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_id() -> str:
    return datetime.utcnow().strftime(constants.RUN_ID_FORMAT)


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return asdict(value)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported manifest object: {type(value)!r}")


def unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique_values: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
