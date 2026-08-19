"""Run-directory status and provenance helpers for gsplat experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .utils import PipelineError


def initialize_status(run_dir: Path) -> None:
    if run_dir.exists():
        raise PipelineError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "RUNNING").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def finalize_status(
    run_dir: Path, *, succeeded: bool, exit_code: int = 0
) -> Path:
    running = run_dir / "RUNNING"
    if not running.is_file():
        raise PipelineError(f"Run has no RUNNING status: {run_dir}")
    target = run_dir / ("COMPLETED" if succeeded else "FAILED")
    if target.exists():
        raise PipelineError(f"Final run status already exists: {target}")
    if succeeded:
        content = f"completed_at={datetime.now(timezone.utc).isoformat()}\n"
    else:
        content = f"exit_code={exit_code}\n"
    temporary = run_dir / f".{target.name}.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    running.unlink()
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
