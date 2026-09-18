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

"""Run directories, provenance, and the ledger that accumulates across runs.

A run leaves behind a directory of a fixed shape::

    runs/<name>-<timestamp>-<confighash8>/
        config.yaml         the resolved configuration, not the files it came from
        provenance.json     commit, dirty flag, hardware, package versions, data hash
        log.jsonl           append-only structured log
        metrics.json        latest value of every metric
        metrics.csv         every metric at every step
        ckpts/  renders/
        RUNNING | COMPLETED | FAILED

Three decisions in there are worth naming.

**The directory is never reused.** Its name carries a timestamp and the config
hash, and creating it fails if it exists. A run that overwrites another's
artifacts destroys the evidence for a result someone may already have quoted.

**Status is a file, not a log line.** A process that is killed -- OOM, a reboot,
a runner timeout -- writes nothing on its way out. A directory still marked
`RUNNING` hours later is the honest record of that, where a log that simply
stops is ambiguous. Used as a context manager, an exception marks `FAILED` and
re-raises.

**Everything is written atomically**, via a temporary file and a rename. A crash
during a metrics write should not leave a truncated JSON file that makes the run
look corrupt when only the last update was lost.

**The ledger is JSONL, not CSV.** Runs accumulate metrics that later runs do not
have and vice versa; a CSV would need its header rewritten every time the schema
grew, which is exactly the operation that loses data. Appending a single line is
atomic on POSIX for reasonable line lengths, so concurrent runs do not corrupt
each other. :func:`ledger_to_csv` renders it flat when something wants a table.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .config import Config, config_hash, to_dict

__all__ = [
    "RunDirectory",
    "collect_provenance",
    "hash_directory",
    "append_ledger",
    "read_ledger",
    "ledger_to_csv",
    "RUNNING",
    "COMPLETED",
    "FAILED",
]

RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
_STATUSES = (RUNNING, COMPLETED, FAILED)


# --- atomic writes ----------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _append_line(path: Path, line: str) -> None:
    """Append one line, healing a previous one that was cut off mid-write.

    A process killed during a write can leave a record with no terminating
    newline. Appending straight after it fuses the two into a single
    unparseable line and loses **both** -- the torn one, which is expected, and
    the new one, which is not. Checking the last byte costs one seek and turns a
    silent double loss into a single skipped line.

    Two processes may both add the newline, leaving a blank line. `_read_jsonl`
    skips those.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(line.encode())


# --- provenance -------------------------------------------------------------


def _git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_provenance(
    *, source_root: Optional[Path] = None, dataset_hash: Optional[str] = None
) -> Dict[str, Any]:
    """Everything needed to answer "what produced this?" a year from now.

    The **dirty** flag matters as much as the commit. A result from a working
    tree with uncommitted changes is not reproducible from that commit, and
    recording the commit alone would imply that it is.
    """
    root = Path(source_root or Path(__file__).resolve().parent.parent)
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    provenance: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": list(sys.argv),
        "dataset_hash": dataset_hash,
    }

    try:
        import torch

        provenance["torch"] = torch.__version__
        provenance["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            provenance["cuda_device"] = torch.cuda.get_device_name(0)
            provenance["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            provenance["cuda_version"] = torch.version.cuda
    except Exception:  # pragma: no cover - torch is a hard dependency, but be safe
        provenance["torch"] = None

    try:
        import gsplat

        provenance["gsplat"] = getattr(gsplat, "__version__", "unknown")
    except Exception:
        provenance["gsplat"] = None

    return provenance


def hash_directory(
    path: Path | str,
    *,
    suffixes: Optional[Iterable[str]] = None,
    include_contents: bool = False,
) -> str:
    """A hash identifying the contents of a capture directory.

    By default this is a *manifest* hash -- relative paths and byte sizes -- not
    a content hash. Reading a few hundred gigabytes to notice that nothing
    changed is not a reasonable thing to do at the start of every run, and the
    manifest catches what actually goes wrong: a file added, removed, renamed or
    re-exported at a different size.

    Pass ``include_contents=True`` when the stronger guarantee is worth the
    read, such as when publishing a result.

    Args:
        path: Directory to hash.
        suffixes: Restrict to these lowercase extensions, e.g. ``{".jpg"}``.
        include_contents: Hash file bytes as well as the manifest.

    Returns:
        A hex digest, prefixed with the method so the two can never be confused.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    wanted = {s.lower() for s in suffixes} if suffixes else None

    digest = hashlib.sha256()
    entries: List[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if wanted is not None and candidate.suffix.lower() not in wanted:
            continue
        entries.append(candidate)

    for candidate in entries:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(str(candidate.stat().st_size).encode())
        if include_contents:
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)

    method = "sha256-content" if include_contents else "sha256-manifest"
    return f"{method}:{len(entries)}:{digest.hexdigest()}"


# --- the run directory ------------------------------------------------------


class RunDirectory:
    """One run's artifacts, in a directory that is never reused."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ckpts = self.path / "ckpts"
        self.renders = self.path / "renders"

    # -- creation --

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        root: Optional[Path | str] = None,
        name: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> "RunDirectory":
        """Create the directory, write the config and provenance, mark RUNNING.

        Args:
            config: The resolved configuration. Its hash names the directory.
            root: Where runs live. Defaults to ``config.run_root``.
            name: Human-readable prefix. Defaults to ``config.run_name`` or
                ``"run"``.
            dataset_hash: From :func:`hash_directory`, recorded in provenance.
            now: For tests. Defaults to the current UTC time.

        Raises:
            FileExistsError: If the directory already exists. Never reuse.
        """
        digest = config_hash(config)
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        label = name or config.run_name or "run"
        directory = Path(root or config.run_root) / f"{label}-{stamp}-{digest[:8]}"
        # exist_ok=False is the point: a second run must not land on the first.
        directory.mkdir(parents=True, exist_ok=False)

        run = cls(directory)
        run.ckpts.mkdir()
        run.renders.mkdir()
        run.write_config(config)
        run.write_provenance(dataset_hash=dataset_hash)
        run.mark(RUNNING)
        run.log(event="run_created", config_hash=digest, path=str(directory))
        return run

    @classmethod
    def open(cls, path: Path | str) -> "RunDirectory":
        """Reopen an existing run directory, for resume or inspection."""
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"not a run directory: {directory}")
        return cls(directory)

    # -- writing --

    def write_config(self, config: Config) -> None:
        """Write the *resolved* config, not the layered files it came from.

        Reproducing a run from its inputs means reproducing what was actually
        used, and the merge of three YAML files plus overrides is not something
        anyone should have to replay by hand.
        """
        payload = to_dict(config)
        try:
            import yaml

            text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
        except ImportError:  # pragma: no cover - PyYAML is a dependency
            text = json.dumps(payload, indent=2, sort_keys=True)
        _write_atomic(self.path / "config.yaml", text)
        _write_atomic(self.path / "config_hash.txt", config_hash(config) + "\n")

    def write_provenance(self, *, dataset_hash: Optional[str] = None) -> Dict[str, Any]:
        provenance = collect_provenance(dataset_hash=dataset_hash)
        _write_atomic(
            self.path / "provenance.json",
            json.dumps(provenance, indent=2, sort_keys=True),
        )
        return provenance

    def log(self, **fields: Any) -> None:
        """Append one structured record to ``log.jsonl``."""
        record = {"t": time.time(), **fields}
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        _append_line(self.path / "log.jsonl", line)

    def record_metrics(self, step: int, **metrics: Any) -> None:
        """Record metrics at a step: a CSV row, and the latest into JSON."""
        row = {"step": int(step), **metrics}
        line = json.dumps(row, sort_keys=True, default=str) + "\n"
        _append_line(self.path / "metrics.jsonl", line)

        latest_path = self.path / "metrics.json"
        latest: Dict[str, Any] = {}
        if latest_path.is_file():
            try:
                latest = json.loads(latest_path.read_text())
            except json.JSONDecodeError:
                latest = {}
        latest.update(row)
        _write_atomic(latest_path, json.dumps(latest, indent=2, sort_keys=True))
        self._rewrite_metrics_csv()

    def _rewrite_metrics_csv(self) -> None:
        """Render ``metrics.jsonl`` flat.

        Rewritten rather than appended because a later step may introduce a
        metric earlier steps did not have, and a CSV with a fixed header cannot
        grow a column without being rewritten anyway.
        """
        rows = list(self.metrics())
        if not rows:
            return
        columns = ["step"] + sorted({k for row in rows for k in row} - {"step"})
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(_csv_cell(row.get(c)) for c in columns))
        _write_atomic(self.path / "metrics.csv", "\n".join(lines) + "\n")

    def metrics(self) -> Iterator[Dict[str, Any]]:
        path = self.path / "metrics.jsonl"
        if not path.is_file():
            return iter(())
        return _read_jsonl(path)

    def mark(self, status: str) -> None:
        """Set the terminal status marker, removing any previous one."""
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")
        for other in _STATUSES:
            (self.path / other).unlink(missing_ok=True)
        _write_atomic(self.path / status, datetime.now(timezone.utc).isoformat() + "\n")

    @property
    def status(self) -> Optional[str]:
        for candidate in _STATUSES:
            if (self.path / candidate).is_file():
                return candidate
        return None

    # -- context manager --

    def __enter__(self) -> "RunDirectory":
        self.mark(RUNNING)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.mark(COMPLETED)
            self.log(event="run_completed")
        else:
            self.mark(FAILED)
            self.log(
                event="run_failed",
                error_type=exc_type.__name__,
                error=str(exc),
            )
        return False  # never swallow the exception


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A line torn by a crash mid-write. Skip it rather than lose the
                # rest of the file.
                continue


# --- the ledger -------------------------------------------------------------


def append_ledger(path: Path | str, row: Mapping[str, Any]) -> None:
    """Append one run's summary to the ledger.

    Append-only JSONL. A run that adds a metric no previous run had does not
    disturb them, and two runs finishing at once do not corrupt each other.
    """
    line = json.dumps(dict(row), sort_keys=True, default=str) + "\n"
    _append_line(Path(path), line)


def read_ledger(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return list(_read_jsonl(path))


def ledger_to_csv(path: Path | str, destination: Path | str) -> Path:
    """Render the ledger as a flat table, unioning columns across all rows."""
    rows = read_ledger(path)
    destination = Path(destination)
    if not rows:
        _write_atomic(destination, "")
        return destination
    preferred = [
        c
        for c in ("run", "config_hash", "git_commit", "step")
        if any(c in r for r in rows)
    ]
    remaining = sorted({k for row in rows for k in row} - set(preferred))
    columns = preferred + remaining
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(c)) for c in columns))
    _write_atomic(destination, "\n".join(lines) + "\n")
    return destination
