#!/usr/bin/env python3
"""Run one immutable gsplat experiment with status, provenance, and telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from colmap_scripts.gsplat_job import finalize_status, initialize_status, sha256_file


def _capture(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _telemetry_loop(
    stop: threading.Event,
    process: subprocess.Popen[str],
    run_dir: Path,
    maximum_bytes: int,
) -> None:
    telemetry_path = run_dir / "gpu_telemetry.csv"
    telemetry_path.write_text(
        "timestamp,name,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_watts\n",
        encoding="utf-8",
    )
    while not stop.is_set():
        sample = _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        if sample:
            with telemetry_path.open("a", encoding="utf-8") as handle:
                for line in sample.splitlines():
                    handle.write(
                        f"{datetime.now(timezone.utc).isoformat()},{line}\n"
                    )
        if _directory_size(run_dir) > maximum_bytes and process.poll() is None:
            (run_dir / "resource_abort.txt").write_text(
                "artifact cap exceeded\n", encoding="utf-8"
            )
            process.terminate()
        stop.wait(5.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--provenance-file", type=Path, action="append", default=[])
    parser.add_argument("--minimum-free-disk-gib", type=float, default=300.0)
    parser.add_argument("--maximum-artifacts-gib", type=float, default=150.0)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)
    if not trainer_args:
        raise SystemExit("trainer subcommand and options are required after --")
    free_gib = shutil.disk_usage(_existing_parent(args.run_dir)).free / 1024**3
    if free_gib < args.minimum_free_disk_gib:
        raise SystemExit(
            f"free disk {free_gib:.1f} GiB is below {args.minimum_free_disk_gib:.1f} GiB"
        )

    initialize_status(args.run_dir)
    command = [str(args.python), str(args.trainer), *trainer_args]
    started = time.time()
    process: subprocess.Popen[str] | None = None
    stop = threading.Event()
    telemetry: threading.Thread | None = None
    try:
        source_hashes = {
            str(path.resolve()): sha256_file(path)
            for path in args.provenance_file
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "command_display": subprocess.list2cmdline(command),
            "working_directory": str(args.worktree.resolve()),
            "git_commit": _capture(["git", "rev-parse", "HEAD"], cwd=args.worktree),
            "git_status": _capture(["git", "status", "--short"], cwd=args.worktree),
            "gpu": _capture(["nvidia-smi", "-L"]),
            "free_disk_gib": free_gib,
            "minimum_free_disk_gib": args.minimum_free_disk_gib,
            "maximum_artifacts_gib": args.maximum_artifacts_gib,
            "source_sha256": source_hashes,
        }
        (args.run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diff = _capture(["git", "diff", "--binary"], cwd=args.worktree)
        (args.run_dir / "source.patch").write_text(diff + "\n", encoding="utf-8")

        process = subprocess.Popen(
            command,
            cwd=args.worktree,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        telemetry = threading.Thread(
            target=_telemetry_loop,
            args=(
                stop,
                process,
                args.run_dir,
                round(args.maximum_artifacts_gib * 1024**3),
            ),
            daemon=True,
        )
        telemetry.start()
        with (args.run_dir / "trainer.log").open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        exit_code = process.wait()
        stop.set()
        telemetry.join(timeout=10.0)
        elapsed = time.time() - started
        size_bytes = _directory_size(args.run_dir)
        summary = {
            "exit_code": exit_code,
            "runtime_seconds": elapsed,
            "artifact_bytes": size_bytes,
            "artifact_gib": size_bytes / 1024**3,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (args.run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        finalize_status(args.run_dir, succeeded=exit_code == 0, exit_code=exit_code)
        return exit_code
    except BaseException:
        stop.set()
        if process is not None and process.poll() is None:
            process.terminate()
        if telemetry is not None:
            telemetry.join(timeout=10.0)
        if (args.run_dir / "RUNNING").is_file():
            finalize_status(args.run_dir, succeeded=False, exit_code=1)
        raise


if __name__ == "__main__":
    sys.exit(main())
