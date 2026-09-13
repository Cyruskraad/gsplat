#!/usr/bin/env python3
"""Export gsplat validation JSON files as CSV deliverables."""

from __future__ import annotations

import argparse
from pathlib import Path

from colmap_scripts.gsplat_report import export_metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for label, path in export_metrics_csv(args.stats_dir, args.output_dir).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
