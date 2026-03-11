#!/usr/bin/env python3
"""
CLI entry point for the HDF5 to LeRobot Pipeline.

Usage:
    python -m src --config config.json
    python -m src --config config.json --skip-cleaning
    python -m src --config config.json --skip-conversion
"""

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="HDF5 to LeRobot Pipeline (Clean + Convert)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to pipeline config JSON",
    )
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip cleaning step")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip conversion step")

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"[ERROR] Config file not found: {args.config}")
        return

    run_pipeline(
        args.config, skip_cleaning=args.skip_cleaning, skip_conversion=args.skip_conversion
    )


if __name__ == "__main__":
    main()
