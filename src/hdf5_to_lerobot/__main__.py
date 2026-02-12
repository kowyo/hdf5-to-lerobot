#!/usr/bin/env python3
"""
CLI entry point for the HDF5 to LeRobot Pipeline.

Usage:
    python -m src --config config.json
    python -m src --config config.json --skip-cleaning
    python -m src --config config.json --skip-conversion
"""

import argparse
import os

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="HDF5 to LeRobot Pipeline (Clean + Convert)")
    parser.add_argument(
        "--config",
        type=str,
        default="/scratch/e1583450/VLA_Flywheel/utils/dataset/1-convert_config.json",
        help="Path to pipeline config JSON",
    )
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip cleaning step")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip conversion step")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    run_pipeline(
        args.config, skip_cleaning=args.skip_cleaning, skip_conversion=args.skip_conversion
    )


if __name__ == "__main__":
    main()
