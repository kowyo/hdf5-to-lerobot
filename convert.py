#!/usr/bin/env python3
"""
HDF5 to LeRobot Pipeline: Clean + Convert
LeRobot 数据集格式参考 Pi0.5 版本

This is a thin wrapper around the hdf5_to_lerobot library for backward compatibility.
For new code, consider using: python -m hdf5_to_lerobot
"""

import argparse
import os
import sys

# Add src to path for development mode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hdf5_to_lerobot.pipeline import run_pipeline


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
