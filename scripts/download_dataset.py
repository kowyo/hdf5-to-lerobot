#!/usr/bin/env python3
"""
Download a Hugging Face dataset for hdf5-to-lerobot.

Usage:
    uv run python scripts/download_dataset.py <repo_id>
    uv run python scripts/download_dataset.py <repo_id> --output-dir ./data/out
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset for hdf5-to-lerobot"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face dataset ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save the dataset (default: ./data/<dataset_name>)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        dataset_name = args.repo_id.split("/")[-1]
        output_dir = Path("data") / dataset_name
    else:
        output_dir = Path(output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {output_dir}...")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )

    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
