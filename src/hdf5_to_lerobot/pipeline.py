"""
Main pipeline for HDF5 to LeRobot conversion.

This module provides the high-level pipeline that orchestrates the cleaning
and conversion process.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from .cleaning import clean_hdf5_dataset
from .conversion import (
    aggregate_stats,
    convert_cleaned_dataset,
    write_info_json,
    write_stats_json,
    write_tasks_jsonl,
)


def run_pipeline(
    config_path: str, skip_cleaning: bool = False, skip_conversion: bool = False
) -> None:
    """运行完整的清洗+转换流程"""

    with Path(config_path).open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    print("\n" + "=" * 80)
    print("HDF5 to LeRobot Pipeline")
    print("=" * 80)
    print(f"Config: {config_path}")
    num_workers = int(config.get("num_workers", config.get("cleaning", {}).get("num_workers", 1)))
    print(f"Workers: {num_workers}")
    print(f"Skip cleaning: {skip_cleaning}")
    print(f"Skip conversion: {skip_conversion}")
    print("=" * 80 + "\n")

    # Extract parameters
    output_root = config["output_root"]
    fps = config.get("fps", 15)
    image_size = config.get("image_size", 224)
    chunk_size = config.get("chunk_size", 1000)
    use_last = config.get("use_last", False)
    write_stats = config.get("write_stats", True)
    datasets = config["datasets"]

    # TODO: Cleaning parameters
    cleaning_params = config.get(
        "cleaning",
        {
            "gripper_threshold": 0.0005,
            "pos_threshold": 0.0005,
            "rot_threshold": 0.005,
            "min_static_frames": 5,
            "min_episode_length": 20,
        },
    )

    temp_clean_root = config.get("temp_clean_dir", str(Path(output_root) / "_temp_cleaned"))
    cleaning_workers = num_workers
    conversion_workers = num_workers

    # Collect all unique tasks
    all_tasks: list[str] = []
    for ds in datasets:
        task = ds["task"]
        if task not in all_tasks:
            all_tasks.append(task)

    print(f"Unique tasks: {len(all_tasks)}")
    for i, task in enumerate(all_tasks):
        print(f"  Task {i}: {task}")
    print()

    output_p = Path(output_root)
    output_p.mkdir(parents=True, exist_ok=True)
    data_root = output_p / "data"
    meta_root = output_p / "meta"
    data_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    write_tasks_jsonl(str(meta_root), all_tasks)

    episodes_jsonl = meta_root / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()

    # Process each dataset
    total_episodes = 0
    total_frames = 0
    all_errors: list[str] = []
    all_episode_stats = []

    for ds_idx, ds_config in enumerate(datasets):
        input_path = ds_config["path"]
        task_text = ds_config["task"]
        task_index = all_tasks.index(task_text)

        print(f"\n{'#' * 80}")
        print(f"Dataset {ds_idx + 1}/{len(datasets)}")
        print(f"{'#' * 80}")
        print(f"Input: {input_path}")
        print(f"Task: {task_text}")
        print(f"{'#' * 80}\n")

        # Step 1: Clean data
        if not skip_cleaning:
            cleaned_path = str(Path(temp_clean_root) / f"dataset_{ds_idx}")
            num_cleaned, num_frames_cleaned, num_errors = clean_hdf5_dataset(
                input_path=input_path,
                output_path=cleaned_path,
                cleaning_params=cleaning_params,
                fps=fps,
                workers=cleaning_workers,
            )

            if num_cleaned == 0:
                print(f"[WARNING] No files successfully cleaned for dataset {ds_idx}")
                continue
        else:
            cleaned_path = input_path
            print(f"[INFO] Skipping cleaning, using raw data from: {input_path}")

        # Step 2: Convert to LeRobot format
        if not skip_conversion:
            num_episodes, num_frames, errors, episode_stats = convert_cleaned_dataset(
                cleaned_path=cleaned_path,
                output_root=output_root,
                task_text=task_text,
                task_index=task_index,
                episode_offset=total_episodes,
                fps=fps,
                image_size=image_size,
                chunk_size=chunk_size,
                use_last=use_last,
                workers=conversion_workers,
            )

            total_episodes += num_episodes
            total_frames += num_frames
            all_errors.extend(errors)
            all_episode_stats.extend(episode_stats)

    # Write info.json
    if not skip_conversion:
        write_info_json(
            str(meta_root),
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=len(all_tasks),
            fps=fps,
            chunk_size=chunk_size,
            image_size=image_size,
        )
        if write_stats:
            dataset_stats = aggregate_stats(all_episode_stats)
            write_stats_json(str(meta_root), dataset_stats)

    # Clean up temp directory
    if (
        not skip_cleaning
        and not config.get("keep_temp_cleaned", False)
        and Path(temp_clean_root).exists()
    ):
        print(f"\n[INFO] Cleaning up temporary directory: {temp_clean_root}")
        shutil.rmtree(temp_clean_root)

    # Final summary
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Output root: {output_root}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Errors: {len(all_errors)}")

    if all_errors:
        print(f"\n[!] {len(all_errors)} errors encountered:")
        for err in all_errors[:5]:
            print(f"  - {err}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")

    print("=" * 80 + "\n")
