"""
HDF5 to LeRobot Pipeline Library

A library for cleaning HDF5 robot demonstration data and converting it
to the LeRobot format used by Pi0.5 and similar frameworks.
"""

__version__ = "1.0.0"

from .cleaning import (
    analyze_episode_motion,
    clean_hdf5_dataset,
    detect_static_segments_advanced,
    filter_hdf5_file,
    parse_ee_pose,
)
from .conversion import (
    append_episode_meta,
    build_episode_data,
    compute_actions_from_ee_pose,
    convert_cleaned_dataset,
    get_euler_from_pose,
    load_hdf5,
    save_episode_with_datasets,
    wrap_angle_delta,
    write_info_json,
    write_tasks_jsonl,
)
from .pipeline import run_pipeline

__all__ = [
    # Cleaning
    "parse_ee_pose",
    "analyze_episode_motion",
    "detect_static_segments_advanced",
    "filter_hdf5_file",
    "clean_hdf5_dataset",
    # Conversion
    "wrap_angle_delta",
    "get_euler_from_pose",
    "compute_actions_from_ee_pose",
    "load_hdf5",
    "build_episode_data",
    "save_episode_with_datasets",
    "write_tasks_jsonl",
    "append_episode_meta",
    "write_info_json",
    "convert_cleaned_dataset",
    # Pipeline
    "run_pipeline",
]
