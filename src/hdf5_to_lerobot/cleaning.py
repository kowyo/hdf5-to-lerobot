"""
Data cleaning functions for HDF5 datasets.

This module provides functions to clean and filter HDF5 files by detecting
and removing static segments from robot demonstration data.
"""

import os
import glob
from typing import Dict, Tuple
from tqdm import tqdm
import h5py
import numpy as np


def parse_ee_pose(ee_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    自动识别ee_pose维度并提取组件
    Returns: pos (T,3), rot (T,3 or 4), gripper (T,)
    """
    dim = ee_pose.shape[1]
    pos = ee_pose[:, :3]
    if dim == 7:
        # [x, y, z, rx, ry, rz, gripper]
        return pos, ee_pose[:, 3:6], ee_pose[:, 6]
    else:
        # [x, y, z, qx, qy, qz, qw, gripper]
        return pos, ee_pose[:, 3:7], ee_pose[:, 7]


def analyze_episode_motion(ee_pose: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Analyze motion in an episode using sliding window variance."""
    T = len(ee_pose)
    motion_score = np.zeros(T)
    pos, rot, gripper = parse_ee_pose(ee_pose)

    for i in range(T):
        start, end = max(0, i - window_size), min(T, i + window_size + 1)
        pos_var = np.var(pos[start:end], axis=0).sum()
        rot_var = np.var(rot[start:end], axis=0).sum()
        gripper_var = np.var(gripper[start:end])
        motion_score[i] = pos_var + rot_var + gripper_var * 10
    return motion_score


def detect_static_segments_advanced(
    ee_pose: np.ndarray,
    gripper_threshold: float = 0.0005,
    pos_threshold: float = 0.001,
    rot_threshold: float = 0.01,
    min_static_frames: int = 10,
    motion_score_threshold: float = 0.0001,
) -> np.ndarray:
    """
    Detect and mark static segments in the episode data.
    
    Returns a boolean mask where True means the frame should be kept.
    """
    T = len(ee_pose)
    if T == 0:
        return np.array([], dtype=bool)

    motion_score = analyze_episode_motion(ee_pose)
    pos, rot, gripper = parse_ee_pose(ee_pose)

    pos_delta = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    rot_delta = np.linalg.norm(np.diff(rot, axis=0), axis=1)
    gripper_delta = np.abs(np.diff(gripper, axis=0))

    is_static = np.concatenate(
        [[False], (pos_delta < pos_threshold) & (rot_delta < rot_threshold) & (gripper_delta < gripper_threshold)]
    )
    low_motion = motion_score < motion_score_threshold
    is_abnormal = is_static & low_motion

    mask = np.ones(T, dtype=bool)
    i = 0
    while i < T:
        if is_abnormal[i]:
            j = i
            while j < T and is_abnormal[j]:
                j += 1
            if (j - i) >= min_static_frames:
                if not (
                    (i > 0 and motion_score[i - 1] > motion_score_threshold * 2)
                    and (j < T and motion_score[j] > motion_score_threshold * 2)
                ):
                    mask[i:j] = False
            i = j
        else:
            i += 1
    return mask


def filter_hdf5_file(
    input_path: str,
    output_path: str,
    cleaning_params: Dict,
    fps: float = 10.0,
) -> Tuple[bool, int, int]:
    """过滤单个HDF5文件"""
    try:
        with h5py.File(input_path, "r") as f_in:
            ee_pose = f_in["observations"]["ee_pose"][:]
            original_length = len(ee_pose)

            # 检测并生成mask
            mask = detect_static_segments_advanced(
                ee_pose,
                gripper_threshold=cleaning_params["gripper_threshold"],
                pos_threshold=cleaning_params["pos_threshold"],
                rot_threshold=cleaning_params["rot_threshold"],
                min_static_frames=cleaning_params["min_static_frames"],
            )

            filtered_length = np.sum(mask)

            if filtered_length < cleaning_params["min_episode_length"]:
                return False, original_length, 0

            # 创建输出文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f_out:
                # 1. 重新生成连续的timestamp
                new_timestamp = np.arange(filtered_length, dtype=np.float32) / fps
                f_out.create_dataset("timestamp", data=new_timestamp)

                # 2. 过滤其他top-level数据
                for key in ["stage", "joint_action"]:
                    if key in f_in:
                        data = f_in[key][:]
                        f_out.create_dataset(key, data=data[mask])

                # 3. 过滤observations组
                obs_group = f_out.create_group("observations")
                f_in_obs = f_in["observations"]

                for key in f_in_obs.keys():
                    if key == "images":
                        img_group = obs_group.create_group("images")
                        for img_key in f_in_obs["images"].keys():
                            img_data = f_in_obs["images"][img_key][:]
                            img_group.create_dataset(img_key, data=img_data[mask])
                    elif key == "robot_base_pose_in_world":
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])
                    else:
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])

            return True, original_length, filtered_length

    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return False, 0, 0


def clean_hdf5_dataset(input_path: str, output_path: str, cleaning_params: Dict, fps: float) -> Tuple[int, int, int]:
    """清洗单个数据集的所有HDF5文件"""

    # 查找所有HDF5文件
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        patterns = ["**/*.h5", "**/*.hdf5"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(input_path, p), recursive=True))
        files = sorted(files)

    if not files:
        print(f"[WARNING] No HDF5 files found in: {input_path}")
        return 0, 0, 0

    print(f"\n{'='*80}")
    print(f"Cleaning Dataset: {input_path}")
    print(f"{'='*80}")
    print(f"Found {len(files)} HDF5 files")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    stats = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "original_frames": 0,
        "filtered_frames": 0,
    }

    for i, file_path in enumerate(tqdm(files, desc="Cleaning")):
        rel_path = os.path.relpath(file_path, input_path)
        out_file = os.path.join(output_path, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        success, orig_len, filt_len = filter_hdf5_file(
            file_path,
            out_file,
            cleaning_params,
            fps=fps,
        )

        if success:
            if filt_len > 0:
                stats["success"] += 1
                stats["original_frames"] += orig_len
                stats["filtered_frames"] += filt_len
            else:
                stats["skipped"] += 1
        else:
            stats["error"] += 1

    print(f"\n{'='*80}")
    print("Cleaning Summary")
    print(f"{'='*80}")
    print(f"Successfully cleaned: {stats['success']}")
    print(f"Skipped (too short):  {stats['skipped']}")
    print(f"Errors:               {stats['error']}")
    print(f"Original frames:      {stats['original_frames']}")
    print(f"Filtered frames:      {stats['filtered_frames']}")
    if stats["original_frames"] > 0:
        kept_ratio = stats["filtered_frames"] / stats["original_frames"] * 100
        print(f"Kept ratio:           {kept_ratio:.1f}%")
    print(f"{'='*80}\n")

    return stats["success"], stats["filtered_frames"], stats["error"]
