"""
LeRobot format conversion functions.

This module provides functions to convert cleaned HDF5 data to the LeRobot format
used by Pi0.5 and similar robot learning frameworks.
"""

import glob
import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    """Wrap angle delta to [-π, π]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def get_euler_from_pose(pose_row: np.ndarray) -> np.ndarray:
    """
    根据维度自动转换旋转表示为欧拉角。
    Args:
        pose_row: 7位 [x,y,z,rx,ry,rz,g] 或 8位 [x,y,z,qx,qy,qz,qw,g]
    """
    dim = len(pose_row)
    if dim == 7:
        return Rot.from_rotvec(pose_row[3:6]).as_euler("xyz", degrees=False)
    elif dim == 8:
        return Rot.from_quat(pose_row[3:7]).as_euler("xyz", degrees=False)
    else:
        raise ValueError(f"Unsupported ee_pose dimension: {dim}")


def compute_actions_from_ee_pose(ee_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ee_pose trajectory to LeRobot format, auto-detecting 7D or 8D input.
    """
    num_frames = len(ee_pose)
    dim = ee_pose.shape[1]

    # 1. 自动提取旋转并转为欧拉角
    euler_angles = np.array([get_euler_from_pose(ee_pose[i]) for i in range(num_frames)])

    # 2. 确定夹爪索引
    gripper_idx = 6 if dim == 7 else 7

    # 3. Build state (8D)
    states = np.zeros((num_frames, 8), dtype=np.float32)
    states[:, :3] = ee_pose[:, :3]
    states[:, 3:6] = euler_angles
    states[:, 6] = ee_pose[:, gripper_idx] / 2
    states[:, 7] = -ee_pose[:, gripper_idx] / 2

    # 4. Build actions (7D)
    actions = np.zeros((num_frames, 7), dtype=np.float32)

    for t in range(num_frames - 1):
        actions[t, :3] = ee_pose[t + 1, :3] - ee_pose[t, :3]
        delta_rot = euler_angles[t + 1] - euler_angles[t]
        actions[t, 3:6] = wrap_angle_delta(delta_rot)
        actions[t, 6] = 1.0 if ee_pose[t + 1, gripper_idx] >= 0.07 else 0.0

    if num_frames > 1:
        actions[-1] = actions[-2]

    return states, actions


def load_hdf5(
    hdf5_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load HDF5 file with flexible structure support."""
    with h5py.File(hdf5_path, "r") as f:
        if "observations" not in f or "ee_pose" not in f["observations"]:
            raise KeyError(f"Cannot find observations/ee_pose in {hdf5_path}")
        ee_pose = f["observations"]["ee_pose"][:]

        # Load images
        front_imgs = None
        wrist_imgs = None
        left_imgs = None

        if "observations" in f and "images" in f["observations"]:
            imgs = f["observations"]["images"]
            if "camera_front_color" in imgs:
                front_imgs = imgs["camera_front_color"][:]
            if "camera_wrist_color" in imgs:
                wrist_imgs = imgs["camera_wrist_color"][:]
            if "camera_left_color" in imgs:
                left_imgs = imgs["camera_left_color"][:]

        if front_imgs is None:
            raise KeyError(f"Cannot find camera_front_color in {hdf5_path}")
        if wrist_imgs is None:
            raise KeyError(f"Cannot find camera_wrist_color in {hdf5_path}")
        if left_imgs is None:
            raise KeyError(f"Cannot find camera_left_color in {hdf5_path}")

        # Load timestamps
        timestamps = None
        if "timestamp" in f:
            timestamps = f["timestamp"][:]
        elif "observations" in f and "timestamp" in f["observations"]:
            timestamps = f["observations"]["timestamp"][:]

    # Truncate to minimum length
    num_frames = min(len(ee_pose), len(front_imgs), len(wrist_imgs), len(left_imgs))
    if timestamps is not None:
        num_frames = min(num_frames, len(timestamps))

    return (
        ee_pose[:num_frames],
        front_imgs[:num_frames],
        wrist_imgs[:num_frames],
        left_imgs[:num_frames],
        timestamps[:num_frames] if timestamps is not None else None,
    )


def build_episode_data(
    ee_pose: np.ndarray,
    front_imgs: np.ndarray,
    wrist_imgs: np.ndarray,
    left_imgs: np.ndarray,
    timestamps: np.ndarray | None,
    episode_index: int,
    fps: float,
    image_size: int,
    task_index: int = 0,
    use_last: bool = False,
) -> dict:
    """Build episode data as dict for HuggingFace datasets in Pi0.5 format."""

    num_frames = len(ee_pose)
    states, actions = compute_actions_from_ee_pose(ee_pose)

    if timestamps is None:
        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    else:
        timestamps = (timestamps - timestamps[0]).astype(np.float32)

    # ========== OPTIMIZED: Parallel batch resize ==========
    num_workers = min(8, mp.cpu_count())  # 自动适配CPU核心数

    def resize_batch(imgs, size, crop_params=None):
        def resize_single(img):
            if crop_params:
                x, y, c_size = crop_params
                img = img[y : y + c_size, x : x + c_size]

            # 执行缩放
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            return Image.fromarray(resized)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(resize_single, imgs))

    # TODO: 不进行剪裁
    crop_config = {"front": None, "wrist": None, "left": None}

    if not use_last:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "front": executor.submit(
                    resize_batch, front_imgs, image_size, crop_config["front"]
                ),
                "wrist": executor.submit(
                    resize_batch, wrist_imgs, image_size, crop_config["wrist"]
                ),
                "left": executor.submit(resize_batch, left_imgs, image_size, crop_config["left"]),
            }
            front_resized = futures["front"].result()
            wrist_resized = futures["wrist"].result()
            left_resized = futures["left"].result()

    else:
        last_resized = cv2.resize(
            front_imgs[-1], (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        last_pil = Image.fromarray(last_resized)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                "front": executor.submit(resize_batch, front_imgs, image_size),
                "wrist": executor.submit(resize_batch, wrist_imgs, image_size),
            }
            front_resized = futures["front"].result()
            wrist_resized = futures["wrist"].result()

        left_resized = [last_pil] * num_frames

    data = {
        "image": front_resized,
        "wrist_image": wrist_resized,
        "left_image": left_resized,
        "state": states.tolist(),
        "actions": actions.tolist(),
        "timestamp": timestamps.tolist(),
        "frame_index": np.arange(num_frames, dtype=np.int64).tolist(),
        "episode_index": np.full(num_frames, episode_index, dtype=np.int64).tolist(),
        "index": np.arange(num_frames, dtype=np.int64).tolist(),
        "task_index": np.full(num_frames, task_index, dtype=np.int64).tolist(),
    }

    return data


def save_episode_with_datasets(data: dict, out_path: str) -> None:
    """Save episode using HuggingFace datasets library."""
    features = Features(
        {
            "image": ImageFeature(),
            "wrist_image": ImageFeature(),
            "left_image": ImageFeature(),
            "state": Sequence(Value("float32"), length=8),
            "actions": Sequence(Value("float32"), length=7),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
    )

    dataset = Dataset.from_dict(data, features=features)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_parquet(out_path)


def write_tasks_jsonl(meta_dir: str, all_tasks: list[str]) -> None:
    """Write tasks to JSONL file."""
    os.makedirs(meta_dir, exist_ok=True)
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task_idx, task_text in enumerate(all_tasks):
            f.write(
                json.dumps({"task_index": task_idx, "task": task_text}, ensure_ascii=False) + "\n"
            )


def append_episode_meta(meta_dir: str, episode_index: int, length: int, task_text: str) -> None:
    """Append episode metadata to JSONL file."""
    os.makedirs(meta_dir, exist_ok=True)
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    rec = {"episode_index": episode_index, "tasks": [task_text], "length": length}
    with open(episodes_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_info_json(
    meta_dir: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: float,
    chunk_size: int,
    image_size: int,
) -> None:
    """Write dataset info.json file."""
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": 0,
        "total_chunks": int((total_episodes + chunk_size - 1) // chunk_size),
        "chunks_size": int(chunk_size),
        "fps": float(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "left_image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_roll",
                    "delta_pitch",
                    "delta_yaw",
                    "gripper",
                ],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def convert_cleaned_dataset(
    cleaned_path: str,
    output_root: str,
    task_text: str,
    task_index: int,
    episode_offset: int,
    fps: float,
    image_size: int,
    chunk_size: int,
    use_last: bool,
) -> tuple[int, int, list[str]]:
    """
    Convert a single cleaned dataset to LeRobot format.
    Returns: (num_episodes, num_frames, errors)
    """

    # Find all cleaned HDF5 files
    if os.path.isfile(cleaned_path):
        files = [cleaned_path]
    else:
        patterns = ["**/*.h5", "**/*.hdf5"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(cleaned_path, p), recursive=True))
        files = sorted(files)

    if not files:
        print(f"[WARNING] No cleaned HDF5 files found in: {cleaned_path}")
        return 0, 0, []

    print(f"\n{'=' * 80}")
    print(f"Converting Dataset: {cleaned_path}")
    print(f"{'=' * 80}")
    print(f"Task: {task_text}")
    print(f"Files: {len(files)}")
    print(f"{'=' * 80}\n")

    data_root = os.path.join(output_root, "data")
    meta_root = os.path.join(output_root, "meta")

    total_frames = 0
    errors = []

    for local_idx, h5p in enumerate(tqdm(files, desc="Converting")):
        try:
            ep_idx = episode_offset + local_idx
            chunk_id = ep_idx // chunk_size
            chunk_dir = os.path.join(data_root, f"chunk-{chunk_id:03d}")
            os.makedirs(chunk_dir, exist_ok=True)

            # Load data
            ee_pose, front_imgs, wrist_imgs, left_imgs, timestamps = load_hdf5(h5p)

            data = build_episode_data(
                ee_pose=ee_pose,
                front_imgs=front_imgs,
                wrist_imgs=wrist_imgs,
                left_imgs=left_imgs,
                timestamps=timestamps,
                episode_index=ep_idx,
                fps=fps,
                image_size=image_size,
                task_index=task_index,
                use_last=use_last,
            )

            episode_length = len(data["timestamp"])

            # Save parquet
            parquet_path = os.path.join(chunk_dir, f"episode_{ep_idx:06d}.parquet")
            save_episode_with_datasets(data, parquet_path)

            # Append to episodes.jsonl
            append_episode_meta(meta_root, ep_idx, length=episode_length, task_text=task_text)

            total_frames += episode_length

        except Exception as e:
            error_msg = f"Episode {ep_idx} ({os.path.basename(h5p)}): {str(e)}"
            errors.append(error_msg)
            print(f"\n  [✗] ERROR: {error_msg}")

    return len(files) - len(errors), total_frames, errors
