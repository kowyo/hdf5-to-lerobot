"""
LeRobot format conversion functions.

This module provides functions to convert cleaned HDF5 data to the LeRobot format
used by Pi0.5 and similar robot learning frameworks.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm


def _estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def _sample_indices(data_len: int) -> list[int]:
    num_samples = _estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _get_feature_stats(
    array: np.ndarray, axis: int | tuple[int, ...], keepdims: bool
) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(data: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-episode stats in LeRobot v2.0-compatible shape conventions."""
    ep_stats: dict[str, dict[str, np.ndarray]] = {}

    for key, values in data.items():
        if key in [
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.images.left_image",
        ]:
            # Sample image frames and convert to (N, C, H, W) float32 in [0,1].
            sampled = [values[i] for i in _sample_indices(len(values))]
            arr = np.stack([np.asarray(img, dtype=np.uint8) for img in sampled], axis=0)
            arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32) / 255.0
            stats = _get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in stats.items()
            }
            continue

        arr = np.asarray(values)
        if arr.dtype.kind in {"U", "S", "O"}:
            continue

        reduce_axis = 0
        keepdims = arr.ndim == 1
        ep_stats[key] = _get_feature_stats(arr, axis=reduce_axis, keepdims=keepdims)

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> None:
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "observation.images" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def _aggregate_feature_stats(stats_ft_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * counts).sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats."""
    if not stats_list:
        return {}

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats: dict[str, dict[str, np.ndarray]] = {}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = _aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def write_stats_json(meta_dir: str, stats: dict[str, dict[str, np.ndarray]]) -> None:
    """Write legacy v2.0 aggregate stats file (meta/stats.json)."""

    def _to_python(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    meta_p = Path(meta_dir)
    meta_p.mkdir(parents=True, exist_ok=True)
    with (meta_p / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(_to_python(stats), f, ensure_ascii=False, indent=2)


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


def _resolve_worker_count(requested_workers: int, total_jobs: int) -> int:
    if total_jobs <= 0:
        return 1
    if requested_workers <= 0:
        requested_workers = os.process_cpu_count() or 1
    return max(1, min(requested_workers, total_jobs))


def _resolve_resize_workers(episode_workers: int) -> int:
    total_cpus = os.process_cpu_count() or 1
    if episode_workers <= 1:
        return min(8, total_cpus)
    return max(1, min(8, total_cpus // episode_workers))


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
    resize_workers: int = 1,
    frame_offset: int = 0,
) -> dict:
    """Build episode data as dict for HuggingFace datasets in Pi0.5 format."""

    num_frames = len(ee_pose)
    states, actions = compute_actions_from_ee_pose(ee_pose)

    if timestamps is None:
        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    else:
        timestamps = (timestamps - timestamps[0]).astype(np.float32)

    resize_workers = max(1, resize_workers)

    def resize_batch(imgs, size, crop_params=None):
        def resize_single(img):
            if crop_params:
                x, y, c_size = crop_params
                img = img[y : y + c_size, x : x + c_size]

            # 执行缩放
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            return Image.fromarray(resized)

        with ThreadPoolExecutor(max_workers=resize_workers) as executor:
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
        "observation.images.image": left_resized,
        "observation.images.wrist_image": wrist_resized,
        "observation.images.left_image": front_resized,
        "observation.state": states.tolist(),
        "action": actions.tolist(),
        "timestamp": timestamps.tolist(),
        "frame_index": np.arange(num_frames, dtype=np.int64).tolist(),
        "episode_index": np.full(num_frames, episode_index, dtype=np.int64).tolist(),
        "index": (frame_offset + np.arange(num_frames, dtype=np.int64)).tolist(),
        "task_index": np.full(num_frames, task_index, dtype=np.int64).tolist(),
    }

    return data


def _get_hdf5_frame_count(hdf5_path: str) -> int:
    """Quickly read the frame count from an HDF5 file without decoding pixels."""
    with h5py.File(hdf5_path, "r") as f:
        return len(f["observations"]["ee_pose"])


def _convert_episode_task(
    job: tuple[str, str, int, int, str, float, int, int, int, bool, int, int],
) -> tuple[int, int, str | None, dict[str, dict[str, np.ndarray]] | None]:
    (
        h5p,
        output_root,
        episode_offset,
        local_idx,
        file_name,
        fps,
        image_size,
        task_index,
        chunk_size,
        use_last,
        resize_workers,
        frame_offset,
    ) = job

    ep_idx = episode_offset + local_idx

    try:
        output_p = Path(output_root)
        chunk_id = ep_idx // chunk_size
        chunk_dir = output_p / "data" / f"chunk-{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

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
            resize_workers=resize_workers,
            frame_offset=frame_offset,
        )

        parquet_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        save_episode_with_datasets(data, str(parquet_path))

        episode_length = len(data["timestamp"])
        return ep_idx, episode_length, None, compute_episode_stats(data)
    except Exception as e:
        return ep_idx, 0, f"Episode {ep_idx} ({file_name}): {str(e)}", None


def save_episode_with_datasets(data: dict, out_path: str) -> None:
    """Save episode using HuggingFace datasets library."""
    features = Features(
        {
            "observation.images.image": ImageFeature(),
            "observation.images.wrist_image": ImageFeature(),
            "observation.images.left_image": ImageFeature(),
            "observation.state": Sequence(Value("float32"), length=8),
            "action": Sequence(Value("float32"), length=7),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
    )

    dataset = Dataset.from_dict(data, features=features)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path)


def write_tasks_jsonl(meta_dir: str, all_tasks: list[str]) -> None:
    """Write tasks to JSONL file."""
    meta_p = Path(meta_dir)
    meta_p.mkdir(parents=True, exist_ok=True)
    tasks_path = meta_p / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for task_idx, task_text in enumerate(all_tasks):
            f.write(
                json.dumps({"task_index": task_idx, "task": task_text}, ensure_ascii=False) + "\n"
            )


def append_episode_meta(meta_dir: str, episode_index: int, length: int, task_text: str) -> None:
    """Append episode metadata to JSONL file."""
    meta_p = Path(meta_dir)
    meta_p.mkdir(parents=True, exist_ok=True)
    episodes_path = meta_p / "episodes.jsonl"
    rec = {"episode_index": episode_index, "tasks": [task_text], "length": length}
    with episodes_path.open("a", encoding="utf-8") as f:
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
            "observation.images.image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_image": {
                "dtype": "image",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"],
            },
            "action": {
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

    meta_p = Path(meta_dir)
    meta_p.mkdir(parents=True, exist_ok=True)
    with (meta_p / "info.json").open("w", encoding="utf-8") as f:
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
    workers: int = 1,
) -> tuple[int, int, list[str], list[dict[str, dict[str, np.ndarray]]]]:
    """
    Convert a single cleaned dataset to LeRobot format.
    Returns: (num_episodes, num_frames, errors, episode_stats)
    """

    cleaned_p = Path(cleaned_path)
    if cleaned_p.is_file():
        files = [cleaned_path]
    else:
        files = sorted(list(cleaned_p.rglob("*.h5")) + list(cleaned_p.rglob("*.hdf5")))

    if not files:
        print(f"[WARNING] No cleaned HDF5 files found in: {cleaned_path}")
        return 0, 0, [], []

    print(f"\n{'=' * 80}")
    print(f"Converting Dataset: {cleaned_path}")
    print(f"{'=' * 80}")
    print(f"Task: {task_text}")
    print(f"Files: {len(files)}")
    workers = _resolve_worker_count(workers, len(files))
    resize_workers = _resolve_resize_workers(workers)
    print(f"Episode workers: {workers}")
    print(f"Resize workers per episode: {resize_workers}")
    print(f"{'=' * 80}\n")

    meta_root = Path(output_root) / "meta"

    total_frames = 0
    errors: list[str] = []
    episode_stats: list[dict[str, dict[str, np.ndarray]]] = []

    # Pre-scan HDF5 files to compute cumulative frame offsets for continuous indexing.
    frame_counts = [_get_hdf5_frame_count(str(h5p)) for h5p in files]
    cumulative_offsets = []
    running_total = 0
    for count in frame_counts:
        cumulative_offsets.append(running_total)
        running_total += count

    jobs = [
        (
            str(h5p),
            output_root,
            episode_offset,
            local_idx,
            Path(h5p).name,
            fps,
            image_size,
            task_index,
            chunk_size,
            use_last,
            resize_workers,
            cumulative_offsets[local_idx],
        )
        for local_idx, h5p in enumerate(files)
    ]

    if workers == 1:
        results = (_convert_episode_task(job) for job in jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_convert_episode_task, jobs)

    try:
        for ep_idx, episode_length, error_msg, stats in tqdm(
            results, total=len(jobs), desc="Converting"
        ):
            if error_msg is not None:
                errors.append(error_msg)
                print(f"\n  [✗] ERROR: {error_msg}")
                continue

            append_episode_meta(str(meta_root), ep_idx, length=episode_length, task_text=task_text)
            total_frames += episode_length
            if stats is not None:
                episode_stats.append(stats)
    finally:
        if workers > 1:
            executor.shutdown()

    return len(files) - len(errors), total_frames, errors, episode_stats
