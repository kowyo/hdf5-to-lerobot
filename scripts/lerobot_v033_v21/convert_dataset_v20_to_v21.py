# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.0 to
2.1. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --repo-id=aliberts/koch_tutorial
```

"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi

try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
    from lerobot.datasets.v21.convert_stats import check_aggregate_stats, convert_stats
except Exception:
    # Keep script usable in local-only mode without installing full lerobot package.
    CODEBASE_VERSION = "v2.1"
    LeRobotDataset = None
    EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
    STATS_PATH = "meta/stats.json"

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


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


def _compute_episode_stats(row_data: dict[str, list]) -> dict[str, dict[str, np.ndarray]]:
    ep_stats: dict[str, dict[str, np.ndarray]] = {}

    for key, values in row_data.items():
        if key in [
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.images.left_image",
        ]:
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
        ep_stats[key] = _get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)

    return ep_stats


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


def _aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    data_keys = {key for stats in stats_list for key in stats}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data_keys:
        out[key] = _aggregate_feature_stats([stats[key] for stats in stats_list if key in stats])
    return out


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _read_stats(path: Path) -> dict[str, dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)

    def _to_np(x):
        if isinstance(x, dict):
            return {k: _to_np(v) for k, v in x.items()}
        return np.array(x)

    return _to_np(data)


def convert_dataset_local(local_root: str, num_workers: int = 4):
    root = Path(local_root)
    meta_root = root / "meta"
    data_root = root / "data"

    episodes_stats_path = root / EPISODES_STATS_PATH
    if episodes_stats_path.exists():
        episodes_stats_path.unlink()

    parquet_files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {data_root}")

    all_episode_stats = []
    with episodes_stats_path.open("w", encoding="utf-8") as out_f:
        for ep_idx, parquet_path in enumerate(parquet_files):
            ds = Dataset.from_parquet(str(parquet_path))
            ep_data = {col: ds[col] for col in ds.column_names}
            ep_stats = _compute_episode_stats(ep_data)
            all_episode_stats.append(ep_stats)
            out_f.write(json.dumps({"episode_index": ep_idx, "stats": _serialize(ep_stats)}) + "\n")

    ref_stats = _read_stats(root / STATS_PATH)
    if ref_stats is not None:
        agg = _aggregate_stats(all_episode_stats)
        for key, stats in agg.items():
            for stat_name, val in stats.items():
                if key in ref_stats and stat_name in ref_stats[key]:
                    np.testing.assert_allclose(
                        val,
                        ref_stats[key][stat_name],
                        rtol=5e-6,
                        atol=6e-5,
                        err_msg=f"feature='{key}' stats='{stat_name}'",
                    )

    info_path = meta_root / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    info["codebase_version"] = CODEBASE_VERSION
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    old_stats = root / STATS_PATH
    if old_stats.exists():
        old_stats.unlink()

    print(f"Converted local dataset at {root} from {V20} to {CODEBASE_VERSION}")
    print(f"Wrote {episodes_stats_path}")
    print(f"Updated {info_path}")


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    if LeRobotDataset is None:
        raise RuntimeError(
            "lerobot package is not installed. Use --local-root for local migration mode, "
            "or install lerobot==0.3.3 to run hub mode."
        )

    with SuppressWarnings():
        dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True)

    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if (dataset.root / STATS_PATH).is_file:
        (dataset.root / STATS_PATH).unlink()

    hub_api = HfApi()
    if hub_api.file_exists(
        repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(
            path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )

    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--local-root",
        type=str,
        default=None,
        help="Local dataset root containing data/ and meta/ to migrate in-place.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()
    if args.local_root:
        convert_dataset_local(local_root=args.local_root, num_workers=args.num_workers)
    elif args.repo_id:
        convert_dataset(repo_id=args.repo_id, branch=args.branch, num_workers=args.num_workers)
    else:
        raise ValueError(
            "You must provide either --repo-id (hub mode) or --local-root (local mode)."
        )
