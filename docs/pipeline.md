# Pipeline Details

## Step 1 — HDF5 → v2.0

**Entry point:** `uv run python -m hdf5_to_lerobot --config <config>`

Each HDF5 file is expected to contain:

```
observations/
  ee_pose          # (T, 7) or (T, 8) — [x, y, z, rx, ry, rz, gripper] or [..., qw]
  images/
    camera_front_color   # (T, H, W, 3)
    camera_wrist_color   # (T, H, W, 3)
    camera_left_color    # (T, H, W, 3)
timestamp          # (T,) optional
```

The pipeline:
1. **Cleans** each file — removes static start/end frames and episodes that are too short.
2. **Converts** ee_pose to `state` (8D) and `actions` (7D delta format).
3. Writes one `episode_XXXXXX.parquet` per episode into `data/chunk-NNN/`.
4. Writes `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/stats.json`.

Output `codebase_version`: `v2.0`

## Step 2 — v2.0 → v2.1

**Entry point:** `uv run python scripts/lerobot_v033_v21/convert_dataset_v20_to_v21.py --local-root <output_root>`

- Computes per-episode stats and writes `meta/episodes_stats.jsonl`.
- Validates them against the existing aggregate `meta/stats.json`.
- Removes `meta/stats.json`.
- Bumps `codebase_version` to `v2.1` in `meta/info.json`.

## Step 3 — v2.1 → v3.0

**Entry point:** `uv run python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <id> --root <parent>`

- Merges per-episode parquet files into larger files (≤ 100 MB each).
- Converts metadata to parquet format under `meta/episodes/` and `meta/tasks.parquet`.
- Bumps `codebase_version` to `v3.0`.
- Optionally pushes to the Hugging Face Hub and creates a `v3.0` tag.

## State and action format

| Key | Shape | Description |
|-----|-------|-------------|
| `state` | `(8,)` | `[x, y, z, roll, pitch, yaw, gripper/2, -gripper/2]` |
| `actions` | `(7,)` | `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_binary]` |
| `image` | `(H, W, 3)` | Front camera |
| `wrist_image` | `(H, W, 3)` | Wrist camera |
| `left_image` | `(H, W, 3)` | Left camera (or last front frame if `use_last=true`) |

Rotations are converted from rotation-vector or quaternion to Euler angles (XYZ, radians). Gripper action is binarised: `1.0` if next-step gripper ≥ 0.07, else `0.0`.
