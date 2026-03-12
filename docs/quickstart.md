# Quickstart

## Installation

```bash
git clone https://github.com/kowyo/hdf5-to-lerobot.git
cd hdf5-to-lerobot
make prepare
```

## Convert HDF5 to LeRobot v3.0

The `convert_to_v30.sh` script runs the full three-step pipeline:

| Step | Format | Tool |
|------|--------|------|
| 1 | HDF5 → v2.0 | `hdf5_to_lerobot` pipeline |
| 2 | v2.0 → v2.1 | `convert_dataset_v20_to_v21.py` |
| 3 | v2.1 → v3.0 | `lerobot.datasets.v30.convert_dataset_v21_to_v30` |

### Local conversion only

```bash
./scripts/convert_to_v30.sh \
    --config configs/default.json \
    --repo-id <user/dataset-name>
```

### Convert and publish to Hugging Face Hub

```bash
./scripts/convert_to_v30.sh \
    --config configs/default.json \
    --repo-id <user/dataset-name> \
    --push-to-hub
```

This will upload the v3.0 dataset and create a `v3.0` tag on the Hub. Intermediate v2.0/v2.1 data is removed automatically; `output_root` will contain only the final v3.0 dataset.

### Skip cleaning (data already cleaned)

```bash
./scripts/convert_to_v30.sh \
    --config configs/default.json \
    --repo-id <user/dataset-name> \
    --skip-cleaning
```

## Config file

The pipeline is driven by a JSON config. Copy and edit `configs/default.json`:

```json
{
  "output_root": "data/lerobot_dataset",
  "fps": 15,
  "image_size": 224,
  "chunk_size": 1000,
  "use_last": false,
  "write_stats": true,
  "keep_temp_cleaned": false,
  "cleaning": {
    "gripper_threshold": 0.0005,
    "pos_threshold": 0.0005,
    "rot_threshold": 0.005,
    "min_static_frames": 5,
    "min_episode_length": 20
  },
  "datasets": [
    {
      "path": "data/my-recordings",
      "task": "pick up the plug and insert it into the socket"
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `output_root` | Where the converted dataset is written |
| `fps` | Playback framerate |
| `image_size` | Images are resized to `image_size × image_size` |
| `chunk_size` | Max episodes per data chunk |
| `use_last` | Use last frame of front camera as `left_image` placeholder |
| `write_stats` | Write aggregate `stats.json` |
| `keep_temp_cleaned` | Keep intermediate cleaned HDF5 files |
| `cleaning.*` | Thresholds for filtering static/short episodes |
| `datasets` | List of `{path, task}` sources to combine |

Multiple datasets with different tasks can be listed under `datasets` and will be merged into a single LeRobot dataset.
