# hdf5-to-lerobot

HDF5 to LeRobot Pipeline for Franka robot data. Clean, filter, and convert robot demonstration data to LeRobot format.

## Installation

```bash
git clone https://github.com/kowyo/hdf5-to-lerobot.git
cd hdf5-to-lerobot
make prepare
```

## Convert HDF5 to LeRobot Dataset v3.0

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

Make sure you have the Hugging Face Hub CLI installed and configured:

```bash
uv tool install hf
hf auth login
```

```bash
./scripts/convert_to_v30.sh \
    --config configs/default.json \
    --repo-id <user/dataset-name> \
    --push-to-hub
```
