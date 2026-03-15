# hdf5-to-lerobot

HDF5 to LeRobot Pipeline for Franka robot data. Clean, filter, and convert robot demonstration data to LeRobot format.

## Installation

```bash
git clone https://github.com/kowyo/hdf5-to-lerobot.git
cd hdf5-to-lerobot
make prepare
```

## Usage

Converts HDF5 robot data to LeRobot v3.0 format and publishes it to the Hugging Face Hub.

```bash
uv tool install hf
hf auth login
make convert REPO=<user/dataset-name>
```
