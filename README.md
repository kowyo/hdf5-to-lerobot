# hdf5-to-lerobot

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)

HDF5 to LeRobot Pipeline for Franka robot data. Clean, filter, and convert robot demonstration data to LeRobot format (Pi0.5 compatible).

## Installation

```bash
git clone https://github.com/kowyo/hdf5-to-lerobot.git
cd hdf5-to-lerobot
make setup
```

## Usage

```bash
uv run python -m hdf5_to_lerobot --config /path/to/config.json
```
