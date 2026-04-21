---
license: mit
tags:
  - zebra-finch
  - camera-calibration
  - birdpark
---

# BirdPark Camera Calibration & Metadata

Camera intrinsics, extrinsics, and bird-color metadata for the BirdPark multi-view tracking setup (3 cameras: top, back, side).

## Download

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download
snapshot_download("songbirdini/birdpark_calibration", local_dir="tracking/data")
```

Or from the TCBI analysis project:

```bash
python -m tracking.download_calibration
```

## Contents

| File | Description |
|------|-------------|
| `calibration_{view}.npz` | Camera intrinsic matrix (K) + distortion coefficients |
| `camera_pose_{view}.pkl` | Camera extrinsic rotation (rvec) + translation (tvec) |
| `metadata.csv` | Experiment -> bird ID -> backpack color mapping |

Views: `top` (1292x1292), `back` (2202x724), `side` (774x1292, rotated 90 degrees)
