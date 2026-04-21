---
license: cc-by-4.0
pretty_name: V33DA
task_categories:
- audio-classification
- video-classification
tags:
- animal-behavior
- bioacoustics
- bird-vocalization
- multimodal
- 3d-pose
- localization
- attribution
- radio-telemetry
- accelerometer
- zebra-finch
- benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: v33da-*.parquet
---

# V33DA: A Multi-Modal Benchmark for Vocal Attribution in Freely Interacting Zebra Finches

**Task.** Given a detected zebra finch vocalization and the set of birds visible at that moment, determine which bird produced the call.

V33DA provides 33,625 vocalization events from 10 individually identified zebra finches across 3 experiments (2021--2023), each with synchronized 5-channel audio, multi-view video, 3D pose, FM radio telemetry, and accelerometer-derived ground-truth labels.

| | BP01 | BP02 | BP05 | Total |
|---|---|---|---|---|
| Birds | 4 | 3 | 3 | 10 |
| Events | 16,219 | 11,577 | 5,829 | 33,625 |
| Recording days | 10 | 3 | 3 | 16 |
| Raw hours | ~46h | ~14h | ~14h | ~74h |

**Code:** [github.com/marisbasha/v33da](https://github.com/marisbasha/v33da)

## Quick start

```bash
pip install huggingface_hub
huggingface-cli download songbirdini/V33DA --repo-type dataset --local-dir data/v33da
```

Or in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("songbirdini/V33DA", repo_type="dataset", local_dir="data/v33da")
```

### Loading a sample

Each binary array column is serialized with `numpy.save`. To decode:

```python
import io
import numpy as np
import pyarrow.parquet as pq

table = pq.read_table("data/v33da/v33da-00000.parquet")
row = table.to_pydict()

# 3D keypoints: (N_birds, 5, 3) float64
kp3d = np.load(io.BytesIO(row["keypoints_3d"][0]))

# Audio path -> multichannel WAV
audio_path = row["audio_path"][0]  # e.g. "audio/juvExpBP01/2021-06-28/..."
```

The included `explore.ipynb` notebook shows how to inspect audio, video, pose, radio, and accelerometer data, and how to reproject 3D keypoints into 2D camera views using the shipped calibrations.

## Modalities

Each parquet row is one vocalization event:

| Column | Description |
|---|---|
| `audio_path` | 5-channel cage-microphone WAV (24,414 Hz, float32) |
| `accelerometer_path` | Per-bird accelerometer WAV (oracle verification signal) |
| `video_path` | Aligned composite MP4 clip (3 views, 47.68 fps) |
| `keypoints_3d` | Triangulated 3D pose: 5 keypoints per bird (beak, head, backpack, tail base, tail tip) |
| `keypoints_2d_top` | 2D keypoints in top-view image space |
| `keypoints_2d_back` | 2D keypoints in back-view image space |
| `radio_*` | 21 per-bird FM radio telemetry signals per frame |
| `vocalizer_idx` | Ground-truth caller index |
| `bird_color` | Color identity of the vocalizer |
| `experiment` | Experiment name (juvExpBP01 / juvExpBP02 / juvExpBP05) |
| `date` | Recording date |

## Ground-truth labels

Labels come from **on-body accelerometer vibration**, not from microphone-based localization. A WhisperSeg model proposes candidate windows from the demodulated signal; events are retained only when audible in microphones, showing characteristic on-body vibration, with no overlapping call from another bird. All candidates are manually reviewed.

### Filtering

From 39,329 reviewed candidates:
- 1,768 rejected as invalid during manual review
- 3,784 removed by overlap filtering (+/-10 ms)
- 152 removed by 3D-to-2D reprojection error (> 40 px)

Yielding **33,625 released events**.

## Per-bird counts

| Experiment | Bird | Events |
|---|---|---|
| BP01 | blue | 5,735 |
| BP01 | peach | 2,277 |
| BP01 | red | 3,963 |
| BP01 | white | 4,244 |
| BP02 | blue | 4,923 |
| BP02 | peach | 4,589 |
| BP02 | red | 2,065 |
| BP05 | brown | 56 |
| BP05 | purple | 2,140 |
| BP05 | yellow | 3,633 |

## Release contents

| Path | Description |
|---|---|
| `v33da-*.parquet` | 23 pooled parquet shards |
| `audio/` | Multichannel microphone WAV files |
| `accelerometer/` | Multichannel accelerometer WAV files |
| `clips/` | Aligned composite MP4 clips |
| `calibrations/` | Per-experiment camera calibration files |
| `explore.ipynb` | Dataset exploration notebook |
| `metadata.json` | Release schema and filter metadata |

## Evaluation regimes

The [benchmark code](https://github.com/marisbasha/v33da) supports three evaluation regimes:

| Regime | What transfers? | Use case |
|---|---|---|
| **Session-disjoint** | Same birds, different days | In-domain ceiling |
| **Held-out experiment** | New birds, new year, new cage | Cross-cohort transfer |
| **Leave-one-bird-out** | One unseen bird per fold | Identity-shift stress test |

## Limitations

V33DA is collected in one aviary with one recording geometry across three experiments. Cross-experiment evaluation combines bird-identity shift, group-composition shift, and recording-date shift. The released benchmark excludes overlapping vocalizations and filters out samples with poor geometric consistency.
