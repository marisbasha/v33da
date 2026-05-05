# Who Called? V33DA: A Physically Verified Multimodal Benchmark for Vocal Attribution in Zebra Finch Groups

**V33DA** is a multimodal benchmark for spatial vocal attribution in social zebra finches: 33,625 verified vocalization events from 10 individually identified birds across 3 experiments, each with synchronized 5-channel audio, three camera views, calibrated 3D pose for every visible candidate, per-bird FM radio telemetry, and an on-body accelerometer-derived oracle channel that anchors caller labels.

> **Task.** Given a detected vocalization and the set of birds visible at that moment, determine which bird produced the call. Caller identity is verified physically from on-body vibration; the accelerometer channel is withheld from benchmark models and used only by an oracle ceiling.

| | BP01 | BP02 | BP05 | Total |
|---|---|---|---|---|
| Birds | 4 | 3 | 3 | 10 |
| Events | 16,219 | 11,577 | 5,829 | 33,625 |
| Days | 10 | 3 | 3 | 16 |
| Raw hours | ~46h | ~14h | ~14h | ~74h |

**Dataset:** [songbirdini/V33DA](https://huggingface.co/datasets/songbirdini/V33DA) &nbsp;·&nbsp; **Companion release:** [songbirdini/v33da_pp](https://huggingface.co/datasets/songbirdini/v33da_pp) (V33DA++: overlap and longer-context buckets)

## Sample data

A self-contained 100-event sample ships with the repository at [`data/v33da_sample/`](data/v33da_sample/) (~60 MB, stratified across BP01/BP02/BP05). It contains real parquet rows, multichannel audio WAVs, accelerometer WAVs, composite video clips, and per-experiment camera calibrations — enough to run the loader end-to-end without downloading the full dataset.

### What's in the sample

```
data/v33da_sample/
  v33da_sample.parquet        # 100 rows, full V33DA schema
  audio/<experiment>/...       # 5-channel cage-microphone WAVs
  accelerometer/<experiment>/...  # On-body vibration WAVs (oracle)
  clips/<experiment>/...       # Composite multi-view MP4s
  calibrations/<experiment>/   # Camera intrinsics + extrinsics
  metadata.json
  split_summary.json
```

### Walkthrough notebook

After installing the repo, open the walkthrough notebook to inspect every modality (audio spectrogram, accelerometer oracle, 3D pose, radio telemetry, video) for a single event and to see how 3D keypoints reproject into the camera views using the shipped calibrations:

```bash
pip install -e .
pip install matplotlib jupyter soundfile
jupyter notebook notebooks/walkthrough.ipynb
```

### Loading the sample directly

```python
import io, numpy as np, pyarrow.parquet as pq, soundfile as sf
from pathlib import Path

ROOT = Path("data/v33da_sample")
table = pq.read_table(ROOT / "v33da_sample.parquet")
row = table.slice(0, 1).to_pydict()

# 3D pose: (T_frames, N_birds, 5, 3) float32 — 5 keypoints per bird
kp3d = np.load(io.BytesIO(row["keypoints_3d"][0]))

# Multichannel audio
audio, sr = sf.read(ROOT / row["audio_path"][0])  # (T, 5) at 24,414 Hz

# Ground-truth caller index (within visible candidates for this event)
caller = row["vocalizer_idx"][0]
print(f"Experiment {row['experiment'][0]}, caller idx={caller}, kp3d={kp3d.shape}")
```

### Rebuilding or resizing the sample

To regenerate from the full dataset (different N, different seed, with or without video):

```bash
huggingface-cli download songbirdini/V33DA --repo-type dataset --local-dir /tmp/v33da_full
python scripts/build_sample.py --src /tmp/v33da_full --dst data/v33da_sample --n 100 --include-video
```

## Quick start (full dataset)

### 1. Install

```bash
pip install -e .
```

### 2. Download the dataset

```bash
pip install huggingface_hub
huggingface-cli download songbirdini/V33DA --repo-type dataset --local-dir data/v33da
```

Or in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("songbirdini/V33DA", repo_type="dataset", local_dir="data/v33da")
```

### 3. Prepare splits

```bash
python scripts/prepare_v33da_splits.py
```

This creates session-disjoint, held-out experiment, and leave-one-bird-out splits under `splits/`.

### 4. Train and evaluate

```bash
# Run deterministic baselines (Random, SRP — no training)
python scripts/train.py --model baselines --split both

# Train a single model on session split
python scripts/train.py --model spatial_scorer --split session

# Train on held-out experiment transfer
python scripts/train.py --model accdoa --split heldout_experiment --heldout-experiment all

# Leave-one-bird-out
python scripts/train.py --model contrastive --split loo

# Train ALL models across ALL splits
python scripts/train.py --model all --split both
```

Available models: `seldnet`, `seldnet_plus`, `spatial_scorer`, `contrastive`, `pairrank`,
`pose_motion`, `video_candidate`, `beam_fusion`, `neural_srp`, `accel_oracle`,
`accdoa`, `einv2`, `cstformer`, `random`, `srp`.

Models with unique training paradigms have dedicated scripts:

```bash
python scripts/train_whisper_candidate.py --split both    # Frozen Whisper encoder
python scripts/train_srp_reranker.py --split both         # Precomputed SRP features
python scripts/train_loc_first.py                         # Pure localization regression
```

## Data modalities

Each event provides:

| Modality | Description | Shape |
|---|---|---|
| **Audio** | 5-channel cage microphones at 24,414 Hz | `(5, T)` float32 WAV |
| **3D pose** | 5 keypoints per bird (beak, head, backpack, tail base, tail tip) | `(N_birds, 5, 3)` |
| **Head orientation** | Unit vector from head center to beak | `(N_birds, 3)` |
| **Radio telemetry** | 12D summary + temporal signal per bird | `(N_birds, 12)` + `(N_birds, T_radio)` |
| **Video** | Multi-view synchronized RGB (top, back, side) | composite frame |
| **Accelerometer** | On-body vibration (oracle only, withheld from models) | `(N_birds, T)` |

## Evaluation regimes

| Regime | What transfers? | Use case |
|---|---|---|
| **Session-disjoint** | Same birds, different days | In-domain ceiling |
| **Held-out experiment** | New birds, new year, new cage | Cross-cohort transfer |
| **Leave-one-bird-out** | One unseen bird per fold | Identity-shift stress test |

## Baselines

| Family | Models | Key idea |
|---|---|---|
| Physics | SRP, SRP Reranker, NeuralSRP | Steered response power from calibrated geometry |
| Closed-set | SELDnet | Fixed identity head from multichannel audio |
| Candidate-centric | SELDnet++, PairRank, SpatialScorer, Contrastive, Whisper candidate | Score visible candidates directly |
| SELD / loc-first | ACCDOA, EINV2, CST-former, BeamFusion, AudioANN, RadioANN | Localize first, then assign nearest candidate |
| Vision-only | PoseMotion, VideoCandidate | No audio; 3D keypoints or RGB crops |
| Oracle | AccelOracle | On-body accelerometer (solvability ceiling) |

## V33DA++ (companion release)

[V33DA++](https://huggingface.co/datasets/songbirdini/v33da_pp) is an auxiliary extension to the main V33DA benchmark, built from the same recordings and the same 10 birds but covering data **outside** the strict single-vocalizer attribution setting. It is intended for tasks that benefit from overlapping callers or longer temporal context (source separation, vocal activity detection, audio-visual sync, active-speaker detection).

V33DA++ ships three buckets:
- **`overlap/`** — reviewer-verified events excluded from V33DA because another bird vocalized in the same window. Each sample carries the list of overlapping callers verified from on-body accelerometer channels.
- **`padded/`** — ±2 s context windows around each V33DA event, with original event onsets and offsets preserved inside the longer window.
- **`overlap_padded/`** — overlap events with the same ±2 s context.

Download a single bucket:

```bash
huggingface-cli download songbirdini/v33da_pp --repo-type dataset \
    --include "padded/*" --local-dir data/v33da_pp
```

V33DA++ is **not** the benchmark used in the paper's main results tables.

## Repository structure

```
v33da/
  baselines/       # SRP-PHAT, GCC-PHAT, SELDnet, SELDnet++
  configs/         # YAML configs (base, audio_only, audio_3d, ...)
  data/            # Dataset loader, collation, feature extraction, splits
  evaluation/      # Metrics (accuracy, macro-F1, localization error)
  models/          # All learned models (SpatialScorer, ACCDOA, EINV2, ...)
  training/        # Trainer, loss functions
scripts/
  train.py                   # Unified training for all models (--model <name>)
  prepare_v33da_splits.py    # Generate all train/val/test splits
  train_whisper_candidate.py # Whisper-based candidate scorer (frozen encoder)
  train_srp_reranker.py      # SRP reranker (precomputed features)
  train_loc_first.py         # Pure localization regression baseline
  build_sample.py            # Build the 100-event repo sample from full V33DA
  dataset_statistics.py      # Compute dataset statistics
data/
  v33da_sample/              # 100-event self-contained sample (~60 MB)
notebooks/
  walkthrough.ipynb          # Reviewer walkthrough over the sample
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `V33DA_DATA_DIR` | `./data/v33da` | Path to downloaded dataset |
| `V33DA_SPLIT_DIR` | `./splits` | Path to generated splits |
| `V33DA_CACHE_DIR` | `./cache` | Path for feature caches |

## Hardware

All experiments in the paper were run on a single NVIDIA RTX 3090 (24 GB). Total compute: ~53 GPU-hours.

## Citation

If you use V33DA, please cite:

```bibtex
@unpublished{basha2026v33da,
  title  = {Who Called? V33DA: A Physically Verified Multimodal Benchmark for Vocal Attribution in Zebra Finch Groups},
  author = {Basha, Maris and Wang, Yuhang and Chen, Xiaoran and Cheng, Longbiao and Yapura, Luca and Zai, Anja T. and Salzmann, Mathieu and Hahnloser, Richard},
  year   = {2026},
  note   = {Under review},
}
```

## License

MIT
