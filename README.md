# V33DA: A Multi-Modal Benchmark for Vocal Attribution in Freely Interacting Zebra Finches

**V33DA** (**V**ocal **3**-Experiment **3D** **A**ttribution) is the first benchmark for attributing vocalizations to individual birds in a freely interacting group, using synchronized multi-modal recordings: 5-channel audio, multi-view video, 3D pose, and FM radio telemetry.

> **Task.** Given a detected vocalization and the set of birds visible at that moment, determine which bird produced the call.

| | BP01 | BP02 | BP05 | Total |
|---|---|---|---|---|
| Birds | 4 | 3 | 3 | 10 |
| Events | 16,219 | 11,577 | 5,829 | 33,625 |
| Days | 10 | 3 | 3 | 16 |
| Raw hours | ~46h | ~14h | ~14h | ~74h |

## Quick start

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
  dataset_statistics.py      # Compute dataset statistics
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `V33DA_DATA_DIR` | `./data/v33da` | Path to downloaded dataset |
| `V33DA_SPLIT_DIR` | `./splits` | Path to generated splits |
| `V33DA_CACHE_DIR` | `./cache` | Path for feature caches |

## Hardware

All experiments in the paper were run on a single NVIDIA RTX 3090 (24 GB). Total compute: ~53 GPU-hours.

## License

MIT
