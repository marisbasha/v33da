"""V33DL Dataset — loads parquet shards + external media, returns tensors per vocalization."""

import io
import glob
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Cage bounds for normalization (mm, Z-up convention)
CAGE_X = (-90.8, 1155.2)
CAGE_Y = (-1563.9, -667.9)
CAGE_Z = (236.0, 1530.0)

# Keypoint indices: beak=0, head=1, backpack=2, tailbe=3, tailend=4
KP_BEAK = 0
KP_HEAD = 1
KP_BACKPACK = 2

# Radio power antenna keys
RADIO_POWER_KEYS = ["powRA", "powRB", "powRC", "powRD", "powRM"]

# All radio signal keys for temporal features
RADIO_TEMPORAL_KEYS = [
    "frq", "thetaS",
    "powRA", "powRB", "powRC", "powRD", "powRM",
]

NATIVE_SAMPLE_RATE = 24414.0625
N_CAGE_MICS = 5
DEFAULT_DATASET_ROOT = Path(os.environ.get("V33DA_DATA_DIR", "./data/v33da"))
DEFAULT_SPLIT_ROOT = Path(os.environ.get("V33DA_SPLIT_DIR", "./splits"))
DEFAULT_CACHE_ROOT = Path(os.environ.get("V33DA_CACHE_DIR", "./cache"))

COMPOSITE_VIEW_CROPS = {
    "back": {"y": (0, 724), "x": (994, 2976)},
    "top": {"y": (724, 2016), "x": (994, 2976)},
}

# Mic positions (Z-up, mm) — ordered to match H5 audio channels
# ch0: rear wall, ch1: left wall, ch2: front/door, ch3: right wall, ch4: ceiling
MIC_POSITIONS = np.array([
    [682.0, -1060.0, 775.0],    # ch0: rear wall (Mic4-back)
    [303.0, -1266.0, 775.0],    # ch1: left wall (Mic3-left)
    [682.0, -1434.0, 775.0],    # ch2: front/door (Mic5-door)
    [1020.0, -1266.0, 775.0],   # ch3: right wall (Mic2-right)
    [855.0, -1266.0, 1265.0],   # ch4: ceiling (Mic1)
], dtype=np.float64)


def _load_np(raw_bytes):
    return np.load(io.BytesIO(raw_bytes))


def _normalize_positions(positions_mm):
    """Normalize (n_birds, 3) positions to [0,1] within cage bounds."""
    bounds = np.array([CAGE_X, CAGE_Y, CAGE_Z])  # (3, 2)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return (positions_mm - lo) / (hi - lo)


def _min_inter_bird_dist(positions_mm):
    """Minimum pairwise Euclidean distance between birds (mm)."""
    n = positions_mm.shape[0]
    if n < 2:
        return 0.0
    diffs = positions_mm[:, None, :] - positions_mm[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def _selected_video_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    idx = np.linspace(0, n_frames - 1, num=max_frames)
    idx = np.round(idx).astype(np.int64)
    return np.unique(idx)


def _video_cache_key(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return digest


def _clip_box(points_xy: np.ndarray, width: int, height: int, margin: float = 28.0, min_size: float = 56.0):
    valid = ~np.isnan(points_xy).any(axis=1)
    if not valid.any():
        return None
    pts = points_xy[valid]
    x0 = float(np.min(pts[:, 0]) - margin)
    y0 = float(np.min(pts[:, 1]) - margin)
    x1 = float(np.max(pts[:, 0]) + margin)
    y1 = float(np.max(pts[:, 1]) + margin)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0, min_size)
    half = 0.5 * side
    x0 = max(0.0, cx - half)
    y0 = max(0.0, cy - half)
    x1 = min(float(width), cx + half)
    y1 = min(float(height), cy + half)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _nearest_valid_box(boxes: list[tuple[int, int, int, int] | None], idx: int):
    if boxes[idx] is not None:
        return boxes[idx]
    best = None
    best_dist = 10 ** 9
    for j, box in enumerate(boxes):
        if box is None:
            continue
        dist = abs(j - idx)
        if dist < best_dist:
            best_dist = dist
            best = box
    return best


def _extract_video_candidate_crops(
    video_path: Path,
    kp2d_top: np.ndarray,
    kp2d_back: np.ndarray,
    n_birds: int,
    crop_size: int,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        empty = np.zeros((n_birds, 1, 2, 3, crop_size, crop_size), dtype=np.uint8)
        mask = np.zeros(1, dtype=bool)
        return empty, mask

    sel = _selected_video_indices(len(frames), max_frames)
    frames = [frames[i] for i in sel]
    kp_views = {
        "top": kp2d_top[sel].copy(),
        "back": kp2d_back[sel].copy(),
    }
    for view, arr in kp_views.items():
        arr[:, :, :, 0] -= COMPOSITE_VIEW_CROPS[view]["x"][0]
        arr[:, :, :, 1] -= COMPOSITE_VIEW_CROPS[view]["y"][0]

    n_sel = len(frames)
    crops = np.zeros((n_birds, n_sel, 2, 3, crop_size, crop_size), dtype=np.uint8)
    frame_mask = np.ones(n_sel, dtype=bool)

    for view_idx, view in enumerate(("top", "back")):
        bounds = COMPOSITE_VIEW_CROPS[view]
        view_boxes: list[list[tuple[int, int, int, int] | None]] = [[None] * n_sel for _ in range(n_birds)]
        for ti, frame in enumerate(frames):
            patch = frame[bounds["y"][0]:bounds["y"][1], bounds["x"][0]:bounds["x"][1]]
            h, w = patch.shape[:2]
            for bird_idx in range(n_birds):
                view_boxes[bird_idx][ti] = _clip_box(kp_views[view][ti, bird_idx], w, h)
        for ti, frame in enumerate(frames):
            patch = frame[bounds["y"][0]:bounds["y"][1], bounds["x"][0]:bounds["x"][1]]
            h, w = patch.shape[:2]
            full_box = (0, 0, w, h)
            for bird_idx in range(n_birds):
                box = _nearest_valid_box(view_boxes[bird_idx], ti) or full_box
                x0, y0, x1, y1 = box
                crop = patch[y0:y1, x0:x1]
                crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                crops[bird_idx, ti, view_idx] = crop.transpose(2, 0, 1).astype(np.uint8)
    return crops, frame_mask


class V33DLDataset(Dataset):
    """
    Dataset for V33DL vocalization samples (parquet format).

    Each sample returns:
        audio: (n_mics, n_samples) float32
        bird_positions: (n_birds, 3) float32 — normalized [0,1]
        bird_head_orient: (n_birds, 3) float32
        bird_radio: (n_birds, 5) float32 — mean radio power
        bird_mask: (n_birds,) bool
        label_idx: int
        label_position: (3,) float32 — normalized
        n_birds: int
        min_inter_bird_dist_mm: float
        experiment: str
        sample_id: str
    """

    def __init__(
        self,
        split: str,
        dataset_dir: str | Path | None = None,
        split_root: str | Path | None = None,
    ):
        if dataset_dir is None:
            dataset_dir = Path(os.environ.get("V33DA_DATASET_DIR", DEFAULT_DATASET_ROOT))
        if split_root is None:
            split_root = Path(os.environ.get("V33DA_SPLIT_ROOT", DEFAULT_SPLIT_ROOT))

        self.dataset_dir = Path(dataset_dir)
        self.split_root = Path(split_root)

        shard_pattern = str(self.split_root / split / f"{split}-*.parquet")
        shard_files = sorted(glob.glob(shard_pattern))
        if not shard_files:
            raise FileNotFoundError(f"No parquet shards found: {shard_pattern}")

        tables = [pq.read_table(f) for f in shard_files]
        self.table = pa.concat_tables(tables)

        self.n = len(self.table)
        self.return_audio = True
        self.return_accelerometer = True
        self.return_radio_temporal = True
        self.return_pose_temporal = False
        self.return_video_candidate = False
        self.video_crop_size = 64
        self.video_max_frames = 8
        self.video_cache_root = Path(os.environ.get("V33DA_VIDEO_CACHE_ROOT", DEFAULT_CACHE_ROOT / "video_candidate"))

        # Pre-extract lightweight columns
        self._ids = self.table.column("id").to_pylist()
        self._exps = self.table.column("experiment").to_pylist()
        self._voc_idx = self.table.column("vocalizer_idx").to_pylist()
        self._audio_paths = self.table.column("audio_path").to_pylist()
        self._accel_paths = self.table.column("accelerometer_path").to_pylist()
        self._video_paths = self.table.column("video_path").to_pylist()
        self._has_accelerometer_blob = "accelerometer" in self.table.schema.names
        self._has_audio_blob = "audio" in self.table.schema.names

        # Load bird names per experiment from metadata
        import json
        meta_path = self.dataset_dir / "metadata.json"
        self._exp_bird_names = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._exp_bird_names = {
                exp: info["bird_names"]
                for exp, info in meta.get("experiments", {}).items()
                if isinstance(info, dict) and "bird_names" in info
            }

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = {}
        if self.return_audio:
            # Audio from WAV
            audio_path = self.dataset_dir / self._audio_paths[idx]
            if audio_path.exists():
                waveform, sr = sf.read(str(audio_path))  # (n_samples, n_channels)
                audio = torch.tensor(waveform.T, dtype=torch.float32)  # (5, n_samples)
                # Per-channel normalization — critical for cross-experiment generalization
                # Removes mic gain differences between experiments
                for mi in range(audio.shape[0]):
                    ch_std = audio[mi].std()
                    if ch_std > 1e-6:
                        audio[mi] = audio[mi] / ch_std
            elif self._has_audio_blob:
                raw = self.table.column("audio")[idx].as_py()
                if raw is not None:
                    audio = torch.tensor(_load_np(raw), dtype=torch.float32)
                else:
                    audio = torch.zeros(N_CAGE_MICS, 4880, dtype=torch.float32)
            else:
                # Return silence if file missing (rare edge case)
                audio = torch.zeros(N_CAGE_MICS, 4880, dtype=torch.float32)
            sample["audio"] = audio

        # 3D keypoints
        kp3d = _load_np(self.table.column("keypoints_3d")[idx].as_py())
        # (n_frames, n_birds, 5, 3)
        n_birds = kp3d.shape[1]
        mid = kp3d.shape[0] // 2

        # Bird positions: beak at mid-frame (sound source!)
        positions_mm = kp3d[mid, :, KP_BEAK, :]  # (n_birds, 3)
        positions_norm = _normalize_positions(positions_mm).astype(np.float32)

        # Head orientation: beak - head at mid-frame, normalized
        beak = kp3d[mid, :, KP_BEAK, :]
        head = kp3d[mid, :, KP_HEAD, :]
        head_orient = beak - head
        norms = np.linalg.norm(head_orient, axis=1, keepdims=True)
        head_orient = np.where(norms > 1e-6, head_orient / norms, 0.0).astype(np.float32)

        # Radio features: extract informative signals per bird
        # 5 power means + 5 power stds + frq std + thetaS std = 12 features
        radio = np.zeros((n_birds, 12), dtype=np.float32)
        for ai, key in enumerate(RADIO_POWER_KEYS):
            col = f"radio_{key}"
            raw = self.table.column(col)[idx].as_py()
            if raw is not None:
                arr = _load_np(raw)  # (n_birds, n_frames)
                radio[:, ai] = arr.mean(axis=1)      # mean power
                radio[:, 5 + ai] = arr.std(axis=1)   # power variance
        # Frequency std (FM modulation from vocalization)
        frq_raw = self.table.column("radio_frq")[idx].as_py()
        if frq_raw is not None:
            frq = _load_np(frq_raw)
            radio[:, 10] = frq.std(axis=1) / 10000.0  # normalize to ~0-1 range
        # Master-vector phase angle (thetaS) std
        theta_raw = self.table.column("radio_thetaS")[idx].as_py()
        if theta_raw is not None:
            theta = _load_np(theta_raw)
            radio[:, 11] = theta.std(axis=1) / 100.0  # normalize

        # Raw radio temporal signals: (n_birds, n_channels, n_frames)
        n_radio_frames = kp3d.shape[0]  # same as n_frames
        if self.return_radio_temporal:
            radio_temporal = np.zeros((n_birds, len(RADIO_TEMPORAL_KEYS), n_radio_frames), dtype=np.float32)
            for ci, key in enumerate(RADIO_TEMPORAL_KEYS):
                col = f"radio_{key}"
                raw = self.table.column(col)[idx].as_py()
                if raw is not None:
                    arr = _load_np(raw)  # (n_birds, n_frames)
                    # Pad or truncate to match n_radio_frames
                    nf = min(arr.shape[1], n_radio_frames)
                    radio_temporal[:, ci, :nf] = arr[:, :nf]

        # Pose-derived temporal features for vision / active-speaker-style baselines.
        # Features are relative to the backpack to remove global translation,
        # plus frame-to-frame deltas to preserve short beak/head motion bursts.
        if self.return_pose_temporal:
            rel = kp3d - kp3d[:, :, KP_BACKPACK:KP_BACKPACK + 1, :]
            rel = rel.transpose(1, 0, 2, 3).reshape(n_birds, kp3d.shape[0], -1)  # (n_birds, n_frames, 15)
            rel = rel / 100.0  # mm -> ~decimeter scale
            vel = np.zeros_like(rel, dtype=np.float32)
            vel[:, 1:] = rel[:, 1:] - rel[:, :-1]
            pose_temporal = np.concatenate([rel.astype(np.float32), vel.astype(np.float32)], axis=-1)

        if self.return_video_candidate:
            kp2d_top = _load_np(self.table.column("keypoints_2d_top")[idx].as_py())
            kp2d_back = _load_np(self.table.column("keypoints_2d_back")[idx].as_py())
            cache_dir = self.video_cache_root / f"c{self.video_crop_size}_f{self.video_max_frames}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{_video_cache_key(self._ids[idx])}.npz"
            if cache_path.exists():
                cached = np.load(cache_path)
                video_crops = cached["crops"]
                video_frame_mask = cached["frame_mask"]
            else:
                video_path = self.dataset_dir / self._video_paths[idx]
                video_crops, video_frame_mask = _extract_video_candidate_crops(
                    video_path=video_path,
                    kp2d_top=kp2d_top,
                    kp2d_back=kp2d_back,
                    n_birds=n_birds,
                    crop_size=self.video_crop_size,
                    max_frames=self.video_max_frames,
                )
                tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(tmp_path, crops=video_crops, frame_mask=video_frame_mask)
                    os.replace(tmp_path, cache_path)
                except Exception:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

        # Label
        label_idx = self._voc_idx[idx]
        label_position = positions_norm[label_idx]

        # Raw accelerometer / FM-demodulated on-body signal: (n_birds, n_samples)
        if self.return_accelerometer:
            accel_path = self.dataset_dir / self._accel_paths[idx]
            if accel_path.exists():
                accel_waveform, _ = sf.read(str(accel_path))
                accel = accel_waveform.T.astype(np.float32)
                if accel.shape[0] < n_birds:
                    accel = np.pad(accel, ((0, n_birds - accel.shape[0]), (0, 0)))
                elif accel.shape[0] > n_birds:
                    accel = accel[:n_birds]
            elif self._has_accelerometer_blob:
                raw = self.table.column("accelerometer")[idx].as_py()
                if raw is not None:
                    accel = _load_np(raw).astype(np.float32)
                elif self.return_audio:
                    accel = np.zeros((n_birds, sample["audio"].shape[1]), dtype=np.float32)
                else:
                    accel = np.zeros((n_birds, 4880), dtype=np.float32)
            elif self.return_audio:
                accel = np.zeros((n_birds, sample["audio"].shape[1]), dtype=np.float32)
            else:
                accel = np.zeros((n_birds, 4880), dtype=np.float32)

        # Min inter-bird distance
        min_dist = _min_inter_bird_dist(positions_mm)

        # Bird names
        exp = self._exps[idx]
        bird_names = self._exp_bird_names.get(exp, [f"bird_{i}" for i in range(n_birds)])

        sample.update({
            "bird_positions": torch.tensor(positions_norm),
            "bird_head_orient": torch.tensor(head_orient),
            "bird_radio": torch.tensor(radio),
            "bird_mask": torch.ones(n_birds, dtype=torch.bool),
            "label_idx": label_idx,
            "label_position": torch.tensor(label_position),
            "n_birds": n_birds,
            "min_inter_bird_dist_mm": min_dist,
            "bird_names": bird_names,
            "experiment": exp,
            "sample_id": self._ids[idx],
        })
        if self.return_radio_temporal:
            sample["bird_radio_temporal"] = torch.tensor(radio_temporal)
        if self.return_pose_temporal:
            sample["bird_pose_temporal"] = torch.tensor(pose_temporal)
        if self.return_video_candidate:
            sample["bird_video_crops"] = torch.tensor(video_crops.astype(np.float32) / 255.0)
            sample["video_frame_mask"] = torch.tensor(video_frame_mask)
        if self.return_accelerometer:
            sample["accelerometer"] = torch.tensor(accel)
        return sample
