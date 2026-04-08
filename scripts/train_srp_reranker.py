#!/usr/bin/env python3
"""Split-aware SRP + reranker baseline for standalone V33DA.

Physics provides frozen SRP-style spatial evidence; learning only reranks
candidates using that evidence plus candidate-side state.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v33da.data.collate import v33da_collate_fn
from v33da.data.dataset import CAGE_X, CAGE_Y, CAGE_Z, MIC_POSITIONS
from v33da.data.splits import (
    dataset_experiment_counts,
    iter_requested_birds,
    iter_requested_experiments,
    load_split_triplet,
    short_experiment_name,
)
from v33da.evaluation.metrics import attribution_accuracy, macro_f1


SPEED_OF_SOUND = 343000.0
SR = 24414.0
SEEDS = [42, 123, 456, 789, 1337]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIC_PAIRS = list(combinations(range(5), 2))
BOUNDS = np.array([CAGE_X, CAGE_Y, CAGE_Z], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRP + reranker on standalone V33DA.")
    parser.add_argument(
        "--split",
        choices=["session", "unseen_bird", "heldout_experiment", "leave_one_bird_out", "both"],
        default="both",
    )
    parser.add_argument("--heldout-experiment", default="all")
    parser.add_argument("--heldout-bird", default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def denorm_positions(pos_norm):
    return pos_norm * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]


def compute_srp_features(audio_np, pos_mm, n_birds):
    """Per-bird frozen SRP feature stack from GCC peaks."""
    n_samples = audio_np.shape[1]
    audio = audio_np.copy()
    for mi in range(audio.shape[0]):
        audio[mi] = (audio[mi] - audio[mi].mean()) / (audio[mi].std() + 1e-8)

    gccs = []
    for mi, mj in MIC_PAIRS:
        xi = np.fft.rfft(audio[mi])
        xj = np.fft.rfft(audio[mj])
        cross = xi * np.conj(xj)
        cc = np.fft.irfft(cross / (np.abs(cross) + 1e-10))
        cc = np.concatenate([cc[n_samples // 2 :], cc[: n_samples // 2 + 1]])
        gccs.append(cc)

    center = n_samples // 2
    features = np.zeros((4, len(MIC_PAIRS) * 3), dtype=np.float32)
    for bi in range(n_birds):
        fi = 0
        for pi, (mi, mj) in enumerate(MIC_PAIRS):
            d_i = np.linalg.norm(pos_mm[bi] - MIC_POSITIONS[mi])
            d_j = np.linalg.norm(pos_mm[bi] - MIC_POSITIONS[mj])
            expected = int(round((d_i - d_j) / SPEED_OF_SOUND * SR))

            vals = [
                gccs[pi][center + expected + off]
                for off in range(-20, 21)
                if 0 <= center + expected + off < len(gccs[pi])
            ]
            vals_narrow = [
                gccs[pi][center + expected + off]
                for off in range(-5, 6)
                if 0 <= center + expected + off < len(gccs[pi])
            ]

            peak = max(vals) if vals else 0.0
            narrow_mean = float(np.mean(vals_narrow)) if vals_narrow else 0.0
            wide_mean = float(np.mean(vals)) if vals else 0.0

            features[bi, fi] = peak
            fi += 1
            features[bi, fi] = narrow_mean
            fi += 1
            features[bi, fi] = peak / (abs(wide_mean) + 1e-10) if vals else 0.0
            fi += 1

    return features


def srp_baseline_from_features(features, n_birds):
    scores = [features[bi, ::3].sum() for bi in range(n_birds)]
    return int(np.argmax(scores))


def precompute_srp(ds, split_name):
    print(f"  Precomputing SRP features for {split_name} ({len(ds)} samples)...", flush=True)
    start = time.time()
    rows = []
    srp_correct = 0
    for idx in range(len(ds)):
        sample = ds[idx]
        pos_mm = denorm_positions(sample["bird_positions"].numpy())
        feats = compute_srp_features(sample["audio"].numpy(), pos_mm[: sample["n_birds"]], sample["n_birds"])
        rows.append(feats)
        if srp_baseline_from_features(feats, sample["n_birds"]) == sample["label_idx"]:
            srp_correct += 1
        if (idx + 1) % 2000 == 0:
            print(f"    {idx + 1}/{len(ds)} in {time.time() - start:.0f}s", flush=True)
    print(f"    done in {time.time() - start:.0f}s", flush=True)
    return rows, srp_correct / len(ds)


class FeatureDataset(Dataset):
    def __init__(self, base_ds, feature_rows):
        self.base_ds = base_ds
        self.feature_rows = feature_rows

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        sample["srp_features"] = torch.tensor(self.feature_rows[idx], dtype=torch.float32)
        return sample


def feature_collate(batch):
    out = v33da_collate_fn(batch)
    target_birds = 4

    def _pad_bird_dim(x, pad_value=0.0):
        if x.shape[1] >= target_birds:
            return x
        pad_amt = target_birds - x.shape[1]
        if x.ndim == 2:
            return F.pad(x, (0, pad_amt), value=pad_value)
        if x.ndim == 3:
            return F.pad(x, (0, 0, 0, pad_amt), value=pad_value)
        if x.ndim == 4:
            return F.pad(x, (0, 0, 0, 0, 0, pad_amt), value=pad_value)
        raise ValueError(f"Unsupported bird tensor rank: {x.ndim}")

    out["accelerometer"] = _pad_bird_dim(out["accelerometer"])
    out["bird_positions"] = _pad_bird_dim(out["bird_positions"])
    out["bird_head_orient"] = _pad_bird_dim(out["bird_head_orient"])
    out["bird_radio"] = _pad_bird_dim(out["bird_radio"])
    out["bird_radio_temporal"] = _pad_bird_dim(out["bird_radio_temporal"])
    out["bird_mask"] = _pad_bird_dim(out["bird_mask"], pad_value=0).to(torch.bool)
    out["srp_features"] = torch.stack([b["srp_features"] for b in batch])
    return out


class SRPReranker(nn.Module):
    """Rerank candidates from frozen SRP features plus candidate state."""

    def __init__(self, srp_dim=30, dropout=0.2):
        super().__init__()
        in_dim = srp_dim + 3 + 3 + 12
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, batch):
        pos = batch["bird_positions"]
        n_birds = pos.shape[1]
        head = batch.get("bird_head_orient", torch.zeros_like(pos))
        radio = batch.get("bird_radio", torch.zeros(pos.shape[0], n_birds, 12, device=pos.device))
        feat = torch.cat([batch["srp_features"][:, :n_birds], pos, head, radio], dim=-1)
        logits = self.scorer(feat).squeeze(-1)
        logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))
        return {"logits": logits}


def maybe_limit(ds, max_samples):
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return torch.utils.data.Subset(ds, list(range(max_samples)))


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
            labels.append(batch["label_idx"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "acc": float(attribution_accuracy(preds, labels)),
        "f1": float(macro_f1(preds, labels)),
    }


def train_once(train_loader, val_loader, test_loader, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = SRPReranker(dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_test = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["label_idx"], label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            correct += (out["logits"].argmax(dim=-1) == batch["label_idx"]).sum().item()
            total += batch["label_idx"].size(0)
        scheduler.step()

        val = evaluate(model, val_loader)
        if val["acc"] > best_val:
            best_val = val["acc"]
            best_test = evaluate(model, test_loader)
            best_epoch = epoch + 1

        if (epoch + 1) % 20 == 0:
            print(f"    ep{epoch+1}: train={correct/total:.3f} val={val['acc']:.3f}/{val['f1']:.3f}", flush=True)

    return best_test, best_epoch


def run_split(split_name, args, heldout_experiment=None, heldout_bird=None):
    train_ds, val_ds, test_ds, split_meta = load_split_triplet(split_name, heldout_experiment, heldout_bird)
    train_ds = maybe_limit(train_ds, args.max_samples)
    val_ds = maybe_limit(val_ds, args.max_samples)
    test_ds = maybe_limit(test_ds, args.max_samples)

    header = split_name.upper()
    if split_name == "heldout_experiment" and heldout_experiment is not None:
        header = f"{header} ({short_experiment_name(heldout_experiment)})"
    elif split_name == "leave_one_bird_out":
        header = f"LEAVE_ONE_BIRD_OUT ({short_experiment_name(heldout_experiment)} bird {heldout_bird})"
    print(f"\n{'=' * 72}\n  SRP RERANKER: {header}\n{'=' * 72}", flush=True)
    print(f"  sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    if split_name == "heldout_experiment" and heldout_experiment is not None:
        print(f"  train experiments: {dataset_experiment_counts(train_ds)}", flush=True)
        print(f"  val experiments:   {dataset_experiment_counts(val_ds)}", flush=True)
        print(f"  test experiments:  {dataset_experiment_counts(test_ds)}", flush=True)
    elif split_name == "leave_one_bird_out":
        print(f"  heldout bird: {heldout_bird}", flush=True)

    train_feats, train_srp = precompute_srp(train_ds, "train")
    val_feats, val_srp = precompute_srp(val_ds, "val")
    test_feats, test_srp = precompute_srp(test_ds, "test")
    print(f"  SRP baseline test acc={test_srp:.3f}", flush=True)

    train_feature_ds = FeatureDataset(train_ds, train_feats)
    val_feature_ds = FeatureDataset(val_ds, val_feats)
    test_feature_ds = FeatureDataset(test_ds, test_feats)

    kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": feature_collate,
        "pin_memory": DEVICE.startswith("cuda"),
    }

    results = {
        "train_srp": float(train_srp),
        "val_srp": float(val_srp),
        "test_srp": float(test_srp),
        "seeds": {},
    }
    accs = []

    for seed in SEEDS:
        print(f"  seed {seed}", flush=True)
        train_loader = DataLoader(train_feature_ds, shuffle=True, **kw)
        val_loader = DataLoader(val_feature_ds, shuffle=False, **kw)
        test_loader = DataLoader(test_feature_ds, shuffle=False, **kw)
        best_test, best_epoch = train_once(train_loader, val_loader, test_loader, args.epochs, seed)
        results["seeds"][str(seed)] = {"best_epoch": best_epoch, **best_test}
        accs.append(best_test["acc"])
        print(
            f"    best test acc={best_test['acc']:.3f} f1={best_test['f1']:.3f} @ ep{best_epoch}",
            flush=True,
        )

    results["summary"] = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "delta_vs_srp": float(np.mean(accs) - test_srp),
    }
    return split_meta["key"], results


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)

    if args.split == "both":
        eval_specs = [("session", None, None), ("unseen_bird", None, None)]
    elif args.split == "heldout_experiment":
        eval_specs = [("heldout_experiment", exp, None) for exp in iter_requested_experiments(args.heldout_experiment)]
    elif args.split == "leave_one_bird_out":
        eval_specs = []
        for exp in iter_requested_experiments(args.heldout_experiment):
            for bird in iter_requested_birds(exp, args.heldout_bird):
                eval_specs.append(("leave_one_bird_out", exp, bird))
    else:
        eval_specs = [(args.split, None, None)]

    all_results = {}
    for split_name, heldout_experiment, heldout_bird in eval_specs:
        split_key, result = run_split(split_name, args, heldout_experiment, heldout_bird)
        all_results[split_key] = result

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path("results") / f"srp_reranker{suffix}.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
