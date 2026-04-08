#!/usr/bin/env python3
"""
Unified training script for V33DA baselines.

Supports all candidate-centric and SELD models via --model flag.
Models with fundamentally different training paradigms (whisper_candidate,
srp_reranker, loc_first) have their own scripts.

Examples:
    # Session split, single model
    python scripts/train.py --model spatial_scorer --split session

    # Held-out experiment transfer
    python scripts/train.py --model accdoa --split heldout_experiment --heldout-experiment all

    # Leave-one-bird-out
    python scripts/train.py --model contrastive --split loo

    # All models, all splits
    python scripts/train.py --model all --split both
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v33da.data.collate import v33da_collate_fn
from v33da.data.dataset import CAGE_X, CAGE_Y, CAGE_Z, MIC_POSITIONS
from v33da.data.features import MelSpecExtractor
from v33da.data.splits import (
    dataset_experiment_counts,
    iter_requested_birds,
    iter_requested_experiments,
    load_split_triplet,
    short_experiment_name,
)
from v33da.evaluation.metrics import attribution_accuracy, localization_error, macro_f1

BOUNDS = np.array([CAGE_X, CAGE_Y, CAGE_Z], dtype=np.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_seldnet(n_birds, **kw):
    from v33da.baselines.seldnet import SELDnet
    return SELDnet(n_mics=5, max_birds=n_birds)


def _build_seldnet_plus(n_birds, use_radio=True, **kw):
    from v33da.models.seldnet_plus import SELDnetPlus
    return SELDnetPlus(n_mics=5, use_radio=use_radio, n_radio_channels=7)


def _build_spatial_scorer(n_birds, use_radio=True, n_experiments=1, **kw):
    from v33da.models import SpatialScorer
    return SpatialScorer(use_radio=use_radio, n_radio_channels=7,
                         n_experiments=n_experiments, dropout=0.15)


def _build_contrastive(n_birds, use_radio=True, **kw):
    from v33da.models.contrastive import ContrastiveAttributor
    return ContrastiveAttributor(use_radio=use_radio, n_radio_channels=7)


def _build_pairrank(n_birds, **kw):
    from v33da.models.pairrank import PairRankNet
    return PairRankNet()


def _build_pose_motion(n_birds, **kw):
    from v33da.models import PoseMotionCandidate
    return PoseMotionCandidate()


def _build_video_candidate(n_birds, **kw):
    from v33da.models import VideoCandidate
    return VideoCandidate()


def _build_beam_fusion(n_birds, modality="audio_3d_radio", **kw):
    from v33da.models.beam_fusion import BeamFusion
    return BeamFusion(modality=modality, n_mics=5, d_model=256)


def _build_neural_srp(n_birds, use_radio=True, **kw):
    from v33da.models.neural_srp import NeuralSRP
    return NeuralSRP(use_radio=use_radio, n_radio_channels=7)


def _build_accel_oracle(n_birds, **kw):
    from v33da.models.accel_oracle import AccelOracleNet
    return AccelOracleNet()


def _build_literature(n_birds, variant="accdoa", **kw):
    from v33da.models.literature_seld import literature_model
    return literature_model(variant)


MODEL_REGISTRY = {
    "seldnet":          dict(build=_build_seldnet, epochs=50, lr=1e-4, bs=64),
    "seldnet_plus":     dict(build=_build_seldnet_plus, epochs=60, lr=1e-4, bs=64),
    "spatial_scorer":   dict(build=_build_spatial_scorer, epochs=8, lr=1e-4, bs=16, lambda_adv=0.05),
    "contrastive":      dict(build=_build_contrastive, epochs=8, lr=1e-4, bs=16),
    "pairrank":         dict(build=_build_pairrank, epochs=8, lr=1e-4, bs=16, wd=1e-2),
    "pose_motion":      dict(build=_build_pose_motion, epochs=8, lr=1e-4, bs=16),
    "video_candidate":  dict(build=_build_video_candidate, epochs=8, lr=1e-4, bs=4),
    "beam_fusion":      dict(build=_build_beam_fusion, epochs=8, lr=1e-4, bs=16),
    "neural_srp":       dict(build=_build_neural_srp, epochs=8, lr=3e-4, bs=16),
    "accel_oracle":     dict(build=_build_accel_oracle, epochs=6, lr=1e-4, bs=16),
    "accdoa":           dict(build=lambda **kw: _build_literature(variant="accdoa", **kw), epochs=8, lr=1e-4, bs=16),
    "einv2":            dict(build=lambda **kw: _build_literature(variant="einv2", **kw), epochs=8, lr=1e-4, bs=16),
    "cstformer":        dict(build=lambda **kw: _build_literature(variant="cstformer", **kw), epochs=8, lr=1e-4, bs=16),
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def denorm(pos):
    return pos * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]


def mutate_dataset_for_model(ds, model_name, args):
    """Apply model-specific dataset mutations."""
    if model_name == "pose_motion":
        ds.use_audio = False
        ds.use_accel = False
        ds.use_radio = False
        ds.use_pose_temporal = True
    elif model_name == "video_candidate":
        ds.use_audio = False
        ds.use_accel = False
        ds.use_radio = False
        ds.use_pose_temporal = False
        ds.use_video_candidate = True
        ds.video_crop_size = getattr(args, "crop_size", 32)
        ds.video_n_frames = getattr(args, "video_frames", 4)
    elif model_name == "accel_oracle":
        ds.use_accel = True


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss(model_name, out, batch, args):
    """Compute loss for a model. Returns (total_loss, n_correct)."""
    labels = batch["label"].to(DEVICE)

    if model_name == "pairrank":
        from v33da.models.pairrank import pairrank_loss
        loss, correct = pairrank_loss(out, batch)
        return loss, correct.item()

    if model_name == "accel_oracle":
        from v33da.models.accel_oracle import accel_oracle_loss
        loss, correct = accel_oracle_loss(out, batch)
        return loss, correct.item()

    if model_name in ("accdoa", "einv2", "cstformer"):
        from v33da.models.literature_seld import literature_loss
        variant = model_name
        if hasattr(args, "radio_ctx") and args.radio_ctx:
            variant += "_radioctx"
        loss = literature_loss(variant, out, batch, BOUNDS, DEVICE)
        # attribution from model output
        if "logits" in out:
            preds = out["logits"].argmax(dim=-1)
        elif "position" in out:
            pred_pos = out["position"].detach().cpu().numpy()
            cand_pos = batch["candidate_positions"].numpy()
            preds = torch.tensor([
                np.argmin(np.linalg.norm(cand_pos[i] - pred_pos[i], axis=-1))
                for i in range(len(pred_pos))
            ])
        else:
            preds = labels  # fallback
        correct = (preds.to(labels.device) == labels).sum().item()
        return loss, correct

    # Default: CE + optional localization MSE
    logits = out["logits"]
    ls = 0.1 if model_name == "spatial_scorer" else 0.0
    loss = F.cross_entropy(logits, labels, label_smoothing=ls)

    if "position" in out and "target_position" in batch:
        tgt = batch["target_position"].to(DEVICE)
        lambda_loc = getattr(args, "lambda_loc", 0.05)
        loss = loss + lambda_loc * F.mse_loss(out["position"], tgt)

    if model_name == "spatial_scorer" and "exp_logits" in out:
        exp_labels = batch.get("experiment_idx")
        if exp_labels is not None:
            lambda_adv = getattr(args, "lambda_adv", 0.05)
            adv_loss = F.cross_entropy(out["exp_logits"], exp_labels.to(DEVICE))
            loss = loss + lambda_adv * adv_loss

    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return loss, correct


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_run(model_name, train_ds, val_ds, args, seed, extra_kw=None):
    """Train a single model on a single split with a single seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = MODEL_REGISTRY[model_name]
    epochs = getattr(args, "epochs", None) or cfg.get("epochs", 8)
    bs = getattr(args, "batch_size", None) or cfg.get("bs", 16)
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("wd", 1e-4)

    mutate_dataset_for_model(train_ds, model_name, args)
    mutate_dataset_for_model(val_ds, model_name, args)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=v33da_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=v33da_collate_fn, pin_memory=True)

    build_kw = dict(n_birds=len(set(train_ds.bird_names)),
                    use_radio=not getattr(args, "no_radio", False))
    if extra_kw:
        build_kw.update(extra_kw)
    model = cfg["build"](**build_kw).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc, best_state = 0.0, None
    patience, patience_limit = 0, max(10, epochs // 3)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)
            out = model(batch)
            loss, correct = compute_loss(model_name, out, batch, args)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch["label"].size(0)
            train_correct += correct
            train_total += batch["label"].size(0)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                _, correct = compute_loss(model_name, out, batch, args)
                val_correct += correct
                val_total += batch["label"].size(0)

        val_acc = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  [{epoch+1:3d}/{epochs}] train_acc={train_acc:.3f} val_acc={val_acc:.3f} best={best_acc:.3f}")

        if patience >= patience_limit:
            print(f"  Early stop at epoch {epoch+1}")
            break

    return {"best_val_acc": best_acc, "n_params": n_params}


# ---------------------------------------------------------------------------
# Split matrix runner
# ---------------------------------------------------------------------------

def run_split_matrix(model_name, args):
    """Run model across the requested split matrix with multiple seeds."""
    seeds = [42, 123, 456]
    all_results = []

    def run_one_config(train_ds, val_ds, test_ds, config_name, extra_kw=None):
        print(f"\n{'='*60}")
        print(f"  {model_name} | {config_name}")
        print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            result = train_one_run(model_name, train_ds, val_ds, args, seed, extra_kw)
            seed_results.append(result)

        accs = [r["best_val_acc"] for r in seed_results]
        entry = {
            "model": model_name,
            "config": config_name,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "seeds": seed_results,
        }
        all_results.append(entry)
        print(f"  => {config_name}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    split = args.split
    if split in ("session", "both"):
        train_ds, val_ds, test_ds = load_split_triplet("session")
        run_one_config(train_ds, val_ds, test_ds, "session")

    if split in ("heldout_experiment", "both"):
        for exp_name, train_ds, val_ds, test_ds in iter_requested_experiments(args.heldout_experiment):
            extra_kw = {}
            if model_name == "spatial_scorer":
                extra_kw["n_experiments"] = len(dataset_experiment_counts(train_ds))
            run_one_config(train_ds, val_ds, test_ds,
                           f"heldout_{short_experiment_name(exp_name)}", extra_kw)

    if split in ("loo", "leave_one_bird_out", "both"):
        for bird_name, train_ds, val_ds, test_ds in iter_requested_birds(args.heldout_bird):
            run_one_config(train_ds, val_ds, test_ds, f"loo_{bird_name}")

    return all_results


# ---------------------------------------------------------------------------
# Non-neural baselines
# ---------------------------------------------------------------------------

def run_deterministic_baselines(args):
    """Run Random, Majority, SRP baselines (no training)."""
    from itertools import combinations
    from v33da.baselines.srp_phat import SRPPHATBaseline

    results = []
    for split_name in ("session", "heldout_experiment", "loo"):
        if args.split not in (split_name, "both"):
            continue
        if split_name == "session":
            configs = [("session", *load_split_triplet("session"))]
        elif split_name == "heldout_experiment":
            configs = [(f"heldout_{short_experiment_name(e)}", tr, va, te)
                       for e, tr, va, te in iter_requested_experiments(args.heldout_experiment)]
        else:
            configs = [(f"loo_{b}", tr, va, te)
                       for b, tr, va, te in iter_requested_birds(args.heldout_bird)]

        for config_name, train_ds, val_ds, test_ds in configs:
            n_test = len(test_ds)
            # Random
            labels = np.array([test_ds[i]["label"] for i in range(n_test)])
            n_cands = np.array([test_ds[i]["n_candidates"] for i in range(n_test)])
            random_acc = float(np.mean(1.0 / n_cands))
            results.append({"model": "random", "config": config_name, "acc": random_acc})

            # SRP
            srp = SRPPHATBaseline(mic_positions=MIC_POSITIONS, sample_rate=24414, half_window=20)
            srp_correct = 0
            for i in range(n_test):
                sample = test_ds[i]
                pred = srp.predict(sample)
                if pred == sample["label"]:
                    srp_correct += 1
            srp_acc = srp_correct / n_test
            results.append({"model": "srp", "config": config_name, "acc": srp_acc})
            print(f"  {config_name}: random={random_acc:.3f} srp={srp_acc:.3f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train V33DA baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models: """ + ", ".join(sorted(MODEL_REGISTRY.keys())) + """, random, srp, all

Examples:
  python scripts/train.py --model spatial_scorer --split session
  python scripts/train.py --model accdoa --split heldout_experiment --heldout-experiment all
  python scripts/train.py --model all --split both
""")
    p.add_argument("--model", required=True,
                   help="Model name, 'all' for all learned models, or 'baselines' for random+srp.")
    p.add_argument("--split", choices=["session", "heldout_experiment", "loo", "both"], default="both")
    p.add_argument("--heldout-experiment", default="all")
    p.add_argument("--heldout-bird", default="all")
    p.add_argument("--epochs", type=int, default=None, help="Override default epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override default batch size.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-radio", action="store_true", help="Disable radio telemetry input.")
    p.add_argument("--lambda-loc", type=float, default=0.05)
    p.add_argument("--lambda-adv", type=float, default=0.05)
    p.add_argument("--tag", default="")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model.lower()

    if model_name in ("random", "srp", "baselines"):
        results = run_deterministic_baselines(args)
    elif model_name == "all":
        results = run_deterministic_baselines(args)
        for name in sorted(MODEL_REGISTRY.keys()):
            results.extend(run_split_matrix(name, args))
    elif model_name in MODEL_REGISTRY:
        results = run_split_matrix(model_name, args)
    else:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}, random, srp, baselines, all")
        sys.exit(1)

    # Save results
    tag = f"_{args.tag}" if args.tag else ""
    out_path = args.output_dir / f"{model_name}_{args.split}{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
