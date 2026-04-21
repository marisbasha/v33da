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

# Models that consume candidate geometry (position and/or head orientation).
GEOMETRY_AWARE_MODELS = {
    "spatial_scorer", "neural_srp", "contrastive", "pairrank",
    "beam_fusion", "seldnet_plus", "seldnet_pp_open",
    "pose_motion", "video_candidate",
    "accdoa", "einv2", "cstformer",
}


def _rotate_unit_vectors(vecs: torch.Tensor, sigma_rad: float) -> torch.Tensor:
    """Apply axis-angle rotations with Gaussian-σ angle to per-row unit vectors."""
    shape = vecs.shape
    flat = vecs.reshape(-1, 3)
    axes = torch.randn_like(flat)
    axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
    angles = torch.randn(flat.shape[0], device=flat.device) * sigma_rad
    cos = angles.cos().unsqueeze(-1)
    sin = angles.sin().unsqueeze(-1)
    dot = (axes * flat).sum(dim=-1, keepdim=True)
    cross = torch.cross(axes, flat, dim=-1)
    rotated = flat * cos + cross * sin + axes * dot * (1 - cos)
    return (rotated / rotated.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(shape)


def _perturb_batch(batch: dict, pos_sigma_norm, orient_sigma_rad: float) -> dict:
    if pos_sigma_norm is not None and "bird_positions" in batch:
        noise = torch.randn_like(batch["bird_positions"]) * pos_sigma_norm
        batch["bird_positions"] = (batch["bird_positions"] + noise).clamp(0.0, 1.0)
    if orient_sigma_rad > 0 and "bird_head_orient" in batch:
        batch["bird_head_orient"] = _rotate_unit_vectors(
            batch["bird_head_orient"], orient_sigma_rad
        )
    return batch


def _parse_sigma_list(spec: str) -> list[float]:
    if not spec:
        return []
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_seldnet(n_birds, **kw):
    from v33da.baselines.seldnet import SELDnet
    return SELDnet(n_mics=5, max_birds=n_birds)


def _build_seldnet_plus(n_birds, use_radio=True, **kw):
    from v33da.models.seldnet_plus import SELDnetPlus
    return SELDnetPlus(n_mics=5, use_radio=use_radio, n_radio_channels=7)


def _build_seldnet_pp_open(n_birds, **kw):
    from v33da.baselines.seldnet_pp_open import SELDnetPPOpen
    return SELDnetPPOpen(n_mics=5)


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
    "seldnet_pp_open":  dict(build=_build_seldnet_pp_open, epochs=8, lr=1e-4, bs=16),
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


def _infer_n_birds(ds) -> int:
    """Infer max candidate count from a dataset by probing a few samples."""
    n = min(len(ds), 64)
    best = 1
    for i in range(n):
        k = int(ds[i]["n_birds"])
        if k > best:
            best = k
    return best


def _set_mode(ds, **flags):
    """Set dataset return_* flags if present (forwards to wrappers too)."""
    for k, v in flags.items():
        attr = f"return_{k}" if not k.startswith("return_") else k
        for target in (ds, getattr(ds, "base", None), getattr(ds, "bases", None)):
            if target is None:
                continue
            if isinstance(target, dict):
                for sub in target.values():
                    if hasattr(sub, attr):
                        setattr(sub, attr, v)
            elif hasattr(target, attr):
                setattr(target, attr, v)


def mutate_dataset_for_model(ds, model_name, args):
    """Apply model-specific dataset return-flag mutations."""
    if model_name == "pose_motion":
        _set_mode(ds, audio=False, accelerometer=False, radio_temporal=False, pose_temporal=True)
    elif model_name == "video_candidate":
        _set_mode(ds, audio=False, accelerometer=False, radio_temporal=False,
                  pose_temporal=False, video_candidate=True)
    elif model_name == "accel_oracle":
        _set_mode(ds, accelerometer=True)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss(model_name, out, batch, args):
    """Compute loss for a model. Returns (total_loss, n_correct)."""
    labels = batch["label_idx"].to(DEVICE)

    if model_name == "pairrank":
        from v33da.models.pairrank import pairrank_loss
        loss, _ = pairrank_loss(out, batch)
        preds = out["logits"].argmax(dim=-1)
        correct = int((preds == labels).sum().item())
        return loss, correct

    if model_name == "accel_oracle":
        from v33da.models.accel_oracle import accel_oracle_loss
        loss = accel_oracle_loss(out, batch)
        preds = out["logits"].argmax(dim=-1)
        correct = int((preds == labels).sum().item())
        return loss, correct

    if model_name in ("accdoa", "einv2", "cstformer"):
        from v33da.models.literature_seld import literature_loss
        lambda_loc = getattr(args, "lambda_loc", 0.05)
        loss, _ = literature_loss(model_name, out, batch, lambda_loc=lambda_loc)
        preds = out["logits"].argmax(dim=-1)
        correct = int((preds == labels).sum().item())
        return loss, correct

    # Default: CE + optional localization MSE
    logits = out["logits"]
    ls = 0.1 if model_name == "spatial_scorer" else 0.0
    loss = F.cross_entropy(logits, labels, label_smoothing=ls)

    if "pred_position" in out and "label_position" in batch:
        tgt = batch["label_position"].to(DEVICE)
        lambda_loc = getattr(args, "lambda_loc", 0.05)
        loss = loss + lambda_loc * F.mse_loss(out["pred_position"], tgt)

    preds = logits.argmax(dim=-1)
    correct = int((preds == labels).sum().item())
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

    build_kw = dict(n_birds=_infer_n_birds(train_ds),
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
            train_loss += loss.item() * batch["label_idx"].size(0)
            train_correct += correct
            train_total += batch["label_idx"].size(0)
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
                val_total += batch["label_idx"].size(0)

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

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": best_acc, "n_params": n_params}, model


def evaluate_on_loader(model, model_name, loader, args,
                       perturb_pos_cm: float = 0.0,
                       perturb_orient_deg: float = 0.0) -> float:
    """Return attribution accuracy on a loader, optionally under perturbed geometry."""
    model.eval()
    bound_range = torch.tensor(BOUNDS[:, 1] - BOUNDS[:, 0], device=DEVICE, dtype=torch.float32)
    pos_sigma_norm = (perturb_pos_cm * 10.0) / bound_range if perturb_pos_cm > 0 else None
    orient_sigma_rad = float(np.deg2rad(perturb_orient_deg)) if perturb_orient_deg > 0 else 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)
            batch = _perturb_batch(batch, pos_sigma_norm, orient_sigma_rad)
            out = model(batch)
            _, c = compute_loss(model_name, out, batch, args)
            correct += c
            total += batch["label_idx"].size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Split matrix runner
# ---------------------------------------------------------------------------

def run_split_matrix(model_name, args):
    """Run model across the requested split matrix with multiple seeds."""
    seeds = [42, 123, 456]
    all_results = []

    pos_sigmas = _parse_sigma_list(getattr(args, "perturb_pos_cm", ""))
    orient_sigmas = _parse_sigma_list(getattr(args, "perturb_orient_deg", ""))
    run_perturb = (pos_sigmas or orient_sigmas) and model_name in GEOMETRY_AWARE_MODELS

    def run_one_config(train_ds, val_ds, test_ds, config_name, extra_kw=None):
        print(f"\n{'='*60}")
        print(f"  {model_name} | {config_name}")
        print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            result, trained_model = train_one_run(model_name, train_ds, val_ds, args, seed, extra_kw)
            if run_perturb:
                mutate_dataset_for_model(test_ds, model_name, args)
                test_loader = DataLoader(
                    test_ds, batch_size=MODEL_REGISTRY[model_name].get("bs", 16),
                    shuffle=False, num_workers=args.num_workers,
                    collate_fn=v33da_collate_fn, pin_memory=True,
                )
                result["clean_test_acc"] = evaluate_on_loader(trained_model, model_name, test_loader, args)
                print(f"    clean test_acc={result['clean_test_acc']:.3f}")
                result["perturb_pos_cm"] = {}
                for sigma in pos_sigmas:
                    torch.manual_seed(seed)
                    acc = evaluate_on_loader(trained_model, model_name, test_loader, args,
                                             perturb_pos_cm=sigma)
                    result["perturb_pos_cm"][f"{sigma}"] = acc
                    print(f"    perturb pos σ={sigma}cm: acc={acc:.3f}")
                result["perturb_orient_deg"] = {}
                for sigma in orient_sigmas:
                    torch.manual_seed(seed)
                    acc = evaluate_on_loader(trained_model, model_name, test_loader, args,
                                             perturb_orient_deg=sigma)
                    result["perturb_orient_deg"][f"{sigma}"] = acc
                    print(f"    perturb orient σ={sigma}°: acc={acc:.3f}")
                del test_loader
            del trained_model
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
        train_ds, val_ds, test_ds, _ = load_split_triplet("session")
        run_one_config(train_ds, val_ds, test_ds, "session")

    if split in ("heldout_experiment", "both"):
        for exp_name in iter_requested_experiments(args.heldout_experiment):
            train_ds, val_ds, test_ds, _ = load_split_triplet("heldout_experiment", exp_name)
            extra_kw = {}
            if model_name == "spatial_scorer":
                extra_kw["n_experiments"] = len(dataset_experiment_counts(train_ds))
            run_one_config(train_ds, val_ds, test_ds,
                           f"heldout_{short_experiment_name(exp_name)}", extra_kw)

    if split in ("loo", "leave_one_bird_out", "both"):
        for exp_name in iter_requested_experiments(args.heldout_experiment):
            for bird_idx in iter_requested_birds(exp_name, args.heldout_bird):
                train_ds, val_ds, test_ds, _ = load_split_triplet(
                    "leave_one_bird_out", exp_name, bird_idx,
                )
                run_one_config(
                    train_ds, val_ds, test_ds,
                    f"loo_{short_experiment_name(exp_name)}_bird{bird_idx}",
                )

    return all_results


# ---------------------------------------------------------------------------
# Non-neural baselines
# ---------------------------------------------------------------------------

def run_deterministic_baselines(args):
    """Run Random, SRP baselines (no training)."""
    from v33da.baselines.srp_phat import SRPPHATBaseline

    results = []
    configs = []
    if args.split in ("session", "both"):
        _, _, te, _ = load_split_triplet("session")
        configs.append(("session", te))
    if args.split in ("heldout_experiment", "both"):
        for exp in iter_requested_experiments(args.heldout_experiment):
            _, _, te, _ = load_split_triplet("heldout_experiment", exp)
            configs.append((f"heldout_{short_experiment_name(exp)}", te))
    if args.split in ("loo", "leave_one_bird_out", "both"):
        for exp in iter_requested_experiments(args.heldout_experiment):
            for bird in iter_requested_birds(exp, args.heldout_bird):
                _, _, te, _ = load_split_triplet("leave_one_bird_out", exp, bird)
                configs.append((f"loo_{short_experiment_name(exp)}_bird{bird}", te))

    for config_name, test_ds in configs:
        n_test = len(test_ds)
        n_cands = np.array([test_ds[i]["n_birds"] for i in range(n_test)])
        random_acc = float(np.mean(1.0 / np.maximum(n_cands, 1)))
        results.append({"model": "random", "config": config_name, "acc": random_acc})

        srp = SRPPHATBaseline(mic_positions=MIC_POSITIONS, sample_rate=24414, half_window=20)
        srp_correct = 0
        for i in range(n_test):
            sample = test_ds[i]
            pred = srp.predict(sample)
            if pred == sample["label_idx"]:
                srp_correct += 1
        srp_acc = srp_correct / max(n_test, 1)
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
    p.add_argument("--perturb-pos-cm", default="",
                   help="Comma-separated σ values (cm) for Gaussian 3D position perturbation at test time.")
    p.add_argument("--perturb-orient-deg", default="",
                   help="Comma-separated σ values (deg) for head-orientation perturbation at test time.")
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
