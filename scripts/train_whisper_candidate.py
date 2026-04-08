#!/usr/bin/env python3
"""Train a frozen-Whisper candidate scorer across the V33DA split matrix."""

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v33da.data.collate import v33da_collate_fn
from v33da.data.dataset import CAGE_X, CAGE_Y, CAGE_Z, V33DLDataset
from v33da.data.splits import (
    dataset_experiment_counts,
    iter_requested_birds,
    iter_requested_experiments,
    load_split_triplet,
    short_experiment_name,
)
from v33da.evaluation.metrics import attribution_accuracy, localization_error, macro_f1
from v33da.models.encoders.whisper import WhisperEncoder


BOUNDS = np.array([CAGE_X, CAGE_Y, CAGE_Z], dtype=np.float32)
SEEDS = [42, 123, 456]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_ROOT = Path("cache/whisper_embeddings")
BASE_SPLITS = ("train", "val", "test")
_BASE_EMBED_INDEX: dict[tuple[str, int], dict[str, torch.Tensor]] = {}


class WhisperCandidateScorer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        model_size: str = "base",
        use_radio: bool = True,
        use_precomputed_audio: bool = False,
    ):
        super().__init__()
        self.use_radio = use_radio
        self.audio_encoder = None
        if not use_precomputed_audio:
            self.audio_encoder = WhisperEncoder(d_out=d_model, model_size=model_size, n_mics=5)
        bird_in = 3 + 3 + (12 if use_radio else 0)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(bird_in, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )
        self.mic_weight = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, batch):
        if "whisper_emb" in batch:
            audio_emb = batch["whisper_emb"]
        elif self.audio_encoder is not None:
            audio_emb = self.audio_encoder.forward_multi_mic_tokens(batch["audio"])
        else:
            raise KeyError("Batch is missing whisper_emb and no audio encoder is attached.")
        pos = batch["bird_positions"]
        head = batch.get("bird_head_orient", torch.zeros_like(pos))
        if self.use_radio:
            radio = batch.get("bird_radio", torch.zeros(pos.shape[0], pos.shape[1], 12, device=pos.device))
            cand_feat = torch.cat([pos, head, radio], dim=-1)
        else:
            cand_feat = torch.cat([pos, head], dim=-1)
        cand_emb = self.candidate_encoder(cand_feat)
        audio_exp = audio_emb.unsqueeze(2).expand(-1, -1, cand_emb.shape[1], -1)
        cand_exp = cand_emb.unsqueeze(1).expand(-1, audio_emb.shape[1], -1, -1)
        interaction = audio_exp * cand_exp
        mic_logits = self.scorer(torch.cat([audio_exp, cand_exp, interaction], dim=-1)).squeeze(-1)
        mic_weights = torch.softmax(self.mic_weight(audio_emb).squeeze(-1), dim=-1)
        logits = (mic_logits * mic_weights.unsqueeze(-1)).sum(dim=1)
        if "bird_mask" in batch:
            logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))
        attn = F.softmax(logits, dim=-1)
        pred_position = (attn.unsqueeze(-1) * pos).sum(dim=1)
        return {"logits": logits, "pred_position": pred_position, "attn_weights": attn}


def parse_args():
    p = argparse.ArgumentParser(description="Train a frozen-Whisper candidate scorer on V33DA.")
    p.add_argument(
        "--split",
        choices=["session", "unseen_bird", "heldout_experiment", "leave_one_bird_out", "both"],
        default="both",
    )
    p.add_argument("--heldout-experiment", default="all")
    p.add_argument("--heldout-bird", default="all")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--model-size", default="base")
    p.add_argument("--no-radio", action="store_true")
    p.add_argument("--lambda-loc", type=float, default=0.05)
    p.add_argument("--tag", default="")
    return p.parse_args()


class LimitedDataset:
    def __init__(self, base, max_samples: int):
        self.base = base
        self.max_samples = min(len(base), max_samples)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        return self.base[idx]


def maybe_limit(ds, max_samples: int):
    if max_samples <= 0:
        return ds
    return LimitedDataset(ds, max_samples)


class WhisperEmbeddingDataset(Dataset):
    def __init__(self, base, embeddings: torch.Tensor):
        self.base = base
        self.embeddings = embeddings

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["whisper_emb"] = self.embeddings[idx]
        return sample


def _set_dataset_flags(ds, *, audio: bool, accelerometer: bool, radio_temporal: bool):
    cur = ds
    while cur is not None:
        if hasattr(cur, "return_audio"):
            cur.return_audio = audio
        if hasattr(cur, "return_accelerometer"):
            cur.return_accelerometer = accelerometer
        if hasattr(cur, "return_radio_temporal"):
            cur.return_radio_temporal = radio_temporal
        cur = getattr(cur, "base", None)


def _dataset_sample_ids(ds):
    if hasattr(ds, "_ids"):
        return list(ds._ids)
    if hasattr(ds, "indices") and hasattr(ds, "bases"):
        ids = []
        for split, i in ds.indices:
            base = ds.bases[split]
            ids.append(base._ids[i])
        return ids
    base = getattr(ds, "base", None)
    if base is not None:
        ids = _dataset_sample_ids(base)
        if hasattr(ds, "max_samples"):
            return ids[: ds.max_samples]
        return ids
    raise AttributeError("Unable to resolve sample ids for dataset cache verification.")


def _base_cache_path(model_size: str, base_split: str, d_model: int) -> Path:
    return CACHE_ROOT / f"whisper_{model_size}_d{d_model}_permic_base_{base_split}.pt"


def _build_base_cache(base_split: str, model_size: str, d_model: int, batch_size: int, num_workers: int):
    cache_path = _base_cache_path(model_size, base_split, d_model)
    ds = V33DLDataset(base_split)
    expected_ids = list(ds._ids)
    payload = None
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("sample_ids") != expected_ids:
            print(f"  base cache mismatch for {cache_path.name}; rebuilding", flush=True)
            payload = None

    if payload is None:
        print(f"  caching Whisper embeddings once for base split: {cache_path.name}", flush=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        ds.return_audio = True
        ds.return_accelerometer = False
        ds.return_radio_temporal = False
        encoder = WhisperEncoder(d_out=d_model, model_size=model_size, n_mics=5).to(DEVICE)
        encoder.eval()
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=v33da_collate_fn,
            pin_memory=DEVICE.startswith("cuda"),
        )
        all_embeddings = []
        sample_ids = []
        for i, batch in enumerate(loader, start=1):
            audio = batch["audio"].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                emb = encoder.forward_multi_mic_tokens(audio).cpu()
            all_embeddings.append(emb)
            sample_ids.extend(batch["sample_ids"])
            if i % 20 == 0 or i == len(loader):
                print(f"    cache {base_split}: {i}/{len(loader)} batches", flush=True)
        payload = {
            "sample_ids": sample_ids,
            "embeddings": torch.cat(all_embeddings, dim=0),
        }
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(cache_path)
        del encoder
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
    return payload


def _get_base_embedding_index(model_size: str, d_model: int, batch_size: int, num_workers: int):
    key = (model_size, d_model)
    if key in _BASE_EMBED_INDEX:
        return _BASE_EMBED_INDEX[key]

    index = {}
    for base_split in BASE_SPLITS:
        payload = _build_base_cache(base_split, model_size, d_model, batch_size, num_workers)
        for sample_id, emb in zip(payload["sample_ids"], payload["embeddings"]):
            index[sample_id] = emb
    _BASE_EMBED_INDEX[key] = index
    return index


def prepare_whisper_dataset(
    ds,
    *,
    split_key: str,
    role: str,
    model_size: str,
    d_model: int,
    batch_size: int,
    num_workers: int,
):
    expected_ids = _dataset_sample_ids(ds)
    index = _get_base_embedding_index(model_size, d_model, batch_size, num_workers)
    missing = [sample_id for sample_id in expected_ids if sample_id not in index]
    if missing:
        raise KeyError(f"Missing {len(missing)} Whisper embeddings for split {split_key}/{role}.")
    _set_dataset_flags(ds, audio=False, accelerometer=False, radio_temporal=False)
    embeddings = torch.stack([index[sample_id] for sample_id in expected_ids], dim=0)
    return WhisperEmbeddingDataset(ds, embeddings)


def denorm(pos):
    return pos * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]


def evaluate(model, loader):
    model.eval()
    preds, labels, pred_pos, true_pos = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
            labels.append(batch["label_idx"].cpu().numpy())
            pred_pos.append(out["pred_position"].cpu().numpy())
            true_pos.append(batch["label_position"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    pred_pos = denorm(np.concatenate(pred_pos))
    true_pos = denorm(np.concatenate(true_pos))
    loc = localization_error(pred_pos, true_pos)
    return {"acc": attribution_accuracy(preds, labels), "f1": macro_f1(preds, labels), "loc_cm": loc["mean_cm"]}


def train_once(train_loader, val_loader, test_loader, seed, model_size, use_radio, epochs, lambda_loc):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = WhisperCandidateScorer(
        model_size=model_size,
        use_radio=use_radio,
        use_precomputed_audio=True,
    ).to(DEVICE)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best_val = -1.0
    best = None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_correct = 0
        train_total = 0
        losses = []
        for batch in train_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch)
            ce = F.cross_entropy(out["logits"], batch["label_idx"])
            loc = F.mse_loss(out["pred_position"], batch["label_position"])
            loss = ce + lambda_loc * loc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            train_correct += (out["logits"].argmax(dim=-1) == batch["label_idx"]).sum().item()
            train_total += batch["label_idx"].size(0)
        sched.step()

        val = evaluate(model, val_loader)
        if val["acc"] > best_val:
            best_val = val["acc"]
            best = {"val": val, "test": evaluate(model, test_loader), "epoch": epoch + 1}

        print(
            f"    ep{epoch+1}: train={train_correct/max(train_total, 1):.3f} "
            f"loss={np.mean(losses):.3f} val={val['acc']:.3f}/{val['f1']:.3f}/{val['loc_cm']:.1f}cm "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    return best


def main():
    args = parse_args()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=v33da_collate_fn,
        pin_memory=DEVICE.startswith("cuda"),
    )

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
        train_ds, val_ds, test_ds, split_meta = load_split_triplet(split_name, heldout_experiment, heldout_bird)
        header = split_name.upper()
        if split_name == "heldout_experiment":
            header = f"{header} ({short_experiment_name(heldout_experiment)})"
        elif split_name == "leave_one_bird_out":
            header = f"LEAVE_ONE_BIRD_OUT ({short_experiment_name(heldout_experiment)} bird {heldout_bird})"
        print(f"\n{'=' * 72}\n  WHISPER CANDIDATE ({args.model_size}): {header}\n{'=' * 72}", flush=True)

        train_ds = maybe_limit(train_ds, args.max_samples)
        val_ds = maybe_limit(val_ds, args.max_samples)
        test_ds = maybe_limit(test_ds, args.max_samples)
        print(f"  sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
        if split_name == "heldout_experiment":
            print(f"  train experiments: {dataset_experiment_counts(train_ds)}", flush=True)
            print(f"  val experiments:   {dataset_experiment_counts(val_ds)}", flush=True)
            print(f"  test experiments:  {dataset_experiment_counts(test_ds)}", flush=True)
        elif split_name == "leave_one_bird_out":
            print(f"  heldout bird: {heldout_bird}", flush=True)

        train_ds = prepare_whisper_dataset(
            train_ds,
            split_key=split_meta["key"],
            role="train",
            model_size=args.model_size,
            d_model=256,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
        )
        val_ds = prepare_whisper_dataset(
            val_ds,
            split_key=split_meta["key"],
            role="val",
            model_size=args.model_size,
            d_model=256,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
        )
        test_ds = prepare_whisper_dataset(
            test_ds,
            split_key=split_meta["key"],
            role="test",
            model_size=args.model_size,
            d_model=256,
            batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
        )

        split_results = []
        for seed in SEEDS:
            print(f"  seed {seed}", flush=True)
            train_loader = DataLoader(train_ds, shuffle=True, **kw)
            val_loader = DataLoader(val_ds, shuffle=False, **kw)
            test_loader = DataLoader(test_ds, shuffle=False, **kw)
            res = train_once(
                train_loader,
                val_loader,
                test_loader,
                seed=seed,
                model_size=args.model_size,
                use_radio=not args.no_radio,
                epochs=args.epochs,
                lambda_loc=args.lambda_loc,
            )
            split_results.append(res)
            print(
                f"    best: val={res['val']['acc']:.3f} "
                f"test={res['test']['acc']:.3f}/{res['test']['f1']:.3f}/{res['test']['loc_cm']:.1f}cm @ ep{res['epoch']}",
                flush=True,
            )
        all_results[split_meta["key"]] = split_results

    model_tag = "no_radio" if args.no_radio else "radio"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"whisper_candidate_{args.model_size}_{model_tag}{suffix}.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
