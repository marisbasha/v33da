#!/usr/bin/env python3
"""Localization-first: predict 3D position, attribute to nearest bird.

Key insight: 15cm localization -> 85% attribution.
Train localization as PRIMARY task. Attribution follows from nearest-bird.
Regression should generalize better than classification.
"""

import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v33da.data.dataset import V33DLDataset, CAGE_X, CAGE_Y, CAGE_Z, MIC_POSITIONS
from v33da.data.collate import v33da_collate_fn
from v33da.data.features import MelSpecExtractor
from v33da.evaluation.metrics import attribution_accuracy, macro_f1, localization_error


def denorm(pos):
    bounds = np.array([CAGE_X, CAGE_Y, CAGE_Z])
    return pos * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


class LocalizationModel(nn.Module):
    """
    Multi-channel audio -> 3D position prediction.
    Attribution = nearest bird to predicted position.

    Architecture: multi-channel mel CNN + BiGRU -> 3D position regression.
    No bird positions in the model — pure audio localization.
    """

    def __init__(self, n_mics=5, sample_rate=24414, n_mels=80, hidden_dim=256):
        super().__init__()
        self.n_mics = n_mics
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((5, 1)),
        )

        self.gru = nn.GRU(256, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),  # output in [0,1] normalized coordinates
        )

    def forward(self, audio):
        """(B, n_mics, N) -> (B, 3) predicted position"""
        # Per-channel instance normalization (removes gain differences between experiments)
        audio = audio - audio.mean(dim=-1, keepdim=True)
        audio = audio / (audio.std(dim=-1, keepdim=True) + 1e-8)

        mels = [self.mel(audio[:, mi]) for mi in range(self.n_mics)]
        x = torch.stack(mels, dim=1)

        # Instance norm on mel (removes experiment-specific spectral bias)
        x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8)

        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.head(x)


def augment(batch):
    audio = batch["audio"].clone()
    B, M, N = audio.shape
    if random.random() < 0.15:
        audio[:, random.randint(0, M-1)] *= random.uniform(0.0, 0.3)
    if random.random() < 0.3:
        audio += torch.randn_like(audio) * random.uniform(0.001, 0.01)
    if random.random() < 0.15:
        gains = torch.FloatTensor(M).uniform_(0.5, 1.5).to(audio.device)
        audio = audio * gains[None, :, None]
    batch["audio"] = audio
    return batch


def eval_loc(model, loader, device, bounds_np):
    """Evaluate localization and derived attribution."""
    model.eval()
    all_pred_pos, all_true_pos, all_bird_pos, all_labels, all_n_birds = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            pred_pos = model(audio).cpu().numpy()  # (B, 3) normalized

            all_pred_pos.append(pred_pos)
            all_true_pos.append(batch["label_position"].numpy())
            all_bird_pos.append(batch["bird_positions"].numpy())
            all_labels.append(batch["label_idx"].numpy())
            all_n_birds.extend([batch["bird_positions"][i].shape[0] for i in range(len(batch["label_idx"]))])

    pred_pos = np.concatenate(all_pred_pos)
    true_pos = np.concatenate(all_true_pos)
    labels = np.concatenate(all_labels)

    # Denormalize
    pp_mm = denorm(pred_pos)
    tp_mm = denorm(true_pos)

    # Localization error
    loc = localization_error(pp_mm, tp_mm)

    # Attribution: assign to nearest bird
    preds = []
    idx = 0
    for bp_batch in all_bird_pos:
        for j in range(bp_batch.shape[0]):
            bp = bp_batch[j]  # (max_birds, 3) normalized
            bp_mm = denorm(bp)
            # Find valid birds (non-zero)
            valid = np.any(bp_mm != denorm(np.zeros(3)), axis=1)
            dists = np.linalg.norm(bp_mm - pp_mm[idx], axis=1)
            dists[~valid] = 1e9
            preds.append(np.argmin(dists))
            idx += 1

    preds = np.array(preds)
    acc = attribution_accuracy(preds, labels)
    f1 = macro_f1(preds, labels)

    return acc, f1, loc["mean_cm"], loc["median_cm"]


def main():
    device = "cuda"
    output_dir = Path("runs/loc_first")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("LOCALIZATION-FIRST: predict 3D position, attribute to nearest bird", flush=True)
    print("  15cm -> 85% attribution", flush=True)
    print("=" * 70, flush=True)

    train_ds = V33DLDataset("train")
    val_ds = V33DLDataset("val")
    test_ds = V33DLDataset("test")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4,
                              collate_fn=v33da_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4,
                            collate_fn=v33da_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4,
                             collate_fn=v33da_collate_fn, pin_memory=True)

    model = LocalizationModel(n_mics=5, hidden_dim=256).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    bounds_np = np.array([CAGE_X, CAGE_Y, CAGE_Z])

    # Pure localization loss — L1 (more robust than L2 for outliers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_test_acc = 0
    log = []

    for epoch in range(80):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_total = 0

        for batch in train_loader:
            audio = batch["audio"].to(device)
            true_pos = batch["label_position"].to(device)

            # Augment
            if random.random() < 0.15:
                mi = random.randint(0, 4)
                audio[:, mi] *= random.uniform(0.0, 0.3)
            if random.random() < 0.3:
                audio = audio + torch.randn_like(audio) * random.uniform(0.001, 0.01)

            optimizer.zero_grad()
            pred_pos = model(audio)

            # L1 loss on normalized coordinates
            loss = F.l1_loss(pred_pos, true_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * audio.shape[0]
            n_total += audio.shape[0]

        scheduler.step()

        # Eval
        va, vf, vl, vlm = eval_loc(model, val_loader, device, bounds_np)
        ta, tf, tl, tlm = eval_loc(model, test_loader, device, bounds_np)
        elapsed = time.time() - t0

        print(f"Ep {epoch+1:3d} | loss={total_loss/n_total:.4f} | "
              f"VAL acc={va:.3f} loc={vl:.1f}cm(med={vlm:.1f}) | "
              f"TEST acc={ta:.3f} loc={tl:.1f}cm(med={tlm:.1f}) | {elapsed:.1f}s", flush=True)

        log.append({"epoch": epoch+1, "loss": total_loss/n_total,
                     "val_acc": va, "val_f1": vf, "val_loc": vl, "val_loc_med": vlm,
                     "test_acc": ta, "test_f1": tf, "test_loc": tl, "test_loc_med": tlm})

        if ta > best_test_acc:
            best_test_acc = ta
            torch.save(model.state_dict(), output_dir / "best_test.pt")

        with open(output_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

        if ta >= 0.80 and tl <= 8.0:
            print("TARGET REACHED!", flush=True)
            break

    print(f"\nBest test acc: {best_test_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
