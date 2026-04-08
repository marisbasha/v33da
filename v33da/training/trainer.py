"""Training loop with checkpointing and early stopping."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..evaluation.metrics import attribution_accuracy, macro_f1, localization_error
from ..data.dataset import CAGE_X, CAGE_Y, CAGE_Z


def _denormalize_positions(pos_norm):
    """Convert normalized [0,1] positions back to mm."""
    bounds = np.array([CAGE_X, CAGE_Y, CAGE_Z])
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return pos_norm * (hi - lo) + lo


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        n_epochs: int = 50,
        patience: int = 10,
        warmup_epochs: int = 5,
        output_dir: str | Path = "runs/default",
        device: str = "cuda",
        test_loader: DataLoader = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.device = device
        self.n_epochs = n_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr * 0.01,
        )

        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def _warmup_lr(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * factor / max(factor, (epoch) / self.warmup_epochs if epoch > 0 else 1)

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_l2 = 0.0
        n_correct = 0
        n_total = 0

        n_batches = len(self.train_loader)
        for bi, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch)
            losses["loss"].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses["loss"].item() * batch["label_idx"].size(0)
            total_ce += losses["ce_loss"].item() * batch["label_idx"].size(0)
            total_l2 += losses["l2_loss"].item() * batch["label_idx"].size(0)

            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == batch["label_idx"]).sum().item()
            n_total += batch["label_idx"].size(0)

            if (bi + 1) % 50 == 0 or (bi + 1) == n_batches:
                print(f"  batch {bi+1}/{n_batches} loss={losses['loss'].item():.4f}", flush=True)

        return {
            "loss": total_loss / n_total,
            "ce_loss": total_ce / n_total,
            "l2_loss": total_l2 / n_total,
            "accuracy": n_correct / n_total,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """Evaluate on an arbitrary loader (used for test set)."""
        return self._eval_loop(loader)

    def val_epoch(self) -> dict:
        return self._eval_loop(self.val_loader)

    def _eval_loop(self, loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_l2 = 0.0
        all_preds = []
        all_labels = []
        all_pred_pos = []
        all_true_pos = []
        all_dists = []
        n_total = 0

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch)

            B = batch["label_idx"].size(0)
            total_loss += losses["loss"].item() * B
            total_ce += losses["ce_loss"].item() * B
            total_l2 += losses["l2_loss"].item() * B
            n_total += B

            all_preds.append(outputs["logits"].argmax(dim=-1).cpu().numpy())
            all_labels.append(batch["label_idx"].cpu().numpy())
            all_pred_pos.append(outputs["pred_position"].detach().cpu().numpy())
            all_true_pos.append(batch["label_position"].cpu().numpy())
            all_dists.append(batch["min_inter_bird_dist_mm"].cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        pred_pos = np.concatenate(all_pred_pos)
        true_pos = np.concatenate(all_true_pos)

        acc = attribution_accuracy(preds, labels)
        f1 = macro_f1(preds, labels)
        loc = localization_error(_denormalize_positions(pred_pos), _denormalize_positions(true_pos))

        return {
            "loss": total_loss / n_total,
            "ce_loss": total_ce / n_total,
            "l2_loss": total_l2 / n_total,
            "accuracy": acc,
            "macro_f1": f1,
            "loc_error_cm": loc["mean_cm"],
            "loc_median_cm": loc["median_cm"],
            "pct_within_5cm": loc["pct_within_5cm"],
        }

    def fit(self):
        log = []
        for epoch in range(self.n_epochs):
            t0 = time.time()

            # Warmup LR
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * warmup_factor if "initial_lr" in pg else pg["lr"]

            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()

            test_str = ""
            test_metrics = None
            if self.test_loader is not None:
                test_metrics = self.evaluate(self.test_loader)
                test_str = (f" | TEST acc={test_metrics['accuracy']:.3f} "
                           f"f1={test_metrics['macro_f1']:.3f} "
                           f"loc={test_metrics['loc_error_cm']:.1f}cm")

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d}/{self.n_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
                f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['macro_f1']:.3f} "
                f"loc={val_metrics['loc_error_cm']:.1f}cm"
                f"{test_str} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            entry = {"epoch": epoch + 1, "lr": lr, "elapsed": elapsed}
            entry.update({f"train_{k}": v for k, v in train_metrics.items()})
            entry.update({f"val_{k}": v for k, v in val_metrics.items()})
            if test_metrics is not None:
                entry.update({f"test_{k}": v for k, v in test_metrics.items()})
            log.append(entry)

            # Checkpointing
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.output_dir / "best.pt")
            else:
                self.epochs_without_improvement += 1

            # Save latest
            torch.save(self.model.state_dict(), self.output_dir / "latest.pt")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        # Save training log
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"\nBest val accuracy: {self.best_val_acc:.4f}")
        return log
