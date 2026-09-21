"""Training loop for MCAFF (Multi-Head Cross-Attention Feature Fusion)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import classification_report_dict, compute_metrics


def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency weights cho imbalanced G1/G2/G3."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        device: str | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.cfg = cfg
        train_cfg = cfg["training"]
        self.device = device or train_cfg.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = "cpu"

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.group_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.family_loss_fn = nn.CrossEntropyLoss()
        self.family_weight = 0.3 if cfg["model"].get("family_classifier") else 0.0
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        self.model.train()
        batch = self._to_device(batch)
        out = self.model(
            batch["static_x"],
            batch["behavior_x"],
            batch["graph_x"],
        )
        loss = self.group_loss_fn(out["logits_group"], batch["y_group"])
        metrics = {"loss_group": loss.item()}

        if "logits_family" in out:
            loss_family = self.family_loss_fn(out["logits_family"], batch["y_family"])
            loss = loss + self.family_weight * loss_family
            metrics["loss_family"] = loss_family.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        metrics["loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        all_group_true, all_group_pred = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(
                batch["static_x"],
                batch["behavior_x"],
                batch["graph_x"],
            )
            loss = self.group_loss_fn(out["logits_group"], batch["y_group"])
            total_loss += loss.item()
            n_batches += 1

            preds = out["logits_group"].argmax(dim=-1).cpu().numpy()
            labels = batch["y_group"].cpu().numpy()
            all_group_pred.extend(preds.tolist())
            all_group_true.extend(labels.tolist())

        y_true = np.array(all_group_true)
        y_pred = np.array(all_group_pred)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["report"] = classification_report_dict(y_true, y_pred)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        epochs = epochs or self.cfg["training"]["epochs"]
        patience = self.cfg["training"]["early_stopping_patience"]
        history: list[dict] = []

        if len(val_loader.dataset) == 0:
            print("WARNING: empty validation set — using train metrics for early stopping")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                _, metrics = self._step(batch)
                epoch_loss += metrics["loss"]
                n_batches += 1
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            eval_loader = val_loader if len(val_loader.dataset) > 0 else train_loader
            val_metrics = self.evaluate(eval_loader)
            val_f1 = val_metrics["f1_macro"]
            self.scheduler.step(val_f1)

            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                **{k: v for k, v in val_metrics.items() if k != "report"},
            }
            history.append(record)
            print(
                f"  val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                f"f1={val_f1:.4f}"
            )

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return {"history": history, "best_f1": self.best_f1}

    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "best_f1": self.best_f1,
                "config": self.cfg,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out
