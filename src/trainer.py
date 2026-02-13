"""
Training pipeline.

Handles the PyTorch training loop with:
- Early stopping to prevent overfitting
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for training stability
- Best-model checkpointing
- Detailed epoch-level logging
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
) -> tuple[nn.Module, dict]:
    """
    Train a PyTorch model and return the best checkpoint.

    Returns:
        model: model loaded with best weights
        history: dict with train_loss, val_loss lists
    """
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    patience = train_cfg.get("patience", 10)
    sched_factor = train_cfg.get("scheduler_factor", 0.5)
    sched_patience = train_cfg.get("scheduler_patience", 5)
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience
    )
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = np.inf
    best_state = None

    logger.info("Starting training: %d epochs, lr=%.4f, patience=%d", epochs, lr, patience)
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"

        if epoch % 5 == 0 or epoch == 1 or marker:
            logger.info(
                "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.6f | %.1fs%s",
                epoch, epochs, avg_train_loss, avg_val_loss, current_lr, elapsed, marker,
            )

        if early_stopping.step(avg_val_loss):
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    total_time = time.time() - total_start
    logger.info("Training complete in %.1fs. Best val_loss=%.6f", total_time, best_val_loss)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
