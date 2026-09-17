"""DP-SGD training loop using Opacus.

Provides:
- `train_dpsgd()`: end-to-end DP-SGD training with per-sample gradient clipping
  and Gaussian noise addition (via Opacus).
- Returns the trained model, training metrics, and Opacus privacy engine
  (for extracting spent privacy budget).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.accounting import compare_accountants

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Container for DP-SGD training outputs."""

    model: nn.Module
    privacy_engine: PrivacyEngine | None
    train_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    # epsilon_history: method → list of ε values at each eval epoch
    epsilon_history: dict[str, list[float]] = field(default_factory=dict)
    eps_audit_history: list[float] = field(default_factory=list)
    audit_epochs: list[int] = field(default_factory=list)
    # Which epochs were eval epochs (for x-axis alignment)
    eval_epochs: list[int] = field(default_factory=list)
    total_steps: int = 0


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def _append_progress(path: Path, row: dict) -> None:
    """Append a single JSON line to the progress file (flush immediately)."""
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def train_dpsgd(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 10,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    device: str = "cpu",
    max_physical_batch_size: int = 128,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    epoch_callback: Any | None = None,
    eval_every_n_epochs: int = 10,
    accounting_methods: list[str] | None = None,
    disable_dp: bool = False,
    output_dir: Path | None = None,
    lr_schedule: list[list] | None = None,
) -> TrainingResult:
    """Train a model with DP-SGD via Opacus.

    Parameters
    ----------
    model : nn.Module
        Must be Opacus-compatible (no in-place ops, no BatchNorm).
    train_loader : DataLoader
        Training data (Poisson sampling is handled by Opacus).
    val_loader : DataLoader, optional
        Validation data for accuracy tracking.
    epochs : int
        Number of training epochs.
    learning_rate : float
        SGD learning rate.
    max_grad_norm : float
        Per-sample gradient clipping bound C.
    noise_multiplier : float
        Noise multiplier σ (noise std = σ · C).
    delta : float
        Target δ for (ε, δ)-DP accounting.
    device : str
        Torch device.
    max_physical_batch_size : int
        Opacus physical batch size for memory management.
    epoch_callback : callable, optional
        Called as ``epoch_callback(state_dict, epoch)`` → float.
        Receives a CPU state_dict snapshot. Return value (e.g. ε_audit)
        is stored in ``result.eps_audit_history``.
    eval_every_n_epochs : int
        Run validation, accounting, and audit every N epochs
        (and always epoch 1 and the last epoch). Default 10.
    accounting_methods : list of str, optional
        Which ε accountants to run (from: rdp, zcdp, pld, moments).
        Default: ["rdp", "zcdp", "pld"].
    disable_dp : bool
        If True, train WITHOUT Opacus (no noise, no per-sample clipping,
        no privacy accounting).  Still applies standard gradient clipping
        via torch.nn.utils.clip_grad_norm_.  Useful as a non-DP baseline.
    output_dir : Path, optional
        If provided, intermediate results are streamed to
        ``output_dir/progress.jsonl`` (one JSON line per eval epoch)
        so training can be monitored without waiting for completion.

    Returns
    -------
    TrainingResult
        Trained model, privacy engine, and training metrics.
    """
    if accounting_methods is None:
        accounting_methods = ["rdp", "zcdp", "pld"]

    model = model.to(device)
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd_no_momentum":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Use 'adam', 'sgd', or 'sgd_no_momentum'."
        )
    logger.info(f"Optimizer: {optimizer_name} (lr={learning_rate}, wd={weight_decay})")
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    sampling_rate = 0.0

    if disable_dp:
        # --- Non-DP mode: plain PyTorch, gradient clipping only ---
        logger.info(f"*** NON-DP MODE: no noise, clipping C={max_grad_norm} ***")
        sampling_rate = 0.0
    else:
        # Attach Opacus privacy engine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        # Compute sampling rate from the DPDataLoader
        # Opacus DPDataLoader exposes .sample_rate directly (Poisson sampling)
        sampling_rate = getattr(train_loader, "sample_rate", None)
        if sampling_rate is None:
            # Fallback: compute from batch_size / dataset_size
            dataset_size = len(train_loader.dataset)
            batch_size = train_loader.batch_size or max_physical_batch_size
            sampling_rate = batch_size / dataset_size
        dataset_size = len(train_loader.dataset)
        # Logical steps per epoch = number of optimizer updates (noise additions)
        # With Poisson sampling, Opacus uses int(1/q) batches per epoch
        logical_steps_per_epoch = max(1, int(1.0 / sampling_rate))
        logger.info(f"Accounting: σ={noise_multiplier}, q={sampling_rate:.4f}, N={dataset_size}")
        logger.info(
            f"  logical_steps_per_epoch={logical_steps_per_epoch} (physical_batch={max_physical_batch_size})"
        )
        logger.info(f"Accounting methods: {accounting_methods}")

    result = TrainingResult(model=model, privacy_engine=privacy_engine)
    if not disable_dp:
        for method in accounting_methods:
            result.epsilon_history[method] = []

    # Background audit thread state
    _audit_lock = threading.Lock()
    _audit_thread: threading.Thread | None = None
    _pending_audit_results: dict[int, float] = {}  # epoch → eps_audit

    def _run_audit_in_background(state_dict: OrderedDict, epoch: int) -> None:
        """Run audit callback on a CPU copy of model weights."""
        eps_audit = epoch_callback(state_dict, epoch)
        with _audit_lock:
            _pending_audit_results[epoch] = eps_audit

    total_steps = 0
    progress_path = output_dir / "progress.jsonl" if output_dir else None

    # Pre-sort LR schedule milestones: [(epoch, lr), ...]
    _lr_milestones: list[tuple[int, float]] = []
    if lr_schedule:
        _lr_milestones = sorted((int(e), float(lr)) for e, lr in lr_schedule)
        logger.info(f"LR schedule: {_lr_milestones}")

    for epoch in range(1, epochs + 1):
        # Apply LR schedule: use the LR from the most recent milestone <= current epoch
        if _lr_milestones:
            new_lr = learning_rate
            for milestone_epoch, milestone_lr in _lr_milestones:
                if epoch >= milestone_epoch:
                    new_lr = milestone_lr
                else:
                    break
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if disable_dp:
            # --- Non-DP training loop: plain DataLoader + gradient clipping ---
            pbar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch}/{epochs}",
                leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}",
                ncols=100,
            )
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                total_steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        else:
            # --- DP-SGD training loop via Opacus BatchMemoryManager ---
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_loader:
                pbar = tqdm(
                    memory_safe_loader,
                    desc=f"  Epoch {epoch}/{epochs}",
                    leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}",
                    ncols=100,
                )
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    total_steps += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        result.train_losses.append(avg_loss)

        # Only do expensive evaluation/accounting/auditing periodically
        is_eval_epoch = (epoch % eval_every_n_epochs == 0) or (epoch == epochs) or (epoch == 1)

        if not is_eval_epoch:
            # Lightweight progress line for non-eval epochs
            if progress_path is not None:
                _append_progress(progress_path, {"epoch": epoch, "loss": avg_loss})
            continue

        result.eval_epochs.append(epoch)

        # Compute ε via all accounting methods (skip in non-DP mode or σ≈0)
        eps_strs = []
        epoch_epsilons: dict[str, float] = {}
        if not disable_dp and noise_multiplier > 0.001:
            # Use LOGICAL steps (optimizer updates) for accounting, not
            # physical batch iterations.  BatchMemoryManager splits each
            # logical batch into physical_batch chunks, so total_steps
            # overcounts by batch_size/max_physical_batch_size.
            num_logical_steps = epoch * logical_steps_per_epoch
            eps_results = compare_accountants(
                noise_multiplier=noise_multiplier,
                sampling_rate=sampling_rate,
                num_steps=num_logical_steps,
                delta=delta,
                methods=accounting_methods,
            )
            for method in accounting_methods:
                if method in eps_results:
                    eps_val = eps_results[method]["epsilon"]
                    result.epsilon_history[method].append(eps_val)
                    eps_strs.append(f"ε_{method}={eps_val:.2f}")
                    epoch_epsilons[method] = eps_val
        elif disable_dp or noise_multiplier <= 0.001:
            eps_strs.append("ε=∞ (no/minimal noise)")

        # Validation accuracy
        val_acc = 0.0
        if val_loader is not None:
            val_acc = _evaluate(model, val_loader, device)
            result.val_accuracies.append(val_acc)

        # Audit callback — runs in background thread on CPU
        if epoch_callback is not None:
            # Collect any finished background audit results
            with _audit_lock:
                for ep in sorted(_pending_audit_results.keys()):
                    result.audit_epochs.append(ep)
                    result.eps_audit_history.append(_pending_audit_results[ep])
                _pending_audit_results.clear()

            # Skip if previous thread still running (never block training)
            if _audit_thread is not None and _audit_thread.is_alive():
                logger.debug(f"  Skipping audit at epoch {epoch} (previous still running)")
            else:
                # Snapshot weights (strip Opacus prefix) and launch new audit thread
                state_dict = OrderedDict(
                    (k.removeprefix("_module."), v.cpu().clone())
                    for k, v in model.state_dict().items()
                )
                _audit_thread = threading.Thread(
                    target=_run_audit_in_background,
                    args=(state_dict, epoch),
                    daemon=True,
                )
                _audit_thread.start()

        audit_str = (
            f"  ε_audit={result.eps_audit_history[-1]:.4f}" if result.eps_audit_history else ""
        )
        step_str = f"steps={total_steps}"
        if not disable_dp and noise_multiplier > 0.001:
            step_str = f"logical_steps={epoch * logical_steps_per_epoch} (physical={total_steps})"
        logger.info(
            f"  ✓ Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f}  {step_str}  "
            f"{', '.join(eps_strs)}{audit_str}"
        )

        # Stream intermediate results to progress file
        if progress_path is not None:
            row: dict[str, Any] = {
                "epoch": epoch,
                "loss": avg_loss,
                "val_acc": val_acc,
                "total_steps": total_steps,
            }
            if epoch_epsilons:
                row["epsilons"] = epoch_epsilons
            if result.eps_audit_history:
                row["eps_audit"] = result.eps_audit_history[-1]
                row["eps_audit_epoch"] = result.audit_epochs[-1]
            _append_progress(progress_path, row)

    # Store logical steps for accounting correctness
    if not disable_dp and noise_multiplier > 0.001:
        result.total_steps = epochs * logical_steps_per_epoch
    else:
        result.total_steps = total_steps

    # Wait for final audit thread to finish
    if epoch_callback is not None and _audit_thread is not None:
        _audit_thread.join()
        with _audit_lock:
            for ep in sorted(_pending_audit_results.keys()):
                result.audit_epochs.append(ep)
                result.eps_audit_history.append(_pending_audit_results[ep])
            _pending_audit_results.clear()

    return result
