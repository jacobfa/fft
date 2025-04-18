from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Local import
# ---------------------------------------------------------------------
from model import SpectreViT, build_spectre_vit

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def accuracy(pred: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Compute top‑k accuracies (returns list of tensors detached to CPU)."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred_top = pred.topk(maxk, dim=1)
        pred_top = pred_top.t()
        correct = pred_top.eq(target.view(1, -1).expand_as(pred_top))
        return [(correct[:k].reshape(-1).float().sum(0) * 100. / target.size(0)) for k in topk]

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def cifar10_loaders(data_root: Path, batch: int, workers: int = 4) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader

# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> float:
    model.eval()
    top1_sum, n_samples = 0.0, 0
    scaler_ctx = torch.amp.autocast if amp else torch.autocast  # type: ignore
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with scaler_ctx(device_type="cuda", enabled=amp):
            logits = model(x)
        top1 = accuracy(logits, y, topk=(1,))[0]
        top1_sum += top1.item() * x.size(0)
        n_samples += x.size(0)
    return top1_sum / n_samples

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device,
                    epoch: int,
                    sched: optim.lr_scheduler._LRScheduler,
                    grad_clip: float = 1.0,
                    log_interval: int = 10,
                    amp: bool = True) -> None:

    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", ncols=120, leave=False)
    for it, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        sched.step()

        if it % log_interval == 0:
            lr = sched.get_last_lr()[0]
            top1 = accuracy(logits, y, topk=(1,))[0]
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{top1:.2f}", lr=f"{lr:.2e}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpectreViT on CIFAR‑10")
    p.add_argument("--data_root", type=Path, default=Path("./data"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=10, help="warm‑up epochs")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=False, help="mixed precision")
    p.add_argument("--save_dir", type=Path, default=Path("./checkpoints"))
    return p.parse_args()

def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = build_spectre_vit().to(device)
    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    train_loader, test_loader = cifar10_loaders(args.data_root, args.batch, args.workers)

    # ----------------------------------------------------------------
    # Optimiser & schedulers
    # ----------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # warm‑up then cosine
    iters_per_epoch = len(train_loader)
    warmup_iters = args.warmup * iters_per_epoch
    total_iters  = args.epochs * iters_per_epoch

    sched_warm = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters)
    sched_cos  = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters)
    scheduler  = SequentialLR(optimizer, schedulers=[sched_warm, sched_cos],
                              milestones=[warmup_iters])

    # ----------------------------------------------------------------
    # Criterion, AMP, misc.
    # ----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion,
                        optimizer, scaler, device, epoch, scheduler,
                        grad_clip=1.0, amp=args.amp)
        val_acc = evaluate(model, test_loader, device, args.amp)
        tqdm.write(f"[Epoch {epoch:03d}] validation top‑1: {val_acc:6.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = args.save_dir / "spectrevit_best.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            tqdm.write(f"  ↳ New best ({best_acc:.2f}%), checkpoint saved to {ckpt_path}")

    tqdm.write(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
