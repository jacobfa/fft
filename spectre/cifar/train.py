#!/usr/bin/env python3
"""
Train a Vision Transformer with SPECTRE‑RFFT token mixer on CIFAR‑10
===================================================================

Usage
-----
$ python train.py --epochs 200 --batch-size 128 --lr 5e-4 --data ~/datasets

This script expects `model.py` containing the `ViTSpectreCIFAR10` class in
its parent directory or the Python path. Progress is displayed with `tqdm`.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
#                               hyper‑params
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Train ViT‑Spectre on CIFAR‑10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    return parser.parse_args()


# -----------------------------------------------------------------------------
#                                 training
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, epoch, device):
    model.train()
    running_acc = 0.0
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:3d}", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits.detach(), labels)
        running_loss += loss.item() * images.size(0)
        running_acc += acc * images.size(0)

        progress.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc*100:.2f}%")

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    running_acc = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(logits, labels) * images.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


# -----------------------------------------------------------------------------
#                              main procedure
# -----------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # data augmentation
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # datasets & loaders
    train_ds = datasets.CIFAR10(args.data, train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(args.data, train=False, download=True, transform=test_tfms)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    from model import SpectreViT

    model = SpectreViT().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_ld, criterion, optimizer, epoch, device
        )
        val_loss, val_acc = evaluate(model, test_ld, criterion, device)
        scheduler.step()

        tqdm.write(
            f"Epoch {epoch:3d}: "
            f"train_loss={train_loss:.3f}, train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.3f}, val_acc={val_acc*100:.2f}%"
        )

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = Path(args.save_dir) / "vit_spectre_cifar10.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            tqdm.write(f"✔ Saved new best model to {ckpt_path} (acc={best_acc*100:.2f}%)")

    tqdm.write(f"Training complete. Best top‑1 accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
