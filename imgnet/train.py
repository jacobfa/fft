#!/usr/bin/env python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fftnet_vit import FFTNetViT 
from tqdm import tqdm
import threading
import logging

# Use ZeroRedundancyOptimizer to offload optimizer state to CPU.
from torch.distributed.optim import ZeroRedundancyOptimizer

# Enable CuDNN benchmark for improved performance.
torch.backends.cudnn.benchmark = True


def checkpointed_forward(x, model):
    with autocast(device_type='cuda'):
        return model(x)


def main():
    # Distributed training setup.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Setup logging (only on rank 0).
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='training.log',
                            filemode='w')
        logger = logging.getLogger()
        logger.info("Starting training")
    else:
        logger = None

    # Hyperparameters.
    batch_size = 128
    epochs = 300
    learning_rate = 7e-4
    weight_decay = 0.05
    label_smoothing = 0.1
    warmup_epochs = 10

    # Data augmentation pipeline (inspired by DeiT) with additional augmentations.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random jitter.
        transforms.RandomGrayscale(p=0.2),  # Convert image to grayscale with probability 0.2.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Random erasing.
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets.
    train_dataset = datasets.ImageNet(root="/data/jacob/ImageNet", split="train", transform=transform_train)
    val_dataset = datasets.ImageNet(root="/data/jacob/ImageNet", split="val", transform=transform_val)

    # Distributed samplers.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoaders with persistent workers and prefetching.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Model creation and wrapping for distributed training.
    model = FFTNetViT().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Cross-entropy loss with label smoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    
    # Use AdamW wrapped with ZeroRedundancyOptimizer.
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.AdamW,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Cosine annealing scheduler (post-warmup).
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01)

    # Warmup scheduler: linearly increase LR over 'warmup_epochs'.
    def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
        if epoch < warmup_epochs:
            lr = base_lr * float(epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return None

    scaler = GradScaler()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        # Adjust LR during warmup; otherwise step the cosine scheduler.
        current_lr = adjust_learning_rate(optimizer, epoch, warmup_epochs, learning_rate)
        if current_lr is None and epoch >= warmup_epochs:
            scheduler_cosine.step()
            current_lr = optimizer.param_groups[0]['lr']

        if rank == 0 and logger:
            logger.info(f"Epoch {epoch+1}/{epochs}, Learning Rate: {current_lr:.6f}")

        # Training loop.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=(rank != 0)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Use a thread for the forward pass with checkpointing.
            output_container = []
            thread = threading.Thread(target=lambda: output_container.append(checkpointed_forward(images, model)))
            thread.start()
            thread.join()
            outputs = output_container[0]

            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += bs

        # Aggregate training metrics across GPUs.
        train_loss_tensor = torch.tensor(running_loss, device=device)
        train_total_tensor = torch.tensor(total, device=device)
        train_correct_tensor = torch.tensor(correct, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        epoch_train_loss = train_loss_tensor.item() / train_total_tensor.item()
        epoch_train_acc = train_correct_tensor.item() / train_total_tensor.item()

        # Validation loop.
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", disable=(rank != 0)):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(images)
                loss = criterion(outputs, labels)
                bs = images.size(0)
                val_loss += loss.item() * bs
                _, preds = torch.max(outputs, 1)
                val_correct += preds.eq(labels).sum().item()
                val_total += bs

        # Aggregate validation metrics.
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_total_tensor = torch.tensor(val_total, device=device)
        val_correct_tensor = torch.tensor(val_correct, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = val_loss_tensor.item() / val_total_tensor.item()
        epoch_val_acc = val_correct_tensor.item() / val_total_tensor.item()

        if rank == 0:
            log_line = f"{epoch+1},{epoch_train_loss:.4f},{epoch_train_acc:.4f},{epoch_val_loss:.4f},{epoch_val_acc:.4f}"
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            if logger:
                logger.info(log_line)

            # Save the best model.
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.module.state_dict(), "best_model.pth")
                logger.info(f"Saved best model at epoch {epoch+1} with Val Acc: {epoch_val_acc:.4f}")

        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
