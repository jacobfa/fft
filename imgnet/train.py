#!/usr/bin/env python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import threading
import logging

# Use ZeroRedundancyOptimizer to offload optimizer state to CPU.
from torch.distributed.optim import ZeroRedundancyOptimizer

# Use timm's mixup implementation.
import timm.data.mixup as mixup_fn_lib

# Use torchvision transforms and datasets.
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Enable CuDNN benchmark for improved performance.
torch.backends.cudnn.benchmark = True

# === Monkey Patch for Hue Adjustment to Avoid OverflowError ===
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F_pil

def safe_adjust_hue(img, hue_factor):
    """
    Adjust the hue of an image while handling negative hue factors.
    Instead of directly converting hue_factor*255 to np.uint8 (which fails for negatives),
    we compute an integer offset and wrap it with modulo 256.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].')
    # Convert image to HSV.
    hsv = img.convert('HSV')
    np_hsv = np.array(hsv)
    # Calculate hue offset: multiply by 255, round and wrap modulo 256.
    hue_offset = int(round(hue_factor * 255)) % 256
    # Adjust the hue channel (channel 0).
    np_hsv[..., 0] = (np_hsv[..., 0].astype(int) + hue_offset) % 256
    # Convert back to RGB.
    new_img = Image.fromarray(np_hsv, mode='HSV').convert('RGB')
    return new_img

# Monkey-patch the adjust_hue function in torchvision.transforms.functional_pil.
F_pil.adjust_hue = safe_adjust_hue
# === End of Monkey Patch ===

# EMA helper class.
class ModelEma:
    def __init__(self, model, decay=0.9998, device=None):
        self.ema = copy.deepcopy(model)
        self.decay = decay
        self.device = device
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device=device)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1 - self.decay))
    def state_dict(self):
        return self.ema.state_dict()

# Soft cross entropy for mixup.
def soft_cross_entropy(pred, soft_targets):
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(- soft_targets * log_prob, dim=1))

def checkpointed_forward(x, model):
    with autocast(device_type='cuda'):
        return model(x)

def main():
    parser = argparse.ArgumentParser(description="Optimized training for ViT/DeiT with advanced techniques")
    parser.add_argument('--model', default='deit_base_patch16_224', type=str,
                        help="Model to train: 'fftnet_vit' for custom FFTNetViT, or a timm model name (e.g. deit_base_patch16_224)")
    parser.add_argument('--mixup_alpha', default=0.8, type=float, help="Alpha value for mixup")
    parser.add_argument('--cutmix_alpha', default=1.0, type=float, help="Alpha value for cutmix")
    parser.add_argument('--use_ema', action='store_true', help="Enable EMA for model weights")
    parser.add_argument('--ema_decay', default=0.9998, type=float, help="EMA decay rate")
    parser.add_argument('--grad_clip', default=1.0, type=float, help="Max norm for gradient clipping")
    parser.add_argument('--drop_rate', default=0.0, type=float, help="Dropout rate")
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help="Drop path (stochastic depth) rate")
    args = parser.parse_args()

    # Distributed training setup.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Setup logging on rank 0.
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
    label_smoothing = 0.0  
    warmup_epochs = 10

    # -----------------------
    # Define Transform Pipelines using torchvision.
    # -----------------------

    # Training augmentations.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    ])

    # Evaluation transforms: resize directly to 224×224 to avoid any center bias.
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets using torchvision.datasets.ImageNet.
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
    if args.model.lower() == 'fftnet_vit':
        from fftnet_vit import FFTNetViT
        model = FFTNetViT(drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate).to(device)
    else:
        import timm
        model = timm.create_model(
            args.model,
            pretrained=False,
            num_classes=1000,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate
        ).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize EMA if enabled.
    if args.use_ema:
        ema = ModelEma(model.module, decay=args.ema_decay, device=device)
    else:
        ema = None

    # Setup mixup/cutmix.
    mixup_fn_inst = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn_inst = mixup_fn_lib.Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=label_smoothing,
            num_classes=1000)

    # Loss function.
    if mixup_fn_inst is not None:
        criterion = soft_cross_entropy
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # Use AdamW wrapped with ZeroRedundancyOptimizer.
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.AdamW,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Cosine annealing scheduler (post-warmup).
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
    )

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

        current_lr = adjust_learning_rate(optimizer, epoch, warmup_epochs, learning_rate)
        if current_lr is None and epoch >= warmup_epochs:
            scheduler_cosine.step()
            current_lr = optimizer.param_groups[0]['lr']

        if rank == 0 and logger:
            logger.info(f"Epoch {epoch+1}/{epochs}, Learning Rate: {current_lr:.6f}")

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=(rank != 0)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Apply mixup/cutmix if enabled.
            if mixup_fn_inst is not None:
                images, labels = mixup_fn_inst(images, labels)

            optimizer.zero_grad(set_to_none=True)

            output_container = []
            thread = threading.Thread(
                target=lambda: output_container.append(checkpointed_forward(images, model))
            )
            thread.start()
            thread.join()
            outputs = output_container[0]

            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model.module)

            bs = images.size(0)
            running_loss += loss.item() * bs
            _, preds = torch.max(outputs, 1)
            if mixup_fn_inst is None:
                correct += preds.eq(labels).sum().item()
            total += bs

        train_loss_tensor = torch.tensor(running_loss, device=device)
        train_total_tensor = torch.tensor(total, device=device)
        train_correct_tensor = torch.tensor(correct, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        epoch_train_loss = train_loss_tensor.item() / train_total_tensor.item()
        epoch_train_acc = train_correct_tensor.item() / train_total_tensor.item()

        # Use EMA model for evaluation if available.
        model_eval = ema.ema if ema is not None else model
        model_eval.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", disable=(rank != 0)):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model_eval(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                bs = images.size(0)
                val_loss += loss.item() * bs
                _, preds = torch.max(outputs, 1)
                val_correct += preds.eq(labels).sum().item()
                val_total += bs

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

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.module.state_dict(), "best_model.pth")
                logger.info(f"Saved best model at epoch {epoch+1} with Val Acc: {epoch_val_acc:.4f}")

        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
