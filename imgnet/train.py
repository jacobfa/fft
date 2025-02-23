#!/usr/bin/env python
import os
# Set environment variable to help mitigate memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fftnet_vit import FFTNetViT  # Ensure FFTNetViT is defined appropriately.
from tqdm import tqdm
import threading

# Use ZeroRedundancyOptimizer to offload optimizer state to CPU.
from torch.distributed.optim import ZeroRedundancyOptimizer

# Enable CuDNN benchmark for improved performance.
torch.backends.cudnn.benchmark = True

def checkpointed_forward(x, model):
    # Run the model's forward pass under mixed precision.
    with autocast(device_type='cuda'):
        return model(x)

def main():
    # Retrieve distributed training parameters from environment variables.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set the GPU device for this process.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize the distributed process group.
    dist.init_process_group(backend="nccl", init_method="env://")

    # Hyperparameters.
    batch_size = 128
    epochs = 300
    learning_rate = 7e-4

    # Define ImageNet transforms.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create ImageNet datasets.
    train_dataset = datasets.ImageNet(root="/data/jacob/ImageNet", split="train", transform=transform_train)
    val_dataset = datasets.ImageNet(root="/data/jacob/ImageNet", split="val", transform=transform_val)

    # Set up distributed samplers.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoaders using multiple worker processes, persistent workers, and prefetching.
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

    # Create and wrap the model in DistributedDataParallel.
    model = FFTNetViT().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    
    # Use ZeroRedundancyOptimizer so that optimizer state is mostly stored on CPU.
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.Adam,
        lr=learning_rate
    )
    
    # Set up automatic mixed precision training.
    scaler = GradScaler()

    best_val_acc = 0.0
    log_file = None
    if rank == 0:
        log_file = open("log.txt", "w")
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
        log_file.flush()

    # Define a custom forward function for use with checkpointing.
    def custom_forward(x):
        return checkpointed_forward(x, model)

    # Optionally offload the forward pass to a separate thread.
    def threaded_forward(x, output_container):
        output_container.append(custom_forward(x))

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=(rank != 0)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Use a thread to run the forward pass.
            output_container = []
            thread = threading.Thread(target=threaded_forward, args=(images, output_container))
            thread.start()
            thread.join()  # Wait for the forward pass to complete.
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

        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_total_tensor = torch.tensor(val_total, device=device)
        val_correct_tensor = torch.tensor(val_correct, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = val_loss_tensor.item() / val_total_tensor.item()
        epoch_val_acc = val_correct_tensor.item() / val_total_tensor.item()

        if rank == 0:
            log_line = f"{epoch+1},{epoch_train_loss:.4f},{epoch_train_acc:.4f},{epoch_val_loss:.4f},{epoch_val_acc:.4f}\n"
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            log_file.write(log_line)
            log_file.flush()

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.module.state_dict(), "best_model.pth")
                print(f"Saved best model at epoch {epoch+1} with Val Acc: {epoch_val_acc:.4f}")

        torch.cuda.empty_cache()

    if log_file is not None:
        log_file.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
