#!/usr/bin/env python
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import threading
import argparse
import numpy as np

# Import your LRA dataset classes.
from lra_datasets import ImdbDataset, ListOpsDataset, Cifar10Dataset
# Import the FFTNet model adapted for LRA tasks.
from fftnet import FFTNetLRA

# Enable CuDNN benchmark for improved performance.
torch.backends.cudnn.benchmark = True

def checkpointed_forward(x, model):
    with autocast(device_type='cuda'):
        return model(x)

def main():
    parser = argparse.ArgumentParser(description="LRA Benchmark Training")
    parser.add_argument('--dataset', type=str, choices=['imdb', 'listops', 'cifar10'], required=True,
                        help="Dataset to use: imdb, listops, or cifar10")
    parser.add_argument('--max_length', type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    parser.add_argument('--seq_len', type=int, default=1024,
                        help="Sequence length input to the model")
    parser.add_argument('--input_dim', type=int, default=1,
                        help="Input dimension per token (e.g. 1 for scalar tokens)")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of classes for classification")
    # Adjusted hyperparameters for LRA tasks.
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument('--depth', type=int, default=6,
                        help="Transformer depth")
    parser.add_argument('--embed_dim', type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help="MLP expansion ratio")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate")
    # New arguments for training improvements.
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help="Label smoothing value for CrossEntropyLoss")
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help="Exponential Moving Average decay rate")
    args = parser.parse_args()
    
    # Define a simple tokenizer function.
    def simple_tokenizer(source, max_length):
        # For text input (IMDB or ListOps), we convert characters to their ordinal values.
        # For CIFAR10, the source is a numpy array (grayscale pixels).
        if isinstance(source, str):
            tokens = [ord(c) for c in source[:max_length]]
        elif isinstance(source, np.ndarray):
            tokens = source.flatten()[:max_length].tolist()
        else:
            tokens = list(source)[:max_length]
        # Pad the sequence if needed.
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        # Return a tensor of shape (max_length,).
        return torch.tensor(tokens, dtype=torch.float32)

    # Create a simple config object to pass to dataset constructors.
    class Config:
        pass
    config = Config()
    config.tokenizer = simple_tokenizer
    config.max_length = args.max_length

    # Instantiate the chosen dataset.
    if args.dataset == 'imdb':
        train_dataset = ImdbDataset(config, split='train')
        val_dataset = ImdbDataset(config, split='eval')
        args.input_dim = 1
        args.seq_len = args.max_length
        args.num_classes = 2
    elif args.dataset == 'listops':
        train_dataset = ListOpsDataset(config, split='train')
        val_dataset = ListOpsDataset(config, split='eval')
        args.input_dim = 1
        args.seq_len = args.max_length
        args.num_classes = 10  # Correct the number of classes for ListOps.
    elif args.dataset == 'cifar10':
        train_dataset = Cifar10Dataset(config, split='train')
        val_dataset = Cifar10Dataset(config, split='eval')
        args.input_dim = 1
        args.seq_len = args.max_length
        args.num_classes = 10

    # Retrieve distributed training parameters.
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set up distributed samplers.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Instantiate the FFTNet model for LRA.
    model = FFTNetLRA(
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_heads=args.num_heads,
        adaptive_spectral=True
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Create loss with label smoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Cosine annealing scheduler.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Create an EMA copy of the model (using the underlying module).
    ema_model = copy.deepcopy(model.module)
    for param in ema_model.parameters():
        param.requires_grad = False

    best_val_acc = 0.0
    log_file = None
    if rank == 0:
        log_file = open("log.txt", "w")
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
        log_file.flush()

    def custom_forward(x):
        return checkpointed_forward(x, model)

    def threaded_forward(x, output_container):
        output_container.append(custom_forward(x))

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop.
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0)):
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)  # (B, seq_len, 1)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            output_container = []
            thread = threading.Thread(target=threaded_forward, args=(inputs, output_container))
            thread.start()
            thread.join()
            outputs = output_container[0]

            loss = criterion(outputs, labels.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update EMA model.
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.module.parameters()):
                    ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels.squeeze()).sum().item()
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

        # Validation loop (using the regular model; you could also evaluate the EMA model if desired).
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0)):
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(-1)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze())
                bs = inputs.size(0)
                val_loss += loss.item() * bs
                _, preds = torch.max(outputs, 1)
                val_correct += preds.eq(labels.squeeze()).sum().item()
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
            log_line = f"{epoch+1},{epoch_train_loss:.4f},{epoch_train_acc:.4f},{epoch_val_loss:.4f},{epoch_val_acc:.4f}\n"
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            log_file.write(log_line)
            log_file.flush()

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                # Optionally, you could save the EMA model instead.
                torch.save(model.module.state_dict(), "best_model.pth")
                print(f"Saved best model at epoch {epoch+1} with Val Acc: {epoch_val_acc:.4f}")

        # Update the learning rate using cosine annealing.
        scheduler.step()
        torch.cuda.empty_cache()

    if log_file is not None:
        log_file.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
