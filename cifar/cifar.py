import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Import the two models.
from fftnet_vit import FFTNetViT
from transformer import VisionTransformer  # Assumes transformer.py defines a Transformer class

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, model_name):
    """Train the given model and save per-epoch validation metrics to a file."""
    metrics_file = f"{model_name}_val_metrics.txt"
    # Write header line to metrics file.
    with open(metrics_file, "w") as f:
        f.write("Epoch,Validation Loss,Validation Accuracy\n")
    
    for epoch in range(num_epochs):
        # Training phase.
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * train_correct / train_total
        print(f"\n{model_name} Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%")
        
        # Validation phase.
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in test_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        val_loss = test_loss / len(test_loader.dataset)
        val_acc = 100. * correct / total
        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")
        
        # Save validation metrics to file.
        with open(metrics_file, "a") as f:
            f.write(f"{epoch+1},{val_loss:.4f},{val_acc:.2f}\n")

def main():
    # Hyperparameters for CIFAR10.
    num_epochs = 100
    batch_size = 128
    learning_rate = 7e-4
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    # Data transforms for CIFAR10.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR10 dataset.
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instantiate the two models.
    # FFTNetViT model.
    fftnet_model = FFTNetViT(img_size=32, patch_size=4, in_chans=3, num_classes=10,
                              embed_dim=192, depth=6, mlp_ratio=3.0, dropout=0.1,
                              num_heads=6, adaptive_spectral=True)
    fftnet_model.to(device)
    
    # Transformer model from transformer.py.
    transformer_model = VisionTransformer(image_size=32, patch_size=4, in_channels=3, num_classes=10,
                                    embed_dim=192, depth=6, mlp_ratio=3.0,
                                    num_heads=6)
    transformer_model.to(device)
    
    # Define loss criterion.
    criterion = nn.CrossEntropyLoss()

    # Create separate optimizers for each model.
    optimizer_fftnet = optim.Adam(fftnet_model.parameters(), lr=learning_rate)
    optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    
    print("Starting training for FFTNetViT model...")
    train_model(fftnet_model, train_loader, test_loader, optimizer_fftnet, criterion,
                num_epochs, device, model_name="FFTNetsViT")
    
    print("Starting training for Transformer model...")
    train_model(transformer_model, train_loader, test_loader, optimizer_transformer, criterion,
                num_epochs, device, model_name="Transformer")
    
if __name__ == "__main__":
    main()
