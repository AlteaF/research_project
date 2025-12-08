import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os



def split_dataset(dataset, val_fraction=0.5, random_state=42):
    """
    Splits a dataset into two equal parts (validation/test) in a stratified way.
    """
    targets = [label for _, label in dataset.samples]
    targets = np.array(targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    indices = np.arange(len(dataset))
    test_idx, val_idx = next(sss.split(indices, targets))

    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    return val_subset, test_subset


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss, correct = 0.0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct.float() / len(dataloader.dataset)
    return avg_loss, accuracy


def main():
    # Deterministic preprocessing only (no augmentation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = torchvision.datasets.ImageFolder(root="../dataset/dataset_prepared/cropped_train", transform=transform)
    test_data_full = torchvision.datasets.ImageFolder(root="../dataset/dataset_prepared/cropped_test", transform=transform)

    # Split test_data_full → validation + test (stratified)
    val_data, test_data = split_dataset(test_data_full, val_fraction=0.5)

    print(f"Validation samples: {len(val_data)} | Test samples: {len(test_data)}")

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # Model setup
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)

    for name, layer in model.named_children():
        if name in ['conv1', 'bn1', 'layer1', 'layer2', "layer3"]:
            for param in layer.parameters():
                param.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🚀 Using device: {device}")

    # Training setup
    num_epochs = 10
    best_val_acc = 0.0
    best_model_path = "saved_models/best_resnet50_finetuned.pth"
    os.makedirs("saved_models", exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save the best model only
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model updated (epoch {epoch+1}) — Val Acc: {val_acc:.4f}")

    # Final evaluation on test set
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n🏁 Final Test Results | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
