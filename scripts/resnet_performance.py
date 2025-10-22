import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os


def split_dataset(dataset, val_fraction=0.5, random_state=42):
    """Same stratified split used during training."""
    targets = [label for _, label in dataset.samples]
    targets = np.array(targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    test_idx, val_idx = next(sss.split(np.arange(len(dataset)), targets))
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    return val_subset, test_subset


def evaluate(model, dataloader, device):
    """Runs inference and returns predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    # --- Paths ---
    dataset_root = "../dataset/dataset_prepared/cropped_test"
    model_path = "saved_models/best_resnet50_finetuned.pth"

    # --- Transforms (same as training) ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Load dataset ---
    full_test_data = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)
    val_data, test_data = split_dataset(full_test_data, val_fraction=0.5)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # --- Load model ---
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(full_test_data.classes))

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")

    # --- Evaluate ---
    print("🔎 Evaluating on test set...")
    true_labels, pred_labels = evaluate(model, test_loader, device)

    # --- Metrics ---
    acc = np.mean(true_labels == pred_labels)
    print(f"\n✅ Test Accuracy: {acc:.4f}\n")

    cm = confusion_matrix(true_labels, pred_labels)
    print("📊 Confusion Matrix:")
    print(cm)

    report = classification_report(true_labels, pred_labels,
                                   target_names=full_test_data.classes)
    print("\n📋 Classification Report:")
    print(report)


if __name__ == "__main__":
    main()
