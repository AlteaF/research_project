import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def split_dataset(dataset, val_fraction=0.5, random_state=42):
    targets = [label for _, label in dataset.samples]
    targets = np.array(targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    indices = np.arange(len(dataset))
    test_idx, val_idx = next(sss.split(indices, targets))
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    return val_subset, test_subset

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_mapping):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',  
        cmap='Reds',
        xticklabels=class_mapping.values(),
        yticklabels=class_mapping.values(),
        vmin=0,
        vmax=1  
    )
    plt.title('Normalized Confusion Matrix for ResNet50 Finetuned ')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    # Transform (must match training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test data
    test_data_full = torchvision.datasets.ImageFolder(root="../dataset/dataset_prepared_resnet/cropped_test", transform=transform)
    _, test_data = split_dataset(test_data_full, val_fraction=0.5)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(test_data_full.classes))
    model.load_state_dict(torch.load("saved_models/best_resnet50_finetuned_up_to_2.pth", map_location=device))
    model = model.to(device)

    # Evaluate
    y_pred, y_true = evaluate_model(model, test_loader, device)
    class_mapping = {
            1: "Starfish",
            2: "Crab",
            3: "Black goby",
            4: "Wrasse",
            5: "Two-spotted goby",
            6: "Cod",
            7: "Painted goby",
            8: "Sand eel",
            9: "Whiting"
        }
    # Metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=test_data_full.classes))
    plot_confusion_matrix(y_true, y_pred, class_mapping)

if __name__ == "__main__":
    
    main()
