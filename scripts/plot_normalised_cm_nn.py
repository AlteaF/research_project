import numpy as np
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# === Label extraction utility ===
def extract_labels(paths):
    return np.array([int(re.search(r'(\d)(?=\.jpg$)', p).group(1)) for p in paths])


# === Define the same model architecture used for training ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# === Function to plot normalized confusion matrix for NN ===
def plot_normalized_confusion_matrix_nn(model_path, npz_path, class_mapping):
    """
    Args:
        model_path: Path to saved NN model (.pth file)
        npz_path: Path to .npz file containing 'embeddings' and 'image_paths'
        class_mapping: Dict mapping class indices to names
    """
    # Load data
    data = np.load(npz_path)
    X_all = data["embeddings"]
    y_all = extract_labels(data["image_paths"])

    # Split into val/test (same split as before)
    X_val, X_test, y_val, y_test = train_test_split(
        X_all, y_all, test_size=0.5, random_state=42, stratify=y_all
    )

    # Convert to tensors
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test - 1, dtype=torch.long)  # 0-indexed for NN

    # Load model (architecture must match training)
    input_dim = X_test.shape[1]
    hidden_dim = 850
    output_dim = len(class_mapping)
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred = outputs.argmax(dim=1).numpy()
        y_true = y_test_t.numpy()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=class_mapping.values(),
        yticklabels=class_mapping.values(),
    )
    plt.title("Normalized Confusion Matrix for Neural Network HD=850")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return cm_norm


# === Example usage ===
if __name__ == "__main__":
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

    model_path = "nn_dinov2_weighted_850_seed42.pth"
    npz_path = "../normalised_embeddings/dino_normalised_embeddings_test.npz"

    cm_norm = plot_normalized_confusion_matrix_nn(model_path, npz_path, class_mapping)
