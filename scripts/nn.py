import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split

# === Utility to extract labels ===
def extract_labels(paths):
    return np.array([int(re.search(r'(\d)(?=\.jpg$)', p).group(1)) for p in paths])

# === Load embeddings ===
train_data = np.load("../normalised_embeddings/dino_normalised_embeddings_train_gray.npz")
test_data = np.load("../normalised_embeddings/dino_normalised_embeddings_test_gray.npz")

X_train = train_data["embeddings"]
y_train = extract_labels(train_data["image_paths"])

X_test_all = test_data["embeddings"]
y_test_all = extract_labels(test_data["image_paths"])

# Split test into validation and final test
X_val, X_test, y_val, y_test = train_test_split(
    X_test_all, y_test_all, test_size=0.5, random_state=42, stratify=y_test_all
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Convert data to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train - 1, dtype=torch.long)  # 0-indexed
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val - 1, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test - 1, dtype=torch.long)

# Compute class weights (inverse frequency)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_t.numpy()), y=y_train_t.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Datasets and loaders
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

# Simple Feedforward NN
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

input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = 9
model = SimpleNN(input_dim, hidden_dim, output_dim)

# Use class weights in CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_pred = val_out.argmax(dim=1)
        val_f1 = f1_score(y_val_t, val_pred, average="macro")
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Val Macro F1: {val_f1:.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_out = model(X_test_t)
    y_pred = test_out.argmax(dim=1).numpy()
    y_true = y_test_t.numpy()

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

print("\n=== Final Test Metrics (NN with class weights) ===")
print(f"Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")
print(classification_report(y_true, y_pred))

# Normalized confusion matrix
plt.figure(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', cbar=False)
plt.title("Normalized Confusion Matrix (NN with class weights)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save model
torch.save(model.state_dict(), "nn_dinov2_weighted_gray.pth")
print("Saved Neural Network model with class weights to 'nn_dinov2_weighted.pth'")
