from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib


# === Utility to extract labels ===
def extract_labels(paths):
    return np.array([int(re.search(r'(\d)(?=\.jpg$)', p).group(1)) for p in paths])

# === Load embeddings ===
train_data = np.load("../normalised_embeddings/dino_normalised_embeddings_train.npz")
test_data = np.load("../normalised_embeddings/dino_normalised_embeddings_test.npz")

X_train = train_data["embeddings"]
y_train = extract_labels(train_data["image_paths"])

X_test_all = test_data["embeddings"]
y_test_all = extract_labels(test_data["image_paths"])

# Split test into validation and final test
X_val, X_test, y_val, y_test = train_test_split(
    X_test_all, y_test_all, test_size=0.5, random_state=42, stratify=y_test_all
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

classifier = MLPClassifier(hidden_layer_sizes=(30, 20, 30, 15)).fit(X_train, y_train)
y_val_pred = classifier.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average="macro")

print(f"Val Macro F1={val_f1:.4f}")
y_pred = classifier.predict(X_test)


acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalized per row

print("\n=== Final Test Metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Normalized Confusion Matrix ===
plt.figure(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Reds', cbar=False)
plt.title("Normalized Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# === Save final model ===
joblib.dump(classifier, "mlp_25_15_25_10_dinov2.joblib")
print(f"\nSaved best model to 'mlp_dinov2.joblib'")