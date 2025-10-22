import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# === Logistic Regression (multinomial) ===
best_model = None
best_val_f1 = 0
best_C = None

for C in [0.01, 0.1, 1, 10]:
    clf = LogisticRegression(
        C=C, solver="lbfgs", multi_class="multinomial", max_iter=1000,
        class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"C={C} | Val Macro F1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = clf
        best_C = C

print(f"\nBest C={best_C} | Validation Macro F1={best_val_f1:.4f}")

# === Evaluate on test set ===
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

print("\n=== Final Test Metrics (Logistic Regression) ===")
print(f"Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")
print(classification_report(y_test, y_pred))

# Normalized confusion matrix
plt.figure(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', cbar=False)
plt.title("Normalized Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save model
joblib.dump(best_model, "logreg_dinov2.joblib")
print(f"Saved Logistic Regression model (C={best_C}) to 'logreg_dinov2.joblib'")
