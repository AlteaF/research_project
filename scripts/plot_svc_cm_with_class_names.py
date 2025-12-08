import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import train_test_split


def extract_labels(paths):
    # Extract the last digit before .jpg, e.g. "image_3.jpg" -> 3
    return np.array([int(re.search(r'(\d)(?=\.jpg$)', p).group(1)) for p in paths])

def plot_normalized_confusion_matrix_svm(model_path, npz_path, class_mapping):
    """
    Args:
        model_path: Path to the saved SVM model (.joblib file).
        npz_path: Path to .npz file containing 'embeddings' and 'image_paths'.
        class_mapping: Dictionary mapping class indices to names.
    """
    # Load model and data
    model = joblib.load(model_path)
    data = np.load(npz_path)
    X_test_all = data["embeddings"]
    y_test_all = extract_labels(data["image_paths"])

    # Split test file into validation and final test (50/50)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_all, y_test_all, test_size=0.5, random_state=42, stratify=y_test_all
    )

    # Predict
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds",
                xticklabels=class_mapping.values(),
                yticklabels=class_mapping.values())
    plt.title("Normalized Confusion Matrix for SVC Polynomial 2nd degree")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return cm_norm

# Example usage
if __name__ == "__main__":
    # Define class mapping 
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

    # Paths
    model_path = "SVC_poly_2_dinov2.joblib"
    npz_path = "../normalised_embeddings/dino_normalised_embeddings_test.npz"

    # Plot and get normalized confusion matrix
    cm_norm = plot_normalized_confusion_matrix_svm(model_path, npz_path, class_mapping)
