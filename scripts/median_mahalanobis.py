import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, pinv, cond, LinAlgError # Added cond and LinAlgError

import seaborn as sns
import matplotlib.pyplot as plt

# Define a threshold for a "poorly conditioned" matrix. 
# A condition number greater than this suggests instability in the inverse.
# Common thresholds are 1e10 to 1e16. 
CONDITION_NUMBER_THRESHOLD = 1e12 

LABEL_MAP = {
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

def extract_class_from_path(path):
    """
    Extract trailing digits from the filename.
    """
    basename = os.path.basename(path)
    name, _ext = os.path.splitext(basename)
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError(f"No trailing digit(s) found in filename '{basename}'")
    return int(m.group(1))

def calculate_median_embeddings(embeddings, classes, unique_classes):
    """
    Calculates the median vector for each unique class.
    """
    median_embeddings = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            raise ValueError(f"No examples found for class {cls}")
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
    return median_embeddings

# ... (imports and helper functions remain the same) ...

def analyze_embeddings_and_save_csv(npz_path, out_csv_path='median_classification_summary.csv',
                                    tie_atol=1e-8, distribute_ties_fractionally=False):
    """
    Reads .npz, calculates Mahalanobis distance using a PER-CLASS DIAGONAL covariance matrix
    to ensure numerical stability in high-dimensional, normalized space.
    """
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data or 'image_paths' not in data:
        raise KeyError("NPZ must contain 'embeddings' and 'image_paths' arrays")

    embeddings = np.array(data['embeddings'])
    image_paths = np.array(data['image_paths'])

    if image_paths.dtype.kind in ('S', 'a', 'O'):
        image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths])

    classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)
    unique_classes = np.unique(classes)
    unique_classes.sort()

    median_embeddings = calculate_median_embeddings(embeddings, classes, unique_classes)
    medians_matrix = np.vstack([median_embeddings[cls] for cls in unique_classes]) 
    
    ## 1. Calculate PER-CLASS DIAGONAL Inverse Covariance Matrices (The stable solution)
    inv_cov_matrices = {}
    epsilon = 1e-6 # Used to prevent division by zero for zero-variance dimensions
    dimension = embeddings.shape[1]
    
    print("\n--- Covariance Matrix Stability Check (Diagonal Mode) ---")
    for cls in unique_classes:
        mask = (classes == cls)
        class_embeddings = embeddings[mask]
        num_samples = class_embeddings.shape[0]
        
        # Calculate full covariance, but we only use the diagonal variances
        cov_matrix = np.cov(class_embeddings, rowvar=False) 
        
        # KEY STEP: Extract only the variance (diagonal) terms and make a writable copy
        class_variances = np.diag(cov_matrix).copy() 
        
        # Check for zero variance dimensions and set to epsilon (for stable division)
        zero_variance_mask = class_variances <= epsilon
        class_variances[zero_variance_mask] = epsilon 
        
        # The Inverse of a Diagonal matrix is a Diagonal matrix with 1/variance
        inv_variances = 1.0 / class_variances
        
        # Create the inverse diagonal matrix V_inv
        inv_cov_matrix = np.diag(inv_variances)
        
        if num_samples < 50:
             print(f"⚠️ Note: Class {cls} ({LABEL_MAP[cls]}) has only {num_samples} samples. Variance estimate may be poor.")
             
        inv_cov_matrices[cls] = inv_cov_matrix
        
    print(f"--- Finished Calculating {len(unique_classes)} Per-Class DIAGONAL Inverse Covariance Matrices. ---\n")
    

    # --- INITIALIZATION FIX: COUNTERS FOR CLASSIFICATION RESULTS ---
    pred_cols = [str(c) for c in unique_classes]
    rows = {str(c): np.zeros(len(unique_classes), dtype=float) for c in unique_classes}
    ties_counts = {str(c): 0 for c in unique_classes}
    totals = {str(c): 0 for c in unique_classes}
    # ---------------------------------------------------------------


    ## 2. Calculate Mahalanobis Distance using per-class diagonal matrices
    N = embeddings.shape[0]
    C = medians_matrix.shape[0]
    dists = np.zeros((N, C))

    for i in range(N):
        x = embeddings[i]
        for j, pred_cls in enumerate(unique_classes):
            mu = medians_matrix[j]
            V_inv = inv_cov_matrices[pred_cls]
            
            # mahalanobis(u, v, VI) works with the diagonal V_inv
            dists[i, j] = mahalanobis(x, mu, V_inv)
            
    ## 3. Classification and Tie Handling
    for i, true_cls in enumerate(classes):
        # totals is now defined and initialized
        totals[str(true_cls)] += 1
        distances = dists[i]
        min_dist = distances.min()
        
        tie_mask = np.isclose(distances, min_dist, atol=tie_atol)
        tie_indices = np.nonzero(tie_mask)[0]
        
        if tie_indices.size > 1:
            # Tie detected
            ties_counts[str(true_cls)] += 1
            if distribute_ties_fractionally:
                frac = 1.0 / tie_indices.size
                for idx in tie_indices:
                    pred_cls_str = str(unique_classes[idx])
                    rows[str(true_cls)][idx] += frac
        else:
            # Single prediction
            pred_idx = int(np.nonzero(tie_mask)[0][0])
            rows[str(true_cls)][pred_idx] += 1.0

    # 4. Build DataFrame
    df_rows = []
    for cls in unique_classes:
        cls_str = str(cls)
        # totals is used here
        row = { 'true_class': cls,
                'total_instances': int(totals[cls_str]),
                'ties': int(ties_counts[cls_str]) }
        
        # rows is used here
        for j, pred_cls in enumerate(unique_classes):
            row[f'pred_{pred_cls}'] = float(rows[cls_str][j])
            
        correct = row[f'pred_{cls}']
        row['correct'] = float(correct)
        row['correct_pct'] = float(correct) / row['total_instances'] if row['total_instances'] > 0 else 0.0
        df_rows.append(row)

    cols_order = ['true_class', 'total_instances', 'ties'] + [f'pred_{c}' for c in unique_classes] + ['correct', 'correct_pct']
    df = pd.DataFrame(df_rows)[cols_order].sort_values('true_class').reset_index(drop=True)

    df.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV to: {out_csv_path}")
    return df

# The rest of the script (plot_pretty_confusion_matrices and example usage) remains the same.
# You will see much more reasonable results now.

def plot_pretty_confusion_matrices(df):
    """
    Creates 2 seaborn heatmaps with updated titles. (Code unchanged)
    """
    # ... (plotting code remains the same as before) ...
    # Extract true classes (rows)
    true_classes = df["true_class"].values

    # Extract predicted class columns
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_classes = [int(c.replace("pred_", "")) for c in pred_cols]

    # Raw confusion matrix
    cm_raw = df[pred_cols].values.astype(float)

    # Normalized confusion matrix (row-wise)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)

    # --------------------------
    # Raw counts heatmap
    # --------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_raw,
                annot=True,
                fmt=".0f",
                cmap="Reds",
                xticklabels=[LABEL_MAP[c] for c in pred_classes],
                yticklabels=[LABEL_MAP[c] for c in true_classes])

    plt.title("Confusion Matrix — Raw Counts (Mahalanobis Distance - Per-Class Covariance)", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("../analysis_of_embedding_distance/heatmap_dino_norm_train_mahalanobis_per_class.pdf")
    plt.show()

    # --------------------------
    # Normalized heatmap
    # --------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                xticklabels=[LABEL_MAP[c] for c in pred_classes],
                yticklabels=[LABEL_MAP[c] for c in true_classes])

    plt.title("Median Analysis (Mahalanobis Distance) \n train data", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("../analysis_of_embedding_distance/normalised_heatmap_dino_norm_train_mahalanobis_per_class.pdf")
    plt.show()


# Example usage:
df = analyze_embeddings_and_save_csv('../normalised_embeddings/dino_normalised_embeddings_train.npz',
                                    out_csv_path='../analysis_of_embedding_distance/median_classification_summary_dino_norm_train_mahalanobis_per_class.csv',
                                    tie_atol=1e-8, # Keep at a small value unless you expect many close ties
                                    distribute_ties_fractionally=False)
print(df)
plot_pretty_confusion_matrices(df)