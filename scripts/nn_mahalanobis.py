import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, pinv, cond, LinAlgError # Necessary for Mahalanobis

import seaborn as sns
import matplotlib.pyplot as plt

# The LABEL_MAP remains the same as it maps class indices to names
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

# --- Utility Functions (unchanged) ---

def calculate_median_embeddings(embeddings, classes, unique_classes):
    """
    embeddings: (N, D) array
    classes: (N,) array of ints (NN predicted classes)
    unique_classes: sorted list/array of classes to compute medians for
    returns dict {cls: median_vector}
    """
    median_embeddings = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            raise ValueError(f"No examples found for class {cls}")
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
    return median_embeddings


def extract_nn_prediction_classes(npz_data):
    """
    Extracts the predicted classes from the loaded NPZ data.
    """
    if 'predicted_classes' not in npz_data:
         raise KeyError("NPZ must contain a 'predicted_classes' array")
    # This is the array we will use to group the embeddings for median calculation
    return np.array(npz_data['predicted_classes'], dtype=int)


def analyze_embeddings_by_nn_prediction(npz_path, out_csv_path='median_classification_summary_nn_preds.csv',
                                    tie_atol=1e-8, distribute_ties_fractionally=False):
    """
    Reads .npz with keys 'embeddings' and 'predicted_classes' (from NN output).
    Uses DIAGONAL MAHALANOBIS DISTANCE to assess if the NN's predictions form tight,
    self-consistent clusters in the embedding space.
    """
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data:
        raise KeyError("NPZ must contain 'embeddings' array")

    embeddings = np.array(data['embeddings'])

    # The 'classes' array now represents the NN's predicted class for each embedding.
    classes = extract_nn_prediction_classes(data)

    # Unique classes sorted
    unique_classes = np.unique(classes)
    unique_classes.sort()

    # Median embeddings per NN-Predicted class
    median_embeddings = calculate_median_embeddings(embeddings, classes, unique_classes)
    medians_matrix = np.vstack([median_embeddings[cls] for cls in unique_classes])  # shape (C, D)
    
    ## 1. Calculate PER-CLASS DIAGONAL Inverse Covariance Matrices
    #    The groups are defined by the NN's predicted class.
    inv_cov_matrices = {}
    epsilon = 1e-6 # Used to prevent division by zero for zero-variance dimensions
    dimension = embeddings.shape[1]
    
    print("\n--- Diagonal Mahalanobis Analysis of NN Predicted Clusters ---")
    for cls in unique_classes:
        mask = (classes == cls)
        class_embeddings = embeddings[mask]
        num_samples = class_embeddings.shape[0]
        
        # Calculate full covariance, but we only use the diagonal variances
        cov_matrix = np.cov(class_embeddings, rowvar=False) 
        
        # KEY STEP: Extract only the variance (diagonal) terms and make a writable copy
        class_variances = np.diag(cov_matrix).copy() 
        
        # Check for near-zero variance dimensions and regularize with epsilon
        zero_variance_mask = class_variances <= epsilon
        class_variances[zero_variance_mask] = epsilon 
        
        # The Inverse of a Diagonal matrix is a Diagonal matrix with 1/variance
        inv_variances = 1.0 / class_variances
        
        # Create the inverse diagonal matrix V_inv
        inv_cov_matrix = np.diag(inv_variances)
        
        if num_samples < 50:
             print(f"⚠️ Note: NN Predicted Class {cls} ({LABEL_MAP[cls]}) has only {num_samples} samples.")
             
        inv_cov_matrices[cls] = inv_cov_matrix
        
    print(f"--- Finished Calculating {len(unique_classes)} Diagonal Inverse Covariance Matrices. ---\n")
    

    # Prepare results table
    pred_cols = [str(c) for c in unique_classes]
    rows = {str(c): np.zeros(len(unique_classes), dtype=float) for c in unique_classes}
    ties_counts = {str(c): 0 for c in unique_classes}
    totals = {str(c): 0 for c in unique_classes}

    ## 2. Calculate Mahalanobis Distance
    N = embeddings.shape[0]
    C = medians_matrix.shape[0]
    dists = np.zeros((N, C))
    
    # We iterate over every embedding (i) and calculate its distance to every class median (j)
    for i in range(N):
        x = embeddings[i]
        for j, median_cls in enumerate(unique_classes):
            mu = medians_matrix[j]
            # When classifying point 'x' to median 'mu_j', we use the inverse covariance 
            # matrix that corresponds to the class of the median, which is 'median_cls'.
            V_inv = inv_cov_matrices[median_cls]
            
            dists[i, j] = mahalanobis(x, mu, V_inv)
            
    ## 3. Classification and Tie Handling
    # 'nn_pred_cls' is now the 'true' class for this analysis (the grouping class)
    for i, nn_pred_cls in enumerate(classes):
        totals[str(nn_pred_cls)] += 1
        distances = dists[i]  # shape (C,)
        min_dist = distances.min()
        
        tie_mask = np.isclose(distances, min_dist, atol=tie_atol)
        tie_indices = np.nonzero(tie_mask)[0]
        
        if tie_indices.size > 1:
            # tie
            ties_counts[str(nn_pred_cls)] += 1
            if distribute_ties_fractionally:
                frac = 1.0 / tie_indices.size
                for idx in tie_indices:
                    pred_cls_str = str(unique_classes[idx])
                    rows[str(nn_pred_cls)][idx] += frac
        else:
            # pred_idx is the index of the closest median
            pred_idx = int(np.nonzero(tie_mask)[0][0])
            rows[str(nn_pred_cls)][pred_idx] += 1.0

    # Build DataFrame
    df_rows = []
    for cls in unique_classes:
        cls_str = str(cls)
        row = { 'nn_predicted_class': cls, # Renamed column for clarity
                'total_instances': int(totals[cls_str]),
                'ties': int(ties_counts[cls_str]) }
        # counts for each predicted class (closest median)
        for j, pred_cls in enumerate(unique_classes):
            # pred_cls is the class of the closest median
            row[f'closest_median_cls_{pred_cls}'] = float(rows[cls_str][j])

        # 'Correct' means the embedding was closest to its *own* NN-predicted median cluster
        correct = row[f'closest_median_cls_{cls}']
        row['correct'] = float(correct)
        row['correct_pct'] = float(correct) / row['total_instances'] if row['total_instances'] > 0 else 0.0
        df_rows.append(row)

    cols_order = ['nn_predicted_class', 'total_instances', 'ties'] + [f'closest_median_cls_{c}' for c in unique_classes] + ['correct', 'correct_pct']
    df = pd.DataFrame(df_rows)[cols_order].sort_values('nn_predicted_class').reset_index(drop=True)

    # Save CSV
    df.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV to: {out_csv_path}")
    return df


def plot_pretty_confusion_matrices(df):
    """
    df: DataFrame returned from analyze_embeddings_by_nn_prediction()
    Creates 2 seaborn heatmaps: raw counts and normalized. (Unchanged, titles updated)
    """

    # The 'True' classes are now the NN Predicted classes
    nn_predicted_classes = df["nn_predicted_class"].values

    # Extract predicted class columns (closest median class)
    pred_cols = [c for c in df.columns if c.startswith("closest_median_cls_")]
    closest_median_classes = [int(c.replace("closest_median_cls_", "")) for c in pred_cols]

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
                xticklabels=[LABEL_MAP[c] for c in closest_median_classes],
                yticklabels=[LABEL_MAP[c] for c in nn_predicted_classes])

    plt.title("Diagonal Mahalanobis: Raw Counts (Grouped by NN Prediction)", fontsize=16)
    plt.xlabel("Closest Class Median (Diagonal Mahalanobis)", fontsize=14)
    plt.ylabel("NN Predicted Class", fontsize=14)
    plt.tight_layout()
    # Updated save path
    plt.savefig("../analysis_of_embedding_distance/heatmap_median_nn_prediction_diagonal_mahalanobis.pdf")
    plt.show()

    # --------------------------
    # Normalized heatmap
    # --------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                xticklabels=[LABEL_MAP[c] for c in closest_median_classes],
                yticklabels=[LABEL_MAP[c] for c in nn_predicted_classes])

    plt.title("Median Analysis of Neural network predictions", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("NN Predicted Class", fontsize=14)
    plt.tight_layout()
    # Updated save path
    plt.savefig("../analysis_of_embedding_distance/normalised_heatmap_median_nn_prediction_diagonal_mahalanobis.pdf")
    plt.show()

# Example usage with the NN prediction file:
df = analyze_embeddings_by_nn_prediction(
    'nn_predictions_dinov2_weighted_seed42.npz',  # Your NPZ file with 'embeddings' and 'predicted_classes'
    out_csv_path='../analysis_of_embedding_distance/median_classification_summary_nn_preds_diagonal_mahalanobis.csv',
    tie_atol=1e-8,
    distribute_ties_fractionally=False
)
print(df)
plot_pretty_confusion_matrices(df)