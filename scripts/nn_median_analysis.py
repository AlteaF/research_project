import os
import re
import numpy as np
import pandas as pd

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

# --- Utility Functions (unchanged, but not strictly needed for the NN approach) ---
# Keeping extract_class_from_path() for completeness if needed for a different analysis

def calculate_median_embeddings(embeddings, classes, unique_classes):
    """
    embeddings: (N, D) array
    classes: (N,) array of ints (now, these are NN predicted classes)
    unique_classes: sorted list/array of classes to compute medians for
    returns dict {cls: median_vector}
    """
    median_embeddings = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            # This should ideally not happen if unique_classes comes from the classes array
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
    Produces CSV summarizing, for each NN-PREDICTED class, how many embeddings were closest to each class median.
    The 'true_class' column in the output DataFrame will represent the NN's predicted class.
    The 'correct' count is how many embeddings were closest to their *own* NN-predicted median.
    """
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data:
        raise KeyError("NPZ must contain 'embeddings' array")

    embeddings = np.array(data['embeddings'])

    # --- KEY CHANGE: Use NN Predicted Classes for grouping ---
    # The 'classes' array now represents the NN's predicted class for each embedding.
    classes = extract_nn_prediction_classes(data)

    # Unique classes sorted
    unique_classes = np.unique(classes)
    unique_classes.sort()

    # Median embeddings per NN-Predicted class
    # Calculate the median for each group defined by the NN's prediction
    median_embeddings = calculate_median_embeddings(embeddings, classes, unique_classes)
    # Stack medians into array for vectorized distance calc
    medians_matrix = np.vstack([median_embeddings[cls] for cls in unique_classes])  # shape (C, D)

    # Prepare results table
    # columns: predicted class labels (as strings), plus 'ties', 'total', 'correct', 'correct_pct'
    # Here, 'pred_cls' columns represent the median the embedding was closest to
    pred_cols = [str(c) for c in unique_classes]
    # 'rows' are keyed by the NN's predicted class (the 'true' class for this analysis)
    rows = {str(c): np.zeros(len(unique_classes), dtype=float) for c in unique_classes}
    ties_counts = {str(c): 0 for c in unique_classes}
    totals = {str(c): 0 for c in unique_classes}

    # Vectorized distance calculation remains the same
    diff = embeddings[:, None, :] - medians_matrix[None, :, :]  # (N, C, D)
    dists_sq = np.sum(diff * diff, axis=2)  # (N, C)
    dists = np.sqrt(dists_sq)

    # Iterate through each embedding, where 'true_cls' is the NN's prediction
    for i, nn_pred_cls in enumerate(classes):
        totals[str(nn_pred_cls)] += 1
        distances = dists[i]  # shape (C,)
        min_dist = distances.min()
        # find indices within tolerance -> ties
        tie_mask = np.isclose(distances, min_dist, atol=tie_atol)
        tie_indices = np.nonzero(tie_mask)[0]
        if tie_indices.size > 1:
            # tie
            ties_counts[str(nn_pred_cls)] += 1
            if distribute_ties_fractionally:
                frac = 1.0 / tie_indices.size
                for idx in tie_indices:
                    # pred_cls_str is the median the embedding was closest to
                    pred_cls_str = str(unique_classes[idx])
                    rows[str(nn_pred_cls)][idx] += frac
            else:
                # keep ties out of rows (they will be visible in 'ties' column)
                pass
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

        # --- KEY LOGIC CHANGE: 'Correct' now means closest to its own NN-predicted median ---
        # The 'correct' count is the value from the diagonal, i.e.,
        # where the NN-predicted class (cls) matches the closest median class (cls).
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
    Creates 2 seaborn heatmaps:
      - raw counts
      - normalized per NN-predicted class
    """

    # The 'True' classes are now the NN Predicted classes
    nn_predicted_classes = df["nn_predicted_class"].values

    # Extract predicted class columns (closest median class)
    # The columns are now named 'closest_median_cls_X'
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

    plt.title("Median Analysis: Raw Counts (Grouped by NN Prediction)", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("NN Predicted Class (Group)", fontsize=14)
    plt.tight_layout()
    # Updated save path
    plt.savefig("../analysis_of_embedding_distance/heatmap_median_nn_prediction.pdf")
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

    plt.title("Heat map (row norm) of NN predictions \n median analysis (Mahalanobis distance)", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("NN Predicted Class (Group)", fontsize=14)
    plt.tight_layout()
    # Updated save path
    plt.savefig("../analysis_of_embedding_distance/normalised_heatmap_median_nn_prediction.pdf")
    plt.show()
# Example usage with the NN prediction file:
df = analyze_embeddings_by_nn_prediction(
    'nn_predictions_dinov2_weighted_seed42.npz',  # The NPZ file saved by your NN script
    out_csv_path='../analysis_of_embedding_distance/median_classification_summary_nn_preds.csv',
    tie_atol=1e-8,
    distribute_ties_fractionally=False
)
print(df)
plot_pretty_confusion_matrices(df)