import os
import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
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
    Extract trailing digits from the filename (digits immediately before the extension).
    Examples:
      /.../img_3.jpg -> 3
      /.../cat12.jpg -> 12
      /.../img-9.png -> 9
    Returns int; raises ValueError if no trailing digits found.
    """
    basename = os.path.basename(path)
    name, _ext = os.path.splitext(basename)
    # find trailing digits at end of name
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError(f"No trailing digit(s) found in filename '{basename}'")
    return int(m.group(1))

def calculate_median_embeddings(embeddings, classes, unique_classes):
    """
    embeddings: (N, D) array
    classes: (N,) array of ints
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


def analyze_embeddings_and_save_csv(npz_path, out_csv_path='median_classification_summary.csv',
                                    tie_atol=1e-8, distribute_ties_fractionally=False):
    """
    Reads .npz with keys 'embeddings' and 'image_paths'.
    Produces CSV summarizing, for each true class, how many embeddings were closest to each class median.
    tie_atol: absolute tolerance used for tie detection with np.isclose
    distribute_ties_fractionally: if True, ties are split equally among the tied predicted classes
    """
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data or 'image_paths' not in data:
        raise KeyError("NPZ must contain 'embeddings' and 'image_paths' arrays")

    embeddings = np.array(data['embeddings'])
    image_paths = np.array(data['image_paths'])

    # If image_paths are bytes, decode to str
    if image_paths.dtype.kind in ('S', 'a', 'O'):
        # convert each to str robustly
        image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths])

    # Extract classes
    classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)

    # Unique classes sorted
    unique_classes = np.unique(classes)
    unique_classes.sort()

    # Median embeddings per class
    median_embeddings = calculate_median_embeddings(embeddings, classes, unique_classes)
    # Stack medians into array for vectorized distance calc
    medians_matrix = np.vstack([median_embeddings[cls] for cls in unique_classes])  # shape (C, D)

    # Prepare results table
    # columns: predicted class labels (as strings), plus 'ties', 'total', 'correct', 'correct_pct'
    pred_cols = [str(c) for c in unique_classes]
    rows = {str(c): np.zeros(len(unique_classes), dtype=float) for c in unique_classes}  # we store counts (float to allow fractional tie splits)
    ties_counts = {str(c): 0 for c in unique_classes}
    totals = {str(c): 0 for c in unique_classes}

    # For speed, vectorized distance: for each embedding compute distances to all medians
    # embeddings shape (N, D), medians_matrix (C, D) -> compute (N, C)
    # we'll use broadcasting and norms
    # dist^2 = sum((emb[:, None, :] - medians[None, :, :])**2, axis=2)
    diff = embeddings[:, None, :] - medians_matrix[None, :, :]  # (N, C, D)
    dists_sq = np.sum(diff * diff, axis=2)  # (N, C)
    dists = np.sqrt(dists_sq)

    for i, true_cls in enumerate(classes):
        totals[str(true_cls)] += 1
        distances = dists[i]  # shape (C,)
        min_dist = distances.min()
        # find indices within tolerance -> ties
        tie_mask = np.isclose(distances, min_dist, atol=tie_atol)
        tie_indices = np.nonzero(tie_mask)[0]
        if tie_indices.size > 1:
            # tie
            ties_counts[str(true_cls)] += 1
            if distribute_ties_fractionally:
                frac = 1.0 / tie_indices.size
                for idx in tie_indices:
                    pred_cls_str = str(unique_classes[idx])
                    rows[str(true_cls)][idx] += frac
            else:
                # keep ties out of rows (they will be visible in 'ties' column)
                pass
        else:
            pred_idx = int(np.nonzero(tie_mask)[0][0])
            rows[str(true_cls)][pred_idx] += 1.0

    # Build DataFrame
    df_rows = []
    for cls in unique_classes:
        cls_str = str(cls)
        row = { 'true_class': cls,
                'total_instances': int(totals[cls_str]),
                'ties': int(ties_counts[cls_str]) }
        # counts for each predicted class
        for j, pred_cls in enumerate(unique_classes):
            row[f'pred_{pred_cls}'] = float(rows[cls_str][j])
        # correct count and pct
        correct = row[f'pred_{cls}']
        row['correct'] = float(correct)
        row['correct_pct'] = float(correct) / row['total_instances'] if row['total_instances'] > 0 else 0.0
        df_rows.append(row)

    cols_order = ['true_class', 'total_instances', 'ties'] + [f'pred_{c}' for c in unique_classes] + ['correct', 'correct_pct']
    df = pd.DataFrame(df_rows)[cols_order].sort_values('true_class').reset_index(drop=True)

    # Save CSV
    df.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV to: {out_csv_path}")
    return df


def plot_pretty_confusion_matrices(df):
    """
    df: DataFrame returned from analyze_embeddings_and_save_csv()
    Creates 2 seaborn heatmaps:
      - raw counts
      - normalized per true class
    """

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

    plt.title("Confusion Matrix — Raw Counts", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("../analysis_of_embedding_distance/heatmap_dino_norm_train.pdf")
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

    plt.title("Median analysis (Euclidean Distance) \n train data", fontsize=16)
    plt.xlabel("Closest Class Median", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("../analysis_of_embedding_distance/normalised_heatmap_dino_norm_train.pdf")
    plt.show()



# Example usage:
df = analyze_embeddings_and_save_csv('../normalised_embeddings/dino_normalised_embeddings_train.npz',
                                    out_csv_path='../analysis_of_embedding_distance/median_classification_summary_dino_norm_train.csv',
                                    tie_atol=1e-8,
                                    distribute_ties_fractionally=False)
print(df)
plot_pretty_confusion_matrices(df)
