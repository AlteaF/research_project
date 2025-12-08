import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# --- CONSTANTS ---
LABEL_MAP = {
    1: "Starfish", 2: "Crab", 3: "Black goby", 4: "Wrasse", 
    5: "Two-spotted goby", 6: "Cod", 7: "Painted goby", 8: "Sand eel", 
    9: "Whiting"
}
OUTPUT_DIR = "../mahala_median/"
FILTERED_EMB_DIR = "../mahala_filtered_embeddings/"
# Small constant for variance stabilization (prevent division by zero)
EPSILON = 1e-6 


# --- HELPER FUNCTIONS ---

def extract_class_from_path(path):
    """
    Extract the single digit label (1-9) immediately preceded by an underscore 
    and followed by the file extension.
    Example: 'dir/foo_123_7.jpg' -> 7
    """
    filename = os.path.basename(path)
    # Regex: find a single digit (\d) that is preceded by '_' and followed 
    # by a dot and the extension (non-dot characters till end $).
    match = re.search(r"\_(\d)(?=\.[^\.]*$)", filename, flags=re.IGNORECASE)
    
    if not match:
        return -1 
    return int(match.group(1))

def calculate_median_and_variance(embeddings, classes, unique_classes):
    """
    Calculates the median vector and the diagonal variance vector for each unique class.
    
    Returns:
        median_embeddings (dict): {cls: median_vector}
        variance_vectors (dict): {cls: variance_vector} (used for diagonal Mahalanobis)
    """
    median_embeddings = {}
    variance_vectors = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            print(f"Warning: No examples found for class {cls}. Skipping calculation.")
            continue 
        
        # We still use the median as the centroid for robustness against outliers
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
        
        # Calculate the variance along each dimension for the Mahalanobis distance
        # NOTE: np.var computes the sample variance along the axis
        variance_vectors[cls] = np.var(embeddings[mask], axis=0) + EPSILON
        
    return median_embeddings, variance_vectors

def calculate_mahalanobis_distance_sq(x_data, median_vector, variance_vector):
    """
    Calculates the squared diagonal Mahalanobis Distance: 
    sum((x_i - mu_i)^2 / sigma_i^2)
    
    Args:
        x_data (np.array): The data point(s) (N, D) or (D,).
        median_vector (np.array): The centroid (D,).
        variance_vector (np.array): The diagonal of the covariance matrix (D,).
        
    Returns:
        np.array: Squared Mahalanobis distance(s) (N,) or (1,).
    """
    # The term (x - mu)^2 / sigma^2
    normalized_diff_sq = (x_data - median_vector)**2 / variance_vector
    
    # Sum over the dimensions (D)
    # If x_data is (N, D), this sums over axis 1 resulting in (N,)
    # If x_data is (D,), this sums over the entire array resulting in a scalar
    return np.sum(normalized_diff_sq, axis=-1)


def create_confusion_df(embeddings, classes, medians_dict, variances_dict, unique_classes, tie_atol=1e-8):
    """
    Classifies embeddings based on the Mahalanobis Distance to the provided medians/variances 
    and returns a DataFrame suitable for confusion matrix plotting.
    """
    N = len(embeddings)
    C = len(unique_classes)
    
    # Initialize distances matrix (N, C)
    dists_sq = np.zeros((N, C))
    
    # 1. Calculate squared MD for all samples against all medians
    for j, cls in enumerate(unique_classes):
        # Calculate MD squared for ALL N samples to the median/variance of class cls
        dists_sq[:, j] = calculate_mahalanobis_distance_sq(
            embeddings, 
            medians_dict[cls], 
            variances_dict[cls]
        )
    
    # 2. Find closest median index
    min_dists = dists_sq.min(axis=1, keepdims=True)
    tie_mask = np.isclose(dists_sq, min_dists, atol=tie_atol)
    
    # 3. Initialize results table
    counts = {str(r): np.zeros(C, dtype=float) for r in unique_classes}
    
    for i in range(N):
        true_cls = classes[i]
        if true_cls not in unique_classes:
            continue
            
        tie_indices = np.nonzero(tie_mask[i])[0]
        
        # Tie handling (distribute ties fractionally)
        frac = 1.0 / tie_indices.size
        for pred_idx in tie_indices:
            counts[str(true_cls)][pred_idx] += frac

    # 4. Build DataFrame
    df_rows = []
    unique_classes_str = [str(c) for c in unique_classes]
    for cls in unique_classes:
        row = {'true_class': cls}
        for j, pred_cls_str in enumerate(unique_classes_str):
            row[f'pred_{pred_cls_str}'] = counts[str(cls)][j]
        df_rows.append(row)

    cols_order = ['true_class'] + [f'pred_{c}' for c in unique_classes]
    df = pd.DataFrame(df_rows)[cols_order].sort_values('true_class').reset_index(drop=True)
    return df

def plot_confusion_matrix(df, title_suffix, filename_suffix, distance_type="Mahalanobis"):
    """Plots a normalized confusion matrix from a classification DataFrame."""
    true_classes = df["true_class"].values
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_classes = [int(c.replace("pred_", "")) for c in pred_cols]
    
    # Raw confusion matrix
    cm_raw = df[pred_cols].values.astype(float)
    # Normalized confusion matrix (row-wise)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                xticklabels=[LABEL_MAP.get(c, str(c)) for c in pred_classes],
                yticklabels=[LABEL_MAP.get(c, str(c)) for c in true_classes])

    plt.title(f"{distance_type} Classification Accuracy (Row Normalized)\n{title_suffix}", fontsize=16)
    plt.xlabel("Predicted Class (Closest Median)", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot to: {output_path}")

# --- MAIN FUNCTION ---

def run_median_analysis_and_filter(input_npz_path):
    """
    Performs the 4 steps: initial analysis, plotting, filtering by MD, and 
    final plotting using original MD parameters.
    """
    print(f"1. Loading and preparing data from: {input_npz_path}")
    
    # Load Data
    data = np.load(input_npz_path, allow_pickle=True)
    embeddings = np.array(data['embeddings'])
    image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in data['image_paths']])
    
    # Extract and filter classes
    classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)
    valid_mask = (classes > 0)
    
    # Filter out samples where class extraction failed
    if valid_mask.sum() < len(classes):
        print(f"   ⚠️ Warning: Skipping {len(classes) - valid_mask.sum()} samples with invalid class ID.")
        embeddings = embeddings[valid_mask]
        image_paths = image_paths[valid_mask]
        classes = classes[valid_mask]
        
    unique_classes = np.unique(classes)
    unique_classes.sort()

    # 2a. Calculate Original Medians and Variances (the stable parameters)
    median_dict, variance_dict = calculate_median_and_variance(embeddings, classes, unique_classes)
    
    print(f"   Calculated {len(unique_classes)} class medians and variances for Mahalanobis Distance.")

    # 2b. Run Initial Median Classification & Plot Confusion Matrix
    print("2. Running initial Mahalanobis classification on all embeddings...")
    df_original = create_confusion_df(embeddings, classes, median_dict, variance_dict, unique_classes)
    plot_confusion_matrix(df_original, 
                          "Before Filtering (Mahalanobis Distance)", 
                          "mahalanobis_analysis_original_all")

    # 3a. Filter Embeddings (Identify samples closest to their OWN median by MD)
    print("3. Filtering embeddings: Keeping only those closest to their own class median...")

    # The true class index in the unique_classes list
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    # Initialize distances matrix (N, C)
    N = len(embeddings)
    C = len(unique_classes)
    dists_sq = np.zeros((N, C))
    
    # Calculate squared MD for all samples against all medians/variances
    for j, cls in enumerate(unique_classes):
        dists_sq[:, j] = calculate_mahalanobis_distance_sq(
            embeddings, 
            median_dict[cls], 
            variance_dict[cls]
        )
    
    # Find the index of the closest median (by Mahalanobis Distance)
    pred_indices = np.argmin(dists_sq, axis=1)

    # Filtering mask: Keep if the closest median index equals the true class index
    true_class_indices = np.array([class_to_idx[cls] for cls in classes])
    closest_to_own_median_mask = (pred_indices == true_class_indices)
    
    filtered_embeddings = embeddings[closest_to_own_median_mask]
    filtered_image_paths = image_paths[closest_to_own_median_mask]

    # 3b. Save Filtered NPZ file
    output_filename = os.path.basename(input_npz_path).replace('.npz', '_filtered_mahalanobis.npz')
    output_npz_path = os.path.join(FILTERED_EMB_DIR, output_filename)
    os.makedirs(FILTERED_EMB_DIR, exist_ok=True)
    
    np.savez(output_npz_path, 
             embeddings=filtered_embeddings, 
             image_paths=filtered_image_paths)
    
    print("-" * 50)
    print(f"Original samples: {len(embeddings)}")
    print(f"Filtered samples (Closest to Own Median by MD): {len(filtered_embeddings)}")
    if len(embeddings) > 0:
        print(f"Percentage kept: {len(filtered_embeddings) / len(embeddings) * 100:.2f}%")
    print(f"✅ Filtered data saved to: **{output_npz_path}**")
    print("-" * 50)

    # 4. Plot the remaining (filtered) data using the ORIGINAL medians/variances
    print("4. Plotting final confusion matrix using the ORIGINAL Mahalanobis parameters...")
    filtered_classes = classes[closest_to_own_median_mask]
    
    # This classification MUST yield 100% accuracy relative to the original parameters.
    df_filtered = create_confusion_df(filtered_embeddings, filtered_classes, median_dict, variance_dict, unique_classes)
    plot_confusion_matrix(df_filtered, 
                          "After Filtering (Classified with ORIGINAL Mahalanobis Parameters)", 
                          "mahalanobis_analysis_filtered_100_percent")


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Median Analysis and Filtering using Mahalanobis Distance.")
    parser.add_argument("--embedding_file", 
                        type=str, 
                        default='../normalised_embeddings/dino_normalised_embeddings_train.npz', 
                        help="Path to the input .npz file with embeddings and image_paths.")
    args = parser.parse_args()

    # Ensure all output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FILTERED_EMB_DIR, exist_ok=True)
    
    try:
        run_median_analysis_and_filter(args.embedding_file)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found at {args.embedding_file}")
        print("Please check the path or provide a correct path via the --embedding_file argument.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")