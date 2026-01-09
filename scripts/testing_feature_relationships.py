import numpy as np
import os
import re

def extract_class_from_path(path):
    basename = os.path.basename(path)
    name, _ext = os.path.splitext(basename)
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError(f"No trailing digit(s) found in filename '{basename}'")
    return int(m.group(1))

def analyze_distance_suitability(file_path):
    # 1. Load the .npz file
    data = np.load(file_path)
    embeddings = data['embeddings']
    paths = data['image_paths']
    
    # 2. Extract labels using your regex logic
    labels = np.array([extract_class_from_path(p) for p in paths])
    unique_classes = np.unique(labels)
    
    print(f"{'Class':<8} | {'Samples':<8} | {'Avg Corr':<10} | {'Condition Num':<12}")
    print("-" * 45)

    results = {}

    for cls in unique_classes:
        class_data = embeddings[labels == cls]
        n_samples, n_features = class_data.shape
        
        # Check for sample size vs dimensionality
        if n_samples <= n_features:
            print(f"Warning: Class {cls} has fewer samples ({n_samples}) than dimensions ({n_features}).")
            print("Mahalanobis distance will likely be unstable/singular.")

        # 3. Calculate Correlation Matrix
        corr_matrix = np.corrcoef(class_data, rowvar=False)
        # Average of absolute off-diagonal correlations
        avg_corr = np.mean(np.abs(np.triu(corr_matrix, k=1)))

        # 4. Calculate Condition Number via Covariance
        cov_matrix = np.cov(class_data, rowvar=False)
        # We add a small epsilon to the diagonal to avoid division by zero
        eigenvalues = np.linalg.eigvalsh(cov_matrix + np.eye(n_features) * 1e-6)
        cond_num = np.max(eigenvalues) / np.min(eigenvalues)
        
        results[cls] = {
            "avg_correlation": avg_corr,
            "condition_number": cond_num,
            "samples": n_samples
        }
        
        print(f"{cls:<8} | {n_samples:<8} | {avg_corr:<10.4f} | {cond_num:<12.2e}")

    return results

# Usage:
analyze_distance_suitability('../normalised_embeddings/dino_normalised_embeddings_train.npz')