import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
import os
import re
import numpy.random as npr

# ---- Edit this mapping in-code ----
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
# -----------------------------------

IMAGE_BASE_DIR = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train"
def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]
    return embeddings, image_paths

def extract_numeric_label_from_path(path):
    """
    Extract the last contiguous digits right before the extension.
    Supports .jpg/.jpeg/.png (case-insensitive).
    Example: 'dir/foo_123_7.jpg' -> 7
             'dir/another-345.png' -> 345
    Raises ValueError if nothing found.
    """
    filename = os.path.basename(path)
    match = re.search(r"(\d+)(?=\.(?:jpe?g|png)$)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract numeric label from filename: {filename}")
    return int(match.group(1))

def resolve_image_path(img_path, npz_base_dir):
    """
    Resolve image paths using the global IMAGE_BASE_DIR first.
    """
    # 1) Try inside IMAGE_BASE_DIR
    candidate = os.path.join(IMAGE_BASE_DIR, os.path.basename(img_path))
    if os.path.exists(candidate):
        return candidate

    # 2) Try as provided
    if os.path.exists(img_path):
        return img_path

    # 3) Try relative to the .npz directory
    candidate2 = os.path.join(npz_base_dir, img_path)
    if os.path.exists(candidate2):
        return candidate2

    # 4) Try basename inside the .npz directory
    candidate3 = os.path.join(npz_base_dir, os.path.basename(img_path))
    if os.path.exists(candidate3):
        return candidate3

    # Not found
    return None


def plot_tsne_with_images(
    embeddings, image_paths, labels,
    label_type='labels', output_path=None,
    samples_per_label=1, perplexity=30, random_state=42,
    show_images=True, npz_base_dir="."
):
    print(f"Running t-SNE on {len(embeddings)} vectors (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)

    # labels should be the mapped word labels (strings)
    mapped_labels = list(labels)
    counts = Counter(mapped_labels)
    print("Counts per label:")
    for lab, c in counts.items():
        print(f"  {lab}: {c}")

    # Unique labels in stable order
    unique_labels = sorted(counts.keys(), key=lambda x: (str(x).lower(), str(x)))
    n_labels = len(unique_labels)
    cmap = plt.get_cmap('tab20')  # use tab20 for more colors

    # Choose a color for each label
    label_to_color = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}

    plt.figure(figsize=(12, 8))

    # Plot points per label so counts and legend align with plotted points
    for lab in unique_labels:
        indices = [i for i, l in enumerate(mapped_labels) if l == lab]
        coords = tsne_results[indices]
        plt.scatter(coords[:, 0], coords[:, 1],
                    s=20, alpha=0.7, label=f"{lab} ({len(indices)})",
                    color=label_to_color[lab])

    # Build legend with word labels (no numeric colorbar)
    legend_handles = [Patch(color=label_to_color[lab], label=f"{lab} ({counts[lab]})") for lab in unique_labels]
    plt.legend(handles=legend_handles, title=label_type, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add thumbnails (one or more per label)
    if show_images:
        print("Attempting to add thumbnails...")
        # group mapped label -> indices
        grouped = defaultdict(list)
        for i, lab in enumerate(mapped_labels):
            grouped[lab].append(i)

        rng = npr.default_rng(random_state)
        missing_images = 0
        added_images = 0

        for lab, indices in grouped.items():
            # sample indices for this label
            sample_count = min(samples_per_label, len(indices))
            chosen = rng.choice(indices, size=sample_count, replace=False)
            for idx in chosen:
                img_path_candidate = resolve_image_path(image_paths[idx], npz_base_dir)
                if img_path_candidate is None:
                    missing_images += 1
                    # don't crash, continue
                    print(f"  ⚠️ Missing image for index {idx}: tried '{image_paths[idx]}' and fallbacks.")
                    continue
                try:
                    img = Image.open(img_path_candidate).convert('RGB')
                    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    imagebox = OffsetImage(np.asarray(img), zoom=1)
                    ab = AnnotationBbox(imagebox,
                                        (tsne_results[idx, 0], tsne_results[idx, 1]),
                                        frameon=False, pad=0.3)
                    plt.gca().add_artist(ab)
                    added_images += 1
                except Exception as e:
                    missing_images += 1
                    print(f"  ⚠️ Could not open/create thumbnail for '{img_path_candidate}': {e}")

        print(f"Thumbnails: added {added_images}, missing {missing_images}")

    plt.title(f"t-SNE {output_path}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="t-SNE on embeddings with filename-extracted labels and embedded label map.")
    parser.add_argument("--embedding_file", type=str, required=True, help="Path to .npz file with 'embeddings' and 'image_paths'")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--samples_per_label", type=int, default=3)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--show_images", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    embeddings, image_paths = load_npz_data(args.embedding_file)
    npz_base_dir = os.path.dirname(os.path.abspath(args.embedding_file)) or "."

    # Extract numeric labels robustly
    numeric_labels = []
    failed_indices = []
    for i, p in enumerate(image_paths):
        try:
            numeric_labels.append(extract_numeric_label_from_path(p))
        except ValueError as e:
            # collect, but continue so we can debug
            failed_indices.append((i, p))
            numeric_labels.append(None)

    if failed_indices:
        print("Warning: some filenames did not yield numeric labels (listed below). They will get label 'unknown'.")
        for i, p in failed_indices:
            print(f"  index {i}: {p}")

    # Map numeric -> word label using LABEL_MAP; fallback to "label_<num>" or "unknown"
    mapped_labels = []
    for num in numeric_labels:
        if num is None:
            mapped_labels.append("unknown")
        else:
            mapped_labels.append(LABEL_MAP.get(num, f"label_{num}"))

    # Sanity: make sure labels length matches embeddings
    if len(mapped_labels) != len(embeddings):
        raise RuntimeError("Number of labels does not match number of embeddings.")

    plot_tsne_with_images(
        embeddings=embeddings,
        image_paths=image_paths,
        labels=mapped_labels,
        label_type="label (name)",
        output_path=args.output_file,
        samples_per_label=args.samples_per_label,
        perplexity=args.perplexity,
        show_images=args.show_images,
        npz_base_dir=npz_base_dir
    )

if __name__ == "__main__":
    main()
