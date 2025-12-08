import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import os

def extract_class_from_path(path):
    """Extract class from image path (last digit before .jpg)"""
    filename = os.path.basename(path)
    class_str = filename.split('.')[0].split('_')[-1]
    return int(class_str)

def plot_2d(embeddings_2d, classes, image_paths):
    """Plot 2D UMAP embeddings"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=classes, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('2D UMAP Embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True)
    plt.savefig('umap_2d.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d(embeddings_3d, classes, image_paths):
    """Plot 3D UMAP embeddings (interactive)"""
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=classes,
        title='3D UMAP Embeddings',
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
        opacity=0.7,
        width=1000,
        height=800
    )
    fig.write_html('umap_3d.html')

def main(npz_path):
    # Load data
    data = np.load(npz_path)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    # Extract classes
    classes = np.array([extract_class_from_path(path) for path in image_paths])

    # UMAP
    reducer_2d = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='euclidean', n_components=2, random_state=42)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    reducer_3d = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='euclidean', n_components=3, random_state=42)
    embeddings_3d = reducer_3d.fit_transform(embeddings)

    # Plot
    plot_2d(embeddings_2d, classes, image_paths)
    plot_3d(embeddings_3d, classes, image_paths)

    # Save UMAP embeddings
    np.savez('../umap_embeddings.npz', embeddings_2d=embeddings_2d, embeddings_3d=embeddings_3d, classes=classes, image_paths=image_paths)
    print("UMAP embeddings saved to umap_embeddings.npz")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, help='Path to the .npz file')
    args = parser.parse_args()
    main(args.npz_path)
