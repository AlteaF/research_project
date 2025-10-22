import numpy as np
import argparse
from sklearn.preprocessing import normalize

def normalize_embeddings(input_file, output_file):
    # Load embeddings and image paths
    data = np.load(input_file, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]

    # Normalize embeddings
    
    embeddings_normalized = normalize(embeddings, axis=0)

    # Save normalized embeddings
    np.savez(output_file, embeddings=embeddings_normalized, image_paths=image_paths)
    print(f"✅ Normalized embeddings saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Normalize embeddings from an .npz file using sklearn's Normalizer.")
    parser.add_argument('--input_file', type=str, required=True, help="Input .npz file containing embeddings and image_paths.")
    parser.add_argument('--output_file', type=str, required=True, help="Output .npz file with normalized embeddings.")
    args = parser.parse_args()

    normalize_embeddings(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
