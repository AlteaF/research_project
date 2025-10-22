

import os
import torch
import timm
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

def extract_embeddings(image_dir, model, device, transform):
    image_paths = []
    embeddings = []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {image_path}")
            image_paths.append(image_path)

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.forward_features(image_tensor)  # Patch tokens including CLS
                embedding = features.mean(dim=1)  # Mean pool over tokens

            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings), image_paths

def save_embeddings(embeddings, image_paths, output_file):
    np.savez(output_file, embeddings=embeddings, image_paths=image_paths)
    print(f"Saved embeddings to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings using timm.")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output_file", required=True, help="Output file (.npz)")

    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load DINOv2 ViT-Small model from timm
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],  # DINOv2 normalization
            std=[0.5, 0.5, 0.5]
        )
    ])

    embeddings, image_paths = extract_embeddings(args.image_dir, model, device, transform)
    save_embeddings(embeddings, image_paths, args.output_file)

if __name__ == "__main__":
    main()

