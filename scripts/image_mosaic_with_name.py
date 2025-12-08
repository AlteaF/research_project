import os
import random
from PIL import Image, ImageDraw, ImageFont

# Example mapping: Replace with your actual class_id to name mapping
CLASS_ID_TO_NAME = {
        1: "Starfish",
        2: "Crab",
        3: "Black\n goby",
        4: "Wrasse",
        5: "Two-spotted\n goby",
        6: "Cod",
        7: "Painted\n goby",
        8: "Sand\n eel",
        9: "Whiting"
    }

def extract_class_from_filename(filename):
    """Extract the class label from the filename (last digit before .jpg)."""
    base = os.path.splitext(filename)[0]
    return int(base[-1])  # Assumes class is the last digit

def load_images_from_folder(folder, target_size=(128, 128)):
    """Load images from folder and group by class."""
    class_images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            class_label = extract_class_from_filename(filename)
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(target_size)
            if class_label not in class_images:
                class_images[class_label] = []
            class_images[class_label].append(img)
    return class_images

def create_mosaic(class_images, n_rows=9, n_cols=30, images_per_class=30):
    """
    Create a mosaic grid with 9 rows (one per class) and 30 columns.
    Each row contains 30 randomly selected images from the corresponding class.
    Class names are displayed on the left side of each row.
    """
    # Create a new image with space for class names
    mosaic = Image.new('RGB', (n_cols * 128 + 200, n_rows * 128), color='white')  # Extra 200px for text
    draw = ImageDraw.Draw(mosaic)
    font = ImageFont.load_default(size=33)  # Adjust size as needed

    for i, class_label in enumerate(sorted(class_images.keys())):
        images = class_images[class_label]
        if len(images) < images_per_class:
            raise ValueError(f"Not enough images for class {class_label}. Found {len(images)}, need {images_per_class}.")
        selected_images = random.sample(images, images_per_class)

        # Draw class name on the left
        class_name = CLASS_ID_TO_NAME.get(class_label, f"Class{class_label}")
        draw.text((10, i * 128 + 10), class_name, fill='black', font=font)

        # Paste images
        for j, img in enumerate(selected_images):
            mosaic.paste(img, (200 + j * 128, i * 128))  # Offset by 200px for text

    return mosaic

def main():
    folder = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train"  # Replace with your folder path
    class_images = load_images_from_folder(folder)
    mosaic = create_mosaic(class_images)
    mosaic.save("mosaic_9x30_with_labels.png")
    print("Mosaic saved as mosaic_9x30_with_labels.png")

if __name__ == "__main__":
    main()
