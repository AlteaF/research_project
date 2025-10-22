import cv2
import os

def convert_to_grayscale(image_path, output_path):
    """Converts an image to grayscale and saves it."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_img)
        print(f"Successfully converted {image_path} to {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_dataset(input_folder, output_folder):
    """Processes all images in a folder to grayscale."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_to_grayscale(image_path, output_path)

# Example Usage
input_dir = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train" # Replace with your input directory
output_dir = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train_gray" # Replace with your desired output directory
process_dataset(input_dir, output_dir)
