import os
import shutil
import re

def organize_dataset(input_dir, output_dir, num_classes=9):
    """
    Organizes images into class subfolders based on the last digit before '.jpg'.
    Moves files (does not copy).

    Example:
    input_dir/
        image_001_5.jpg
        image_002_7.jpg
    → output_dir/
        0/
        1/
        ...
        8/
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create class subfolders (0-8)
    for i in range(1,10):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    # Regex to find the last digit before .jpg (e.g., "image_7.jpg" -> "7")
    pattern = re.compile(r"(\d)(?=\.jpg$)", re.IGNORECASE)

    count = 0
    skipped = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".jpg"):
            continue

        match = pattern.search(filename)
        if not match:
            print(f"⚠️ Skipping file (no label found): {filename}")
            skipped += 1
            continue

        label = match.group(1)
        src_path = os.path.join(input_dir, filename)
        dest_path = os.path.join(output_dir, label, filename)

        # Move the file
        shutil.move(src_path, dest_path)
        count += 1

    print(f"✅ Done. Moved {count} images from '{input_dir}' to '{output_dir}'.")
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} files (no valid class digit found).")


def main():
    base_input = "../dataset/cropped/"  # adjust if needed
    base_output = "../dataset/dataset_prepared"

    folders = ["cropped_train", "cropped_test"]

    for folder in folders:
        input_dir = os.path.join(base_input, folder)
        output_dir = os.path.join(base_output, folder)

        if not os.path.exists(input_dir):
            print(f"❌ Input folder not found: {input_dir}")
            continue

        print(f"\n📂 Processing: {input_dir}")
        organize_dataset(input_dir, output_dir, num_classes=9)

if __name__ == "__main__":
    main()
