import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
ROOT_DIR = "Lesion-Detection-Challange/itobos-2024-detection/"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
ADDITIONAL_DATA_DIR = os.path.join(ROOT_DIR, "additional_data")
OUTPUT_DIR = "split_dataset"

# Subdirectories for images and labels
IMAGES_SUBDIR = "images/"
LABELS_SUBDIR = "labels/"

# Output directories
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test")

# Function to create directory structure
def create_dir_structure(base_dir):
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, subset, IMAGES_SUBDIR), exist_ok=True)
        os.makedirs(os.path.join(base_dir, subset, LABELS_SUBDIR), exist_ok=True)

# Copy files
def copy_files(src, dst, file_list):
    for file in file_list:
        shutil.copy(file, os.path.join(dst, os.path.basename(file)))

# Get paired images and labels  Lesion-Detection-Challange/itobos-2024-detection/train/train/images
def get_image_label_pairs(images_dir, labels_dir):
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))])
    labels = [os.path.join(labels_dir, os.path.splitext(os.path.basename(img))[0] + ".txt") for img in images]
    return [(img, lbl) for img, lbl in zip(images, labels) if os.path.exists(lbl)]

# Main script
def main():
    # Ensure output directories exist
    create_dir_structure(OUTPUT_DIR)

    # Get all image-label pairs
    image_label_pairs = get_image_label_pairs(
        os.path.join(TRAIN_DIR, IMAGES_SUBDIR),
        os.path.join(TRAIN_DIR, LABELS_SUBDIR)
    )

    # Split into train, val, and test
    train_val_pairs, test_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)

    # Split data into respective directories
    for subset, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        images_output = os.path.join(OUTPUT_DIR, subset, IMAGES_SUBDIR)
        labels_output = os.path.join(OUTPUT_DIR, subset, LABELS_SUBDIR)
        for img, lbl in pairs:
            shutil.copy(img, os.path.join(images_output, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(labels_output, os.path.basename(lbl)))

    print(f"Data split completed! Files saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
