import os
import cv2
import numpy as np
import tqdm

# =================== CONFIGURATION ===================
SOURCE_DIR = "split_dataset"  # Path to the original dataset
DEST_DIR = "split_dataset_blurred"  # Path to save the new dataset with blurred backgrounds
BLUR_STRENGTH = 25  # Strength of background blur (higher = more blur)

# =================== FUNCTION: DETECT SKIN ===================
def detect_skin(img_path):
    """ Detects skin in an image and returns a binary mask where skin is white and background is black. """
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ ERROR: Could not read image {img_path}. Skipping...")
        return None, None

    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))

    # Expand mask (buffer region)
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(YCrCb_mask, kernel, iterations=3)  

    # Find largest connected component (assumed to be skin)
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(expanded_mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = expanded_mask  # Use the original mask if no contours found

    return img, final_mask

# =================== FUNCTION: BLUR BACKGROUND ===================
def blur_background(img, skin_mask, blur_strength=BLUR_STRENGTH):
    """
    Blurs only the background while keeping the detected skin region sharp.
    
    Args:
        img: Original image (BGR format).
        skin_mask: Binary mask where skin is white and background is black.
        blur_strength: Strength of background blur (higher = more blur).
    
    Returns:
        Image with blurred background while keeping skin region sharp.
    """
    if img is None or skin_mask is None:
        return None  # Skip processing if image or mask is invalid

    # Convert skin mask to 3 channels (for RGB blending)
    mask_3channel = cv2.merge([skin_mask, skin_mask, skin_mask])

    # Apply Gaussian blur to the entire image
    blurred_img = cv2.GaussianBlur(img, (blur_strength, blur_strength), 15)

    # Keep skin region from original image and blend it with blurred background
    final_output = np.where(mask_3channel == 255, img, blurred_img)

    return final_output

# =================== FUNCTION: PROCESS ALL IMAGES ===================
def process_dataset(source_dir, dest_dir):
    """
    Processes all images in the dataset, applies background blurring, and saves them in the new folder structure.

    Args:
        source_dir: Path to the original dataset.
        dest_dir: Path to save the processed dataset.
    """
    subsets = ["train", "val", "test"]

    for subset in subsets:
        input_images_path = os.path.join(source_dir, subset, "images")
        output_images_path = os.path.join(dest_dir, subset, "images")

        # Ensure output directory exists
        os.makedirs(output_images_path, exist_ok=True)

        if not os.path.exists(input_images_path):
            print(f"❌ ERROR: {input_images_path} does not exist! Skipping...")
            continue

        image_files = [img for img in os.listdir(input_images_path) if img.endswith((".jpg", ".png"))]
        if len(image_files) == 0:
            print(f"⚠️ WARNING: No images found in {input_images_path}. Skipping...")
            continue

        print(f"📂 Processing {subset} set ({len(image_files)} images)...")

        # Process all images with tqdm progress bar
        for img_name in tqdm.tqdm(image_files, desc=f"Processing {subset} images"):
            img_path = os.path.join(input_images_path, img_name)
            output_path = os.path.join(output_images_path, img_name)

            # Process image
            image, skin_mask = detect_skin(img_path)
            blurred_image = blur_background(image, skin_mask)

            if blurred_image is not None:
                # 🔹 Ensure the image is in `uint8` format (cv2.imwrite() needs this!)
                blurred_image = cv2.convertScaleAbs(blurred_image)

                # 🔹 Check if `cv2.imwrite()` is successful
                saved = cv2.imwrite(output_path, blurred_image)
                if not saved:
                    print(f"❌ ERROR: Could not save image {output_path}")
            else:
                print(f"⚠️ WARNING: Could not process {img_name}. Skipping...")

        print(f"✅ {subset} set processed and saved at {output_images_path}")

# =================== RUN SCRIPT ===================
if __name__ == "__main__":
    process_dataset(SOURCE_DIR, DEST_DIR)
    print("\n🎉 All images processed and saved successfully!")
