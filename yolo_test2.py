import os
import numpy as np
from ultralytics import YOLO
import torch

# Step 1: Load the trained YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO("runs/model_3_layer/weights/best.pt")  # Replace with your trained weights path
model.to(device)

# Step 2: Define paths for test images
test_images_path = "dummy_data"  # Path to test images folder
# Ensure the folder exists and is not empty
if not os.path.exists(test_images_path):
    raise FileNotFoundError(f"Test images path does not exist: {test_images_path}")
if len(os.listdir(test_images_path)) == 0:
    raise ValueError(f"No images found in test images path: {test_images_path}")

# Step 3: List all test images
test_images = [os.path.join(test_images_path, img) for img in os.listdir(test_images_path) if img.endswith((".jpg", ".png"))]
if len(test_images) == 0:
    raise ValueError(f"No valid image files found in {test_images_path} (supported: .jpg, .png)")

# Step 4: Run inference on test images
predictions = []
for img_path in test_images:
    result = model(img_path)  # Get predictions for each image
    
    # Check if there are predictions, and provide valid empty tensors if not
    if len(result[0].boxes) > 0:  # Check if any bounding boxes were detected
        preds = result[0].boxes.xywhn.cpu().numpy()  # Bounding boxes in [x_center, y_center, width, height]
        confs = result[0].boxes.conf.cpu().numpy()  # Confidence scores
        classes = result[0].boxes.cls.cpu().numpy()  # Predicted classes
    else:
        preds = np.zeros((0, 4))  # No bounding boxes
        confs = np.zeros((0,))
        classes = np.zeros((0,))
    
    # Store predictions for this image
    predictions.append({
        "image": os.path.basename(img_path).split(".")[0],  # Use the image name without extension
        "boxes": preds,
        "scores": confs,
        "labels": classes.astype(int),  # Ensure labels are integers
    })

# Step 5: Format predictions for output
def format_predictions(predictions):
    """
    Formats predictions into a CSV-style format for saving.
    """
    output_lines = ["image_id,prediction_string"]  # Add header

    for pred in predictions:
        image_id = pred["image"]
        preds = pred["boxes"]
        confs = pred["scores"]
        labels = pred["labels"]

        prediction_string = ""
        if len(preds) > 0:
            for box, cls in zip(preds,labels):
                prediction_string += f"{box[0]} {box[1]} {box[2]} {box[3]}"

        prediction_string = prediction_string.strip()  # Remove trailing space
        output_lines.append(f"{image_id},{prediction_string}")

    return output_lines

# Generate formatted predictions
formatted_predictions = format_predictions(predictions)

# Step 6: Save results to a text file
output_file = "predictions.txt"
with open(output_file, "w") as f:
    f.write("\n".join(formatted_predictions))

print(f"Predictions saved to {output_file}")
