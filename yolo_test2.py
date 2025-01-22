import os
import csv
import numpy as np
from ultralytics import YOLO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image  # To get image dimensions

def get_image_id(image_path):
    # Extract the image name without extension
    image_name = os.path.basename(image_path).split(".")[0]
    # Extract the ID (assumes the format is 'image_<ID>')
    image_id = int(image_name.split("_")[1])  # Extract numeric ID
    return image_id

def load_ground_truth(label_path, image_folder):
    """
    Load ground truth bounding boxes and labels from YOLO-format label files in `xywhn` format.
    Args:
        label_path (str): Path to the label files.
        image_folder (str): Path to the corresponding images to get dimensions.

    Returns:
        List[dict]: List of dictionaries containing ground truth boxes and labels.
    """
    ground_truth = []
    for label_file in sorted(os.listdir(label_path)):
        image_id = os.path.splitext(label_file)[0]
        label_file_path = os.path.join(label_path, label_file)
        image_path = os.path.join(image_folder, f"{image_id}.png")  # Adjust if images have different extensions

        # Get image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        boxes = []
        labels = []

        with open(label_file_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_min = (x_center - width / 2) * image_width
                y_min = (y_center - height / 2) * image_height
                x_max = (x_center + width / 2) * image_width
                y_max = (y_center + height / 2) * image_height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))

        ground_truth.append({
            "image_id": image_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })

    return ground_truth

# Step 1: Load the trained YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO("runs/model_33_flayer_2/weights/best.pt")  # Replace with your trained weights path
model.to(device)

# Step 2: Define paths for test images and labels
test_images_path = "split_dataset/test/images"
test_labels_path = "split_dataset/test/labels"

# Ensure the folder exists
if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
    raise FileNotFoundError("Test images or labels path does not exist.")

# Step 3: List all test images
test_images = [os.path.join(test_images_path, img) for img in os.listdir(test_images_path)]
sorted_test_images = sorted(test_images, key=get_image_id)

# Step 4: Run inference on test images
predictions = []
for img_path in sorted_test_images:
    result = model.predict(img_path, device=device)  # Get predictions for each image
    
    if len(result[0].boxes) > 0:  # Check if any bounding boxes were detected
        preds = result[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_min, y_min, x_max, y_max]
        confs = result[0].boxes.conf.cpu().numpy()  # Confidence scores
        classes = result[0].boxes.cls.cpu().numpy()  # Predicted classes
    else:
        preds = np.zeros((0, 4))  # No bounding boxes
        confs = np.zeros((0,))
        classes = np.zeros((0,))
    
    predictions.append({
        "image_id": os.path.basename(img_path).split(".")[0],  # Use the image name without extension
        "boxes": torch.tensor(preds, dtype=torch.float32),
        "scores": torch.tensor(confs, dtype=torch.float32),
        "labels": torch.tensor(classes, dtype=torch.int64)
    })

# Step 5: Load ground truth
ground_truth = load_ground_truth(test_labels_path, test_images_path)

# Step 6: Compute mAP
map_metric = MeanAveragePrecision(iou_type="bbox")
for pred, gt in zip(predictions, ground_truth):
    map_metric.update([pred], [gt])

map_results = map_metric.compute()
print("mAP Results:")
print(f"mAP@0.5: {map_results['map_50']:.4f}")
print(f"mAP@0.5:0.95: {map_results['map']:.4f}")
