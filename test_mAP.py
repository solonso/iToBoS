import numpy as np
import os
from ultralytics import YOLO
import torch
from sklearn.metrics import average_precision_score

# Define IoU function
def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/model_3_layer/weights/best.pt").to(device)

# Load test images and ground truths
test_images_path = "dummy_data"
test_images = [os.path.join(test_images_path, img) for img in os.listdir(test_images_path) if img.endswith((".jpg", ".png"))]

# Placeholder for ground truth data loading
# This should be replaced with actual ground truth data loading logic
ground_truths = {}

# Run inference and calculate IoUs
predictions = []
ious = []
for img_path in test_images:
    result = model(img_path)
    preds = result[0].boxes.xywhn.cpu().numpy() if len(result[0].boxes) > 0 else np.zeros((0, 4))
    confs = result[0].boxes.conf.cpu().numpy() if len(result[0].boxes) > 0 else np.zeros((0,))
    classes = result[0].boxes.cls.cpu().numpy() if len(result[0].boxes) > 0 else np.zeros((0,))

    # Ground truth for the current image
    gt_boxes = ground_truths.get(os.path.basename(img_path))

    # Calculate IoU for each prediction against each ground truth
    img_ious = []
    for pred in preds:
        for gt in gt_boxes:
            iou = calculate_iou(pred, gt)
            img_ious.append(iou)
    ious.extend(img_ious)

    predictions.append({
        "image": os.path.basename(img_path).split(".")[0],
        "boxes": preds,
        "scores": confs,
        "labels": classes.astype(int),
        "ious": img_ious
    })

# Calculate mAP
all_ious = np.array(ious)
thresholds = np.arange(0.5, 0.76, 0.05)
aps = []
for threshold in thresholds:
    labels = (all_ious > threshold).astype(int)
    ap = average_precision_score(labels, all_ious)
    aps.append(ap)
map_score = np.mean(aps)

print(f"Mean Average Precision: {map_score}")
