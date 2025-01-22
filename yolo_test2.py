import os
import csv
import numpy as np
from ultralytics import YOLO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image  # Required for image size extraction

def load_ground_truth(label_path, image_folder):
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

def get_image_id(image_path):
    # Extract the image name without extension
    image_name = os.path.basename(image_path).split(".")[0]
    # Extract the ID (assumes the format is 'image_<ID>')
    image_id = int(image_name.split("_")[1])  # Extract numeric ID
    return image_id

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
        prediction_string = " "
        if len(preds) >0:
            for box, cls in zip(preds,labels):
                prediction_string += f" {box[0]} {box[1]} {box[2]} {box[3]} {labels[0]}"
            prediction_string = prediction_string.strip()  # Remove trailing space
        else:
            prediction_string = " "   
        output_lines.append(f"{image_id},{prediction_string}")

    return output_lines

def test(model_pth,test_images_path,test_labels_path,compute_mAP=False):
    # Step 1: Load the trained YOLO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO(model_pth)  # Replace with your trained weights path
    model.to(device)
 
    # Ensure the folder exists and is not empty
    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images path does not exist: {test_images_path}")
    if len(os.listdir(test_images_path)) == 0:
        raise ValueError(f"No images found in test images path: {test_images_path}")
    # Step 3: List all test images
    test_images = [os.path.join(test_images_path, img) for img in os.listdir(test_images_path) if img.endswith((".jpg", ".png"))]
    if len(test_images) == 0:
        raise ValueError(f"No valid image files found in {test_images_path} (supported: .jpg, .png)")
    if compute_mAP:
    # Ensure the folder exists
        if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
            raise FileNotFoundError("Test images or labels path does not exist.")

    # Step 4: Run inference on test images
    predictions = []
    for img_path in test_images:
        result = model(img_path)  # Get predictions for each image

        # Check if there are predictions, and provide valid empty tensors if not
        if len(result[0].boxes) > 0:  # Check if any bounding boxes were detected
            preds = result[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_center, y_center, width, height]
            confs = result[0].boxes.conf.cpu().numpy()  # Confidence scores
            classes = result[0].boxes.cls.cpu().numpy()  # Predicted classes
        else:
            preds = torch.zeros((0, 4))  # No bounding boxes
            confs = torch.zeros((0,))
            classes = torch.zeros((0,))

        # Store predictions for this image
        predictions.append({
            "image": os.path.basename(img_path).split(".")[0],  # Use the image name without extension
            "boxes": preds,
            "scores": confs,
            "labels": classes,  # Ensure labels are integers
        })

    if compute_mAP:
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


    # Generate formatted predictions
    formatted_predictions = format_predictions(predictions)
    # Step 6: Save results to a text file
    output_file = "predictions.csv"
    with open(output_file, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(["image_id", "prediction_string"])
        # Write each formatted prediction line
        for line in formatted_predictions[1:]:  # Skip the header from `formatted_predictions`
            image_id, prediction_string = line.split(",", 1)
            csvwriter.writerow([image_id, prediction_string])
    print(f"Predictions saved to {output_file}")



test(model_pth="runs/model_auto_opt1/weights/best.pt",
     test_images_path="/home/mfa/My_Data/Semester1/ML/iToBoS/itobos-2024-detection/_test/images",
     test_labels_path= None,#"split_dataset/test/labels",
     compute_mAP=False
     )