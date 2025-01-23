import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from PIL import Image


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
    label_files = sorted(os.listdir(label_path))  # Sort ground truths by filename
    for label_file in label_files:
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
                labels.append(int(class_id))  # Ensure class_id is an integer

        ground_truth.append({
            "image_id": image_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)  # Ensure labels are int64
        })

    return ground_truth


def run_inference(model, test_images_path):
    """
    Run inference on test images and format the predictions.
    Args:
        model (YOLO): Trained YOLO model.
        test_images_path (str): Path to the test images.

    Returns:
        List[dict]: List of dictionaries containing predicted boxes, scores, and labels.
    """
    predictions = []
    test_images = sorted([os.path.join(test_images_path, img) for img in os.listdir(test_images_path) if img.endswith((".jpg", ".png"))])  # Sort image paths

    for img_path in test_images:
        result = model(img_path)  # Run inference

        # Extract predictions
        if len(result[0].boxes) > 0:  # Check if any bounding boxes were detected
            preds = torch.from_numpy(result[0].boxes.xyxy.cpu().numpy())  # Bounding boxes in [x_min, y_min, x_max, y_max]
            confs = torch.from_numpy(result[0].boxes.conf.cpu().numpy())  # Confidence scores
            classes = torch.from_numpy(result[0].boxes.cls.cpu().numpy()).int()  # Predicted classes as integers
        else:
            preds = torch.zeros((0, 4), dtype=torch.float32)  # No bounding boxes
            confs = torch.zeros((0,), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # Add to predictions
        predictions.append({
            "image_id": os.path.basename(img_path).split(".")[0],  # Image ID without extension
            "boxes": preds,
            "scores": confs,
            "labels": classes
        })

    return predictions


def compute_map(ground_truth, predictions):
    """
    Compute mAP using torchmetrics' MeanAveragePrecision module.
    Args:
        ground_truth (list): Ground truth data as a list of dictionaries.
        predictions (list): Predictions data as a list of dictionaries.

    Returns:
        dict: mAP results including mAP@0.5 and mAP@0.5:0.95.
    """
    map_metric = MeanAveragePrecision(iou_type="bbox")
    for gt, pred in zip(ground_truth, predictions):
        map_metric.update([pred], [gt])

    return map_metric.compute()


def main():
    # Define paths
    test_images_path = "split_dataset/test/images"
    test_labels_path = "split_dataset/test/labels"
    model_path = "runs/model_3/weights/best.pt"

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO(model_path)  # Load YOLO model
    model.to(device)

    # Load ground truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth(test_labels_path, test_images_path)

    # Run inference
    print("Running inference...")
    predictions = run_inference(model, test_images_path)

    # Sort ground truth and predictions by image ID
    ground_truth = sorted(ground_truth, key=lambda x: x["image_id"])
    predictions = sorted(predictions, key=lambda x: x["image_id"])

    # Compute mAP
    print("Computing mAP...")
    map_results = compute_map(ground_truth, predictions)
    print("mAP Results:")
    print(f"mAP@0.5: {map_results['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {map_results['map']:.4f}")


if __name__ == "__main__":
    main()
