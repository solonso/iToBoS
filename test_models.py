import os
import torch
import csv
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from MedViT.MedViT import MedViT_small as tiny
from tqdm import tqdm
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt
import cv2 

# =================== CONFIGURATION ===================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_CONFIDENCE = 0.325  # YOLO confidence threshold
YOLO_IOU = 0.45          # YOLO IoU threshold
# CLASSIFY_THRESHOLD = 0.5  # MedViT classification threshold (adjust if needed)
IMAGE_SIZE = 256  # MedViT input size
CSV_OUTPUT_PATH = "filtered_predictions1.csv"
YOLO_OUTPUT="bbox_preds/"
MVIT_OUTPUT="bmvit_classify/"
# Paths
MODEL_YOLO = "/home/mfa/My_Data/Semester1/ML/iToBoS/pipeline_models/lession_0_6702/lesion_detection_model33/weights/best.pt"  # Path to trained YOLO model
MODEL_MEDVIT = "pipeline_models/model_newdata_1/checkpoint_best.pth"  # Path to trained MedViT model
TEST_IMAGES_PATH = "/home/mfa/Documents/iToBoS/_test/images"  # Path to test images

# =================== LOAD MODELS ===================
print("🔹 Loading YOLO model...")
yolo_model = YOLO(MODEL_YOLO).to(DEVICE)

print("🔹 Loading MedViT model...")
medvit_model = tiny()
medvit_model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=2, bias=True)  # Adjust for 2-class classification
medvit_model.load_state_dict(torch.load(MODEL_MEDVIT, map_location=DEVICE)["model"])
medvit_model.to(DEVICE).eval()

# =================== IMAGE TRANSFORM FOR MEDVIT ===================
# transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     # transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# =================== FUNCTION: PROCESS BOUNDING BOXES ===================

def plot_bbox(image_path, orig_bbox, expanded_bbox):
    """
    Plots the original and expanded bounding boxes on an image.

    Args:
        image_path (str): Path to the image file.
        orig_bbox (tuple): Original bounding box (x_min, y_min, x_max, y_max).
        expanded_bbox (tuple): Expanded bounding box (x_min, y_min, x_max, y_max).
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw original bounding box in red
    draw.rectangle(orig_bbox, outline="red", width=3)
    draw.text((orig_bbox[0], orig_bbox[1] - 10), "Original", fill="red")

    # Draw expanded bounding box in blue
    draw.rectangle(expanded_bbox, outline="blue", width=3)
    draw.text((expanded_bbox[0], expanded_bbox[1] - 10), "Expanded", fill="blue")

    # Show the image with bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def expand_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    """ Expands the bbox to be 240x240 centered at the original bbox. """
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    crop_x1 = max(0, int(x_center - IMAGE_SIZE / 2))
    crop_y1 = max(0, int(y_center - IMAGE_SIZE / 2))
    crop_x2 = min(img_width, int(x_center + IMAGE_SIZE / 2))
    crop_y2 = min(img_height, int(y_center + IMAGE_SIZE / 2))

    return crop_x1, crop_y1, crop_x2, crop_y2

# =================== FUNCTION: CLASSIFY CROPPED IMAGES ===================
def classify_patch(cropped_image,img_path,i):
    """ Classifies a cropped 240x240 lesion image using MedViT. """
    img_tensor = torch.tensor(np.array(cropped_image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = medvit_model(img_tensor)
        probs = torch.softmax(output, dim=1)  # Multi-class probabilities
        preds = torch.argmax(probs, dim=1).cpu().numpy()  # Class with highest probability
    
    cropped_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    if preds.item() == 0: cv2.putText(cropped_image, f"lesion_{float(probs[0][preds.item()])}",fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,org=(50,30), color=(0,255,0), thickness=1)
    else: cv2.putText(cropped_image, f"non_lesion_{float(probs[0][preds.item()])}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,org=(50,30),color=(0,255,255), thickness=1)
    cv2.imwrite(MVIT_OUTPUT + f"{img_path.split('/')[-1].split('.')[0]}_{i}.png", cropped_image)
    return preds.item() == 0, probs[0, preds.item()].item()    # Return classification decision & probability

# =================== FUNCTION: RUN INFERENCE ===================
def run_pipeline():
    """
    Runs the full pipeline: YOLO detection -> MedViT classification -> Filtered CSV output.
    """
    predictions = []
    total_detections = 0
    total_filtered = 0
    total_valid_lesions = 0

    print("\n🚀 Running inference on test images...")
    test_images = [os.path.join(TEST_IMAGES_PATH, img) for img in os.listdir(TEST_IMAGES_PATH) if img.endswith((".jpg", ".png"))]
    yolo_model.eval()
    for img_path in tqdm(test_images):
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path)
        img_width, img_height = image.size

        # Run YOLO detection
        results = yolo_model(img_path, conf=YOLO_CONFIDENCE, iou=YOLO_IOU, imgsz=1024, verbose=False)
        annotated_images = results[0].plot()
        # Convert the annotated image from BGR to RGB
        # annotated_image = cv2.cvtColor(annotated_images[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(YOLO_OUTPUT+f"{img_path.split('/')[-1]}", annotated_images)
        prediction_string = " "
        per_image_detections = 0
        per_image_filtered = 0

        # print(f"\n📌 Image: {base_filename}")
        # print(f"🔍 YOLO detected {len(results[0].boxes)} potential lesions...")

        for i,box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()  # Absolute bounding box (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(float , xyxy)
            # plot_bbox(box.xyxy[0])
            # Expand bbox to 240x240
            crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(x_min, y_min, x_max, y_max, img_width, img_height)

            # Crop and classify
            cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            if float(box.conf)>0.6:
                is_lesion =True
            else: is_lesion, prob = classify_patch(cropped_img,img_path,i)

            if is_lesion:  # Only keep valid detections
                prediction_string += f"{x_min} {y_min} {x_max} {y_max} 0 "
                per_image_detections += 1
                # print(f"✅ Lesion confirmed by MedViT (Prob: {prob:.4f})")
            else:
                prediction_string+=prediction_string
                per_image_filtered += 1
                # print(f"❌ False Positive removed by MedViT (Prob: {prob:.4f})")

        # Track statistics
        total_detections += len(results[0].boxes)
        total_filtered += per_image_filtered
        total_valid_lesions += per_image_detections

        # Format final output
        prediction_string = prediction_string.strip()
        predictions.append(f"{base_filename},{prediction_string}")

    # Save results to CSV
    with open(CSV_OUTPUT_PATH, "w") as f:
        f.write("image_id,prediction_string\n")  # Write header
        f.write("\n".join(predictions))

    # Print summary
    print("\n📊 **Summary of Inference Results**")
    print(f"🔹 Total YOLO Detections: {total_detections}")
    print(f"🔹 Total False Positives Removed: {total_filtered}")
    print(f"🔹 Total Valid Lesions Detected: {total_valid_lesions}")
    print(f"\n✅ Results saved to {CSV_OUTPUT_PATH}")

# =================== RUN PIPELINE ===================
if __name__ == "__main__":
    run_pipeline()
