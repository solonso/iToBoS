from ultralytics import YOLO
import torchvision
import yaml

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 26
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")
 
args_file_path = 'params.yaml'  # Replace with the actual path to your args.yaml

with open(args_file_path, 'r') as file:
    args = yaml.safe_load(file)

model = YOLO("yolov9e.pt") 
# print(model.model)  
# train_path = "split_dataset/train/images/"
# Trueval_path = "split_dataset/val"

model.add_callback("on_train_start", freeze_layer)
 
results = model.train(
    data="params.yaml",#args.get("path"),
    imgsz=args.get("imgsz"),  # Resize images to 640x640 (you can try 800x800 based on GPU memory)
    epochs=args.get("epochs"),  # Number o+++++++++++++++++++++++f epochs
    batch=args.get("batch"),  # Batch size
    workers=args.get("workers"),  # Number of workers for data loading
    project=args.get("project"),  # Output directory
    name="model_3_layer",  # Experiment name
    exist_ok=True,  # Restart training if the output directory exists
    patience=10,  # Early stopping patience (epochs)
    plots=True,  # Generate plots during training
    device='0',  # Use GPU 0; set to -1 for CP
    rect=True,  # Enable rectangular batching to preserve aspect ratio
    augment=True,  # Apply augmentations (e.g., flip, rotation, brightness)
)
