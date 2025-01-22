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

model = YOLO("yolov8s.pt") 
# print(model.model)  
# train_path = "split_dataset/train/images/"
# Trueval_path = "split_dataset/val"

# model.add_callback("on_train_start", freeze_layer)
 
results = model.train(
    data="params.yaml",      # Dataset configuration
    imgsz=640,               # Reduced image size for stability
    epochs=100,              # Number of epochs
    batch=32,                # Safer batch size
    single_cls=True,         # Single-class detection
    warmup_epochs=3,         # Enable warmup
    optimizer="AdamW",       # Use AdamW optimizer
    lr0=0.003,               # Reduced learning rate
    # lrf=0.1,                 # Learning rate final multiplier
    workers=8,               # Number of data loading workers
    project=args.get("project"),
    name="model_auto_opt1",
    exist_ok=True,           # Restart training if the output directory exists
    patience=5,              # Early stopping patience
    plots=True,              # Generate training plots
    device='0',              # Use GPU 0
    val=True,                # Perform validation
    rect=True,               # Rectangular training batches
    
)
