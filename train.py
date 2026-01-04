
from ultralytics import YOLO
import os
    
# --- 2TRAINING ROUTINE ---
def train():
    """
    Submission Training Script
    Model: YOLOv8x (Extra Large)
    Target: Dual NVIDIA T4 GPUs
    """
    

    # Step 1: Locate data.yaml
    # POINT TO YOUR KAGGLE DATASET FOLDER
    data_file = "/kaggle/input/data12/data.yaml"
    
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Ensure the path is correct.")

    
    print("Initializing YOLOv8x for CRACKATHON_DATA Training...")
    
    # Load Model: Start with COCO pre-trained weights for transfer learning
    model = YOLO('yolov8x.pt')  

    # Start Training
    results = model.train(
        data=data_file,         # Path to the yaml we just created
        epochs=50,
        imgsz=640,
        
        # Hardware Optimization: Tuned for 2x T4 GPUs (VRAM efficient)
        batch=24,
        device=[0, 1],
        
        # Logging & Artifacts
        project='Crackathon_submission',
        name='yolov8x_run',
        save=True,
        save_period=5,
        
        # Hyperparameters (The "Winning" config)
        patience=15,            # Early stopping patience
        warmup_epochs=3,        # Warmup
        augment=True,           # Default robust augmentations
        cos_lr=True,            # Cosine Annealing Learning Rate
        close_mosaic=10,        # Disable Mosaic for the final 10 epochs (Precision Boost)
        optimizer='auto'
    )
    print("Training Complete.")

if __name__ == '__main__':
    train()
