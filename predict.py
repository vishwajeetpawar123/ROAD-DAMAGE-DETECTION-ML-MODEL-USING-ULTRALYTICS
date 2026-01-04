import os
import shutil
import zipfile
import torch
import numpy as np
from tqdm.auto import tqdm

# ---INSTALL DEPENDENCIES---
os.system('pip install -q sahi ultralytics ensemble-boxes tqdm')

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ---CONFIGURATION---
MODEL_PATH = '/kaggle/input/my-data2/best.pt'
TEST_IMAGES_DIR = '/kaggle/input/crackathon-data/randomized_dataset/test/images'
OUTPUT_DIR = 'predictions'
ZIP_NAME = 'submission_sahi_nmm.zip'

# ---PRIDICTION CONFIGURATION---
CONF_THRES = 0.15
OVERLAP_RATIO = 0.40  # 40% Overlap
SLICE_SIZE = 640      
POSTPROCESS_TYPE = "NMM" # Non-Maximum Merging
MATCH_THRESH = 0.5

def main():
    print(f"--- FINAL SHOWDOWN INFERENCE: SAHI + NMM ---")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Loading Model: {MODEL_PATH}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRES,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Found {len(image_files)} images.")

    #PROGRESS BAR Loop
    for img_file in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        
        #SAHI CALL with NMM
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_pred=True, 
            postprocess_type=POSTPROCESS_TYPE,
            postprocess_match_threshold=MATCH_THRESH,
            verbose=0
        )
        
        # Convert to Output Format
        txt_name = os.path.splitext(img_file)[0] + '.txt'
        out_path = os.path.join(OUTPUT_DIR, txt_name)
        img_w = result.image_width
        img_h = result.image_height
        
        with open(out_path, 'w') as f:
            for prediction in result.object_prediction_list:
                cls_id = prediction.category.id 
                bbox = prediction.bbox
                
                # Normalize
                box_w = bbox.maxx - bbox.minx
                box_h = bbox.maxy - bbox.miny
                center_x = bbox.minx + (box_w / 2)
                center_y = bbox.miny + (box_h / 2)
                
                n_x = center_x / img_w
                n_y = center_y / img_h
                n_w = box_w / img_w
                n_h = box_h / img_h
                
                score = prediction.score.value
                f.write(f"{cls_id} {n_x:.6f} {n_y:.6f} {n_w:.6f} {n_h:.6f} {score:.6f}\n")

    # Zip
    print("Zipping")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(OUTPUT_DIR, file)
                zipf.write(file_path, arcname)
                
    print(f"{ZIP_NAME} done")

if __name__ == '__main__':
    main()