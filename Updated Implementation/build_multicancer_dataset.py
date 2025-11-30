# build_multicancer_dataset.py
"""
Build a unified multi-cancer dataset by combining all individual cancer datasets.
Creates data/raw/multi_cancer/ with subfolders for each cancer type.
"""
import os
import shutil
from pathlib import Path
from config import RAW_DIR

# Mapping of individual dataset names to multi-cancer class names
CANCER_TYPE_MAPPING = {
    "leukemia": "ALL",  # Acute Lymphoblastic Leukemia
    "brain_tumor": "Brain",
    "lung_colon": "LungColon",
    "breast_cancer": "Breast",
    "cervical_cancer": "Cervical",
    "ct_kidney": "Kidney",
    "oral_cancer": "Oral"
}

def build_multicancer_dataset():
    """
    Build unified multi-cancer dataset by copying/symlinking images from individual datasets.
    Creates data/raw/multi_cancer/<cancer_type>/ structure.
    """
    multicancer_dir = os.path.join(RAW_DIR, "multi_cancer")
    os.makedirs(multicancer_dir, exist_ok=True)
    
    print("="*60)
    print("Building Multi-Cancer Dataset")
    print("="*60)
    
    total_images = 0
    for dataset_key, cancer_type in CANCER_TYPE_MAPPING.items():
        source_dir = os.path.join(RAW_DIR, dataset_key)
        target_dir = os.path.join(multicancer_dir, cancer_type)
        
        if not os.path.exists(source_dir):
            print(f"[SKIP] Source dataset not found: {source_dir}")
            continue
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Collect all images from source dataset (may have multiple class subfolders)
        image_count = 0
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    src_path = os.path.join(root, file)
                    # Create unique filename: {cancer_type}_{original_filename}
                    base_name = os.path.basename(file)
                    name, ext = os.path.splitext(base_name)
                    unique_name = f"{cancer_type}_{name}_{image_count}{ext}"
                    dst_path = os.path.join(target_dir, unique_name)
                    
                    # Copy image (use copy instead of symlink for portability)
                    shutil.copy2(src_path, dst_path)
                    image_count += 1
        
        total_images += image_count
        print(f"[OK] {cancer_type}: {image_count} images from {dataset_key}")
    
    print("="*60)
    print(f"Multi-cancer dataset created: {multicancer_dir}")
    print(f"Total images: {total_images}")
    print(f"Cancer types: {len([d for d in os.listdir(multicancer_dir) if os.path.isdir(os.path.join(multicancer_dir, d))])}")
    print("="*60)
    print("\nYou can now train a unified model:")
    print("  python train.py --dataset_dir data/raw/multi_cancer --model DenseNet121 --out_prefix multi_dense121")

if __name__ == "__main__":
    build_multicancer_dataset()

