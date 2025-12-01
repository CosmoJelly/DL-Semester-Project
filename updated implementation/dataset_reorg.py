# dataset_reorg.py
import os
import shutil
from pathlib import Path
from config import DATA_ROOT, RAW_DIR
import zipfile

# Add dataset-specific reorganizers here if structure differs
# For many Kaggle zips, extracted content already has folders; else adapt below.

DATASET_KEY_FOLDER_MAP = {
    # dataset_key: possible folder names inside downloaded zip (prioritize)
    "leukemia": ["Leukemia", "leukemia", "all_images", "ALL"],
    "breast_cancer": ["breast-cancer-dataset", "breast", "Breast"],
    "cervical_cancer": ["cervical-cancer-largest-dataset", "CERVICAL"],
    "ct_kidney": ["CT-Kidney", "ct_kidney", "CT_KIDNEY"],
    "lung_colon": ["lung-and-colon-cancer", "lung_colon", "lung"],
    "brain_tumor": ["brain-tumor", "Brain_Tumor", "brain", "Brain", "BT"],
    "oral_cancer": ["oral-cancer", "Oral_Cancer", "oral", "Oral", "OC"]
}

def find_source_dir(dataset_root):
    # If already organized, return it
    # Otherwise search common folder names
    entries = list(Path(dataset_root).iterdir())
    # if subfolders already are class folders -> return dataset_root
    subdirs = [e for e in entries if e.is_dir()]
    if any((d / "images").exists() for d in subdirs) or len(subdirs) > 0:
        # heuristic: if any subdir contains many files, assume it's structure root
        return dataset_root
    return None

def reorganize(dataset_key):
    src_root = os.path.join(DATA_ROOT, dataset_key)
    dest_root = os.path.join(RAW_DIR, dataset_key)
    os.makedirs(dest_root, exist_ok=True)
    if not os.path.exists(src_root):
        print(f"[WARN] Source dataset {dataset_key} not found in {DATA_ROOT}.")
        return

    # naive approach: find image files recursively and categorize by parent folder name
    images = []
    for root, dirs, files in os.walk(src_root):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg",".tiff",".bmp")):
                images.append((os.path.join(root, f), root))

    if not images:
        print(f"[WARN] No images found for {dataset_key} under {src_root}")
        return

    # move/copy images into dest_root/<class_name> where class_name is parent folder name
    for img_path, parent in images:
        class_name = os.path.basename(parent)
        # sometimes parent is a generic folder like "images", so try grandparent
        if class_name.lower() in ("images","img","data","dataset"):
            class_name = os.path.basename(os.path.dirname(parent))
        target_dir = os.path.join(dest_root, class_name)
        os.makedirs(target_dir, exist_ok=True)
        # copy image
        base = os.path.basename(img_path)
        dest_path = os.path.join(target_dir, base)
        if not os.path.exists(dest_path):
            shutil.copy2(img_path, dest_path)
    print(f"[OK] Reorganized {dataset_key} -> {dest_root}")

def reorganize_all():
    from config import KAGGLE_KEYS
    for k in KAGGLE_KEYS.keys():
        reorganize(k)

if __name__ == "__main__":
    reorganize_all()
    print("Done reorganizing. Check data/raw/<dataset_key>/")