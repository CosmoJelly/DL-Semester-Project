# config.py
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "datasets")       # where download_datasets.py saved files
RAW_DIR = os.path.join(ROOT, "data", "raw")     # raw reorganized datasets
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
LOGS_DIR = os.path.join(ROOT, "logs")
MODELS_DIR = os.path.join(ROOT, "models")

for p in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(p, exist_ok=True)

# Image / training defaults (tuned for RTX 4050 Laptop GPU - 6GB VRAM)
IMG_SIZE = (224, 224)       # change to (160,160) if memory tight
BATCH_SIZE = 8              # Reduced for 6GB VRAM (can increase to 12-16 if needed)
EPOCHS = 40
LEARNING_RATE = 1e-4

USE_SEGMENTATION = True     # Otsu + watershed cropping pipeline
MIXED_PRECISION = False     # toggle if you want (careful on GTX 1650)

# Segmentation parameters
GAUSSIAN_BLUR_KERNEL = (5, 5)  # Gaussian blur kernel size
GAUSSIAN_BLUR_SIGMA = 1.0      # Gaussian blur sigma

# Data split (matching paper: 70/15/15)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Fine-tuning parameters
FINE_TUNE_EPOCHS_PHASE1 = 10   # Phase 1: frozen backbone
FINE_TUNE_EPOCHS_PHASE2 = 30   # Phase 2: unfrozen last N blocks
FINE_TUNE_LR_PHASE2 = 1e-5     # Lower LR for phase 2
FINE_TUNE_UNFREEZE_LAST_N = 3  # Unfreeze last N blocks (model-dependent)

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_ROTATION = 20      # degrees
AUGMENTATION_ZOOM = 0.2         # zoom range
AUGMENTATION_BRIGHTNESS = 0.2   # brightness range

# Contour features
SAVE_CONTOUR_FEATURES = True
FEATURES_DIR = os.path.join(ROOT, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

RANDOM_SEED = 42

SUPPORTED_MODELS = [
    "DenseNet201","DenseNet121","InceptionResNetV2","InceptionV3",
    "MobileNetV2","NasNetLarge","NasNetMobile","ResNet152V2","VGG19","Xception"
]

# Extended models with improvements
SUPPORTED_MODELS_EXTENDED = SUPPORTED_MODELS + [
    "DenseNet121_Attention",  # DenseNet121 with ECA attention
    "DenseNet121_GAM",  # DenseNet121 with GAM (Global Attention Mechanism)
    "MaxViT",  # Vision Transformer baseline
    "DenseNet121_ContourFusion",  # DenseNet121 with contour feature fusion
    "DenseNet121_Hybrid"  # DenseNet121 with ECA Attention + Contour Fusion (best of both)
]

# Models to use in experiments (includes extended models)
EXPERIMENT_MODELS = SUPPORTED_MODELS_EXTENDED

# mapping of Kaggle keys used by download script (optional)
# Note: Removed footnote suffixes (ii, iv, v, vi, vii) - these are paper references, not Kaggle slugs
KAGGLE_KEYS = {
    "leukemia": "mehradaria/leukemia",
    "breast_cancer": "anaselmasry/breast-cancer-dataset",
    "cervical_cancer": "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
    "ct_kidney": "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone",
    "lung_colon": "biplobdey/lung-and-colon-cancer",
    # Note: brain_tumor and oral_cancer may need manual download from Figshare/other sources
    "brain_tumor": None,  # Add Kaggle key or Figshare link if available
    "oral_cancer": None   # Add Kaggle key if available
}