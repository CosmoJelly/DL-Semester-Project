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

# Image / training defaults (tuned for GTX 1650)
IMG_SIZE = (224, 224)       # change to (160,160) if memory tight
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4

USE_SEGMENTATION = True     # Otsu + watershed cropping pipeline
MIXED_PRECISION = False     # toggle if you want (careful on GTX 1650)

RANDOM_SEED = 42

SUPPORTED_MODELS = [
    "DenseNet201","DenseNet121","InceptionResNetV2","InceptionV3",
    "MobileNetV2","NasNetLarge","NasNetMobile","ResNet152V2","VGG19","Xception"
]

# mapping of Kaggle keys used by download script (optional)
KAGGLE_KEYS = {
    "leukemia": "mehradaria/leukemiaii",
    "breast_cancer": "anaselmasry/breast-cancer-datasetiv",
    "cervical_cancer": "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmedv",
    "ct_kidney": "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stonevi",
    "lung_colon": "biplobdey/lung-and-colon-cancervii"
}