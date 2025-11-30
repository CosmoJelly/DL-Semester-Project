# Multi-Cancer Image Classification with Deep Learning

A comprehensive deep learning project for multi-cancer image classification, implementing and extending the methodology from the Scientific Reports paper. This project includes baseline CNN models, novel architectural improvements (attention mechanisms, contour feature fusion, Vision Transformers), and systematic experimental evaluation.

## Project Overview

This project implements a complete pipeline for medical image classification across 7 cancer types:
- **Leukemia** (ALL - Acute Lymphoblastic Leukemia)
- **Brain Tumor**
- **Breast Cancer**
- **Cervical Cancer**
- **Kidney Cancer** (CT scans)
- **Lung & Colon Cancer**
- **Oral Cancer**

### Key Features

1. **Paper Replication**: Full implementation of the original methodology with:
   - Gaussian blur + Otsu thresholding + Watershed segmentation
   - Contour feature extraction (perimeter, area, epsilon, num_vertices)
   - 70/15/15 train/validation/test split
   - Two-phase fine-tuning (frozen backbone → unfrozen last N blocks)
   - Class weighting for imbalanced datasets
   - Data augmentation (flips, rotations, zoom, brightness)

2. **Novel Architectural Improvements**:
   - **ECA Attention** (Efficient Channel Attention) on DenseNet121
   - **GAM Attention** (Global Attention Mechanism) combining channel and spatial attention
   - **Contour Feature Fusion**: Fusing CNN features with hand-crafted contour features
   - **Hybrid Models**: Combining attention mechanisms with contour fusion
   - **MaxViT**: Vision Transformer baseline for comparison

3. **Comprehensive Evaluation**:
   - Accuracy, Precision, Recall, F1-score, RMSE metrics
   - Per-class and per-dataset performance analysis
   - Segmentation ablation studies
   - Inference latency and memory benchmarking
   - Multi-cancer unified dataset support

## Project Structure

```
.
├── config.py                  # Centralized configuration
├── model_builder.py           # Model architecture definitions
├── dataset_utils.py           # Data preprocessing and dataset creation
├── train.py                   # Training script with two-phase fine-tuning
├── evaluate.py                # Model evaluation script
├── run_experiments.py         # Automated experiment sweeps
├── aggregate_results.py       # Results aggregation into tables
├── research_experiments.py    # Systematic testing of improvements
├── segmentation_ablation.py   # Segmentation ablation study
├── benchmark_inference.py     # Inference latency/memory benchmarking
├── build_multicancer_dataset.py  # Create unified multi-cancer dataset
├── ensemble_model.py          # Ensemble model definitions
├── dataset_reorg.py           # Dataset reorganization utility
├── setup_venv.py              # Virtual environment setup
├── install_requirements.bat   # Windows dependency installation
├── test_gpu.py               # GPU availability test
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Reorganized datasets
│   └── processed/            # Processed datasets
├── models/                   # Saved model checkpoints
├── results/                  # Evaluation metrics and plots
├── logs/                     # Training logs and TensorBoard
└── features/                 # Extracted contour features (CSV)
```

## Installation

### Prerequisites

- Python 3.8-3.11 (TensorFlow compatibility)
- NVIDIA GPU with CUDA support (recommended)
- Kaggle API credentials (for dataset download)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DL-Semester-Project-main
   ```

2. **Set up virtual environment** (Windows):
   ```bash
   python setup_venv.py
   # Or manually:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Windows (handles TensorFlow/CUDA compatibility):
   install_requirements.bat
   
   # Or manually:
   pip install tensorflow
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API**:
   ```bash
   # Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac)
   # Or C:\Users\<username>\.kaggle\kaggle.json (Windows)
   mkdir -p ~/.kaggle
   # Copy your kaggle.json there
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Test GPU availability**:
   ```bash
   python test_gpu.py
   ```

## Usage

### 1. Dataset Preparation

**Download and reorganize datasets**:
```bash
# Download datasets manually from Kaggle or use Kaggle API:
# kaggle datasets download -d <dataset-key> -p datasets/

# Reorganize into standard structure: data/raw/<dataset_key>/<class_name>/
python dataset_reorg.py
```

**Note**: Some datasets (brain_tumor, oral_cancer) may require manual download. See `config.py` for Kaggle dataset keys.

**Create unified multi-cancer dataset** (optional):
```bash
python build_multicancer_dataset.py
# Creates data/raw/multi_cancer/ with all cancer types
```

### 2. Training

**Train a single model**:
```bash
python train.py --dataset_dir data/raw/leukemia --model DenseNet121 --out_prefix leukemia_d121 --epochs 40
```

**Train with extended models** (Attention, ContourFusion, etc.):
```bash
python train.py --dataset_dir data/raw/leukemia --model DenseNet121_Attention --out_prefix leukemia_d121_attn
python train.py --dataset_dir data/raw/leukemia --model DenseNet121_ContourFusion --out_prefix leukemia_d121_cf
python train.py --dataset_dir data/raw/leukemia --model DenseNet121_Hybrid --out_prefix leukemia_d121_hybrid
python train.py --dataset_dir data/raw/leukemia --model MaxViT --out_prefix leukemia_maxvit
```

**Training options**:
- `--no_fine_tuning`: Disable two-phase fine-tuning
- `--no_class_weight`: Disable class weighting
- `--batch_size N`: Override batch size (default: 8)
- `--epochs N`: Override epochs (default: 40)

### 3. Evaluation

**Evaluate a trained model**:
```bash
python evaluate.py --model_path models/leukemia_d121_DenseNet121_best.h5 --dataset_dir data/raw/leukemia --out_prefix eval_leukemia
```

### 4. Experiment Automation

**Run full experiment sweep** (all datasets × all models):
```bash
# With extended models (Attention, MaxViT, ContourFusion, etc.)
python run_experiments.py --skip_existing

# Baseline models only
python run_experiments.py --skip_existing --no_extended

# Specific datasets/models
python run_experiments.py --datasets leukemia breast_cancer --models DenseNet121 DenseNet201
```

**Run research experiments** (systematic comparison of improvements):
```bash
python research_experiments.py --dataset leukemia
```

**Run segmentation ablation study**:
```bash
python segmentation_ablation.py --datasets leukemia breast_cancer
```

### 5. Results Analysis

**Aggregate results into tables**:
```bash
python aggregate_results.py
# Generates:
# - results/table_per_cancer_type.csv
# - results/table_per_model_avg.csv
# - results/table_detailed_per_class.csv
```

**Benchmark inference latency and memory**:
```bash
python benchmark_inference.py --model DenseNet121 --num_classes 2
```

## Configuration

All configuration is centralized in `config.py`:

### Key Parameters

- **Image Processing**:
  - `IMG_SIZE = (224, 224)`: Input image size
  - `USE_SEGMENTATION = True`: Enable Otsu + Watershed segmentation
  - `GAUSSIAN_BLUR_KERNEL = (5, 5)`: Gaussian blur kernel
  - `GAUSSIAN_BLUR_SIGMA = 1.0`: Gaussian blur sigma

- **Data Splitting**:
  - `TRAIN_SPLIT = 0.70`: Training set proportion
  - `VAL_SPLIT = 0.15`: Validation set proportion
  - `TEST_SPLIT = 0.15`: Test set proportion

- **Training**:
  - `BATCH_SIZE = 8`: Batch size (tuned for RTX 4050 6GB VRAM)
  - `EPOCHS = 40`: Total epochs
  - `LEARNING_RATE = 1e-4`: Initial learning rate
  - `FINE_TUNE_EPOCHS_PHASE1 = 10`: Frozen backbone epochs
  - `FINE_TUNE_EPOCHS_PHASE2 = 30`: Fine-tuning epochs
  - `FINE_TUNE_LR_PHASE2 = 1e-5`: Fine-tuning learning rate

- **Data Augmentation**:
  - `USE_AUGMENTATION = True`: Enable augmentation
  - `AUGMENTATION_ROTATION = 20`: Rotation range (degrees)
  - `AUGMENTATION_ZOOM = 0.2`: Zoom range
  - `AUGMENTATION_BRIGHTNESS = 0.2`: Brightness range

- **Contour Features**:
  - `SAVE_CONTOUR_FEATURES = True`: Save contour features to CSV
  - `FEATURES_DIR`: Directory for contour feature CSVs

### Supported Models

**Baseline Models** (from paper):
- DenseNet201, DenseNet121
- InceptionResNetV2, InceptionV3
- MobileNetV2
- NASNetLarge, NASNetMobile
- ResNet152V2
- VGG19
- Xception

**Extended Models** (novel improvements):
- `DenseNet121_Attention`: DenseNet121 with ECA attention
- `DenseNet121_GAM`: DenseNet121 with GAM attention
- `DenseNet121_ContourFusion`: DenseNet121 with contour feature fusion
- `DenseNet121_Hybrid`: DenseNet121 with ECA attention + contour fusion
- `MaxViT`: Vision Transformer baseline

## Output Files

### Models
- `models/{prefix}_{model}_best.h5`: Best model checkpoint

### Results
- `results/{prefix}_{model}_val_metrics.json`: Validation metrics
- `results/{prefix}_{model}_test_metrics.json`: Test metrics (held-out)
- `results/{prefix}_{model}_val_cm.npy`: Validation confusion matrix
- `results/{prefix}_{model}_test_cm.npy`: Test confusion matrix
- `results/{prefix}_{model}_val_cm.png`: Validation confusion matrix plot
- `results/{prefix}_{model}_test_cm.png`: Test confusion matrix plot
- `results/{prefix}_{model}_loss.png`: Training/validation loss plot
- `results/{prefix}_{model}_acc.png`: Training/validation accuracy plot

### Logs
- `logs/{prefix}_{model}.csv`: Training history CSV
- `logs/{prefix}_{model}_tb/`: TensorBoard logs

### Features
- `features/{dataset}_contours.csv`: Extracted contour features

### Aggregated Results
- `results/summary.csv`: Experiment summary
- `results/table_per_cancer_type.csv`: Per-cancer performance table
- `results/table_per_model_avg.csv`: Per-model average performance
- `results/table_detailed_per_class.csv`: Detailed per-class metrics

## Key Improvements Over Baseline

1. **Attention Mechanisms**:
   - ECA (Efficient Channel Attention): Lightweight channel attention
   - GAM (Global Attention Mechanism): Channel + spatial attention

2. **Contour Feature Fusion**:
   - Extracts geometric features (perimeter, area, epsilon, num_vertices)
   - Fuses with CNN features for improved classification

3. **Hybrid Architecture**:
   - Combines attention and contour fusion for maximum performance

4. **Vision Transformer**:
   - MaxViT baseline for comparison with CNN architectures

5. **Segmentation Ablation**:
   - Systematic study of segmentation preprocessing impact

6. **Multi-Cancer Unified Dataset**:
   - Support for training single model across all cancer types

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `IMG_SIZE` in `config.py` to `(160, 160)`
- Reduce `BATCH_SIZE` to 4 or 2
- Avoid `NASNetLarge` (very memory-intensive)

### Import Errors

- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- For TensorFlow/CUDA issues, use `install_requirements.bat`

### Dataset Download Issues

- Verify Kaggle API credentials
- Check dataset keys in `config.py` (removed footnote suffixes)
- Some datasets (brain_tumor, oral_cancer) require manual download

### Segmentation Not Working

- Ensure `USE_SEGMENTATION = True` in `config.py`
- Check that `dataset_utils.py` uses `config.USE_SEGMENTATION` dynamically
- For ablation studies, use `segmentation_ablation.py`

## Citation

If you use this code, please cite the original paper and acknowledge the improvements:

```bibtex
@article{original_paper,
  title={Multi-Cancer Image Classification with Deep Learning},
  journal={Scientific Reports},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- Original paper methodology
- TensorFlow/Keras for deep learning framework
- Kaggle for dataset hosting
- OpenCV, scikit-image for image processing
