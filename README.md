# DL_Project Run Guide

Prereqs:
- Place your `kaggle.json` at `~/.kaggle/kaggle.json`
    ```
  mkdir -p ~/.kaggle
    ```
    copy kaggle.json there, then:
    ```
  chmod 600 ~/.kaggle/kaggle.json
    ```
- Install packages:
  ```
  pip install -r requirements.txt
  ```

Steps:

1. Download datasets (already provided earlier):
   ```
   python3 download_datasets.py
   ```

3. Reorganize raw dataset folders into data/raw/<dataset_key>/<class_name>:
   ```
   python3 dataset_reorg.py
   ```

5. Train a single model:
   ```
   python3 train.py --dataset_dir data/raw/leukemia --model DenseNet121 --out_prefix leukemia_d121 --epochs 30
   ```

7. Run full sweep (all datasets in data/raw and all models):
   ```
   python run_experiments.py --skip_existing
   ```

Notes:
- If you get OOM, reduce IMG_SIZE in config.py to (160,160) and/or reduce BATCH_SIZE to 4 or 2.
- NASNetLarge is prone to OOM
- To disable segmentation preprocessing, set USE_SEGMENTATION=False in config.py.

Outputs:
- models saved under `models/`
- plots & metrics under `results/`
- logs & TensorBoard under `logs/`
