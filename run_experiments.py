# run_experiments.py
import os, csv, argparse, time
from train import train_one
from config import RAW_DIR, RESULTS_DIR, SUPPORTED_MODELS, MODELS_DIR
from dataset_reorg import reorganize_all

def dataset_list():
    # assume reorganize created RAW_DIR/<dataset_key> folders
    keys = sorted([d.name for d in os.scandir(RAW_DIR) if d.is_dir()])
    return keys

def run_sweep(datasets=None, models=None, skip_existing=True):
    datasets = datasets or dataset_list()
    models = models or SUPPORTED_MODELS
    summary_rows = []
    for ds in datasets:
        ds_folder = os.path.join(RAW_DIR, ds)
        if not os.path.exists(ds_folder):
            print(f"[WARN] Dataset folder not found: {ds_folder}, skipping.")
            continue
        for m in models:
            print(f"\n=== Running {m} on {ds} ===")
            prefix = f"{ds}_{m}"
            model_file = os.path.join(MODELS_DIR, f"{prefix}_{m}_best.h5")
            if skip_existing and os.path.exists(model_file):
                print(f"[SKIP] Model already exists: {model_file}")
                # You could load metrics.json instead, but for simplicity skip training
                continue
            try:
                t0 = time.time()
                _, history, rep, cm = train_one(ds_folder, m, prefix)
                t1 = time.time()
                time_min = (t1 - t0)/60.0
                # extract metrics
                val_acc = history.history.get("val_accuracy", [-1])[-1]
                val_loss = history.history.get("val_loss", [-1])[-1]
                row = {"dataset": ds, "model": m, "val_acc": float(val_acc), "val_loss": float(val_loss), "time_min": time_min}
                summary_rows.append(row)
            except Exception as e:
                print(f"[ERROR] {m} on {ds} failed: {e}")
                continue
    # write summary CSV
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","model","val_acc","val_loss","time_min"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Wrote summary to", csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="dataset keys (folder names)")
    parser.add_argument("--models", nargs="*", default=None, help="model names")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    reorganize_all()
    run_sweep(args.datasets, args.models, skip_existing=args.skip_existing)