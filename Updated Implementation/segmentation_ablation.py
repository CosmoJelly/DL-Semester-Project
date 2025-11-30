# segmentation_ablation.py
"""
Systematic ablation study comparing models with vs without segmentation preprocessing.
This addresses the paper's limitation about segmentation dependency.
"""
import os, json, csv, argparse
from train import train_one
from config import RAW_DIR, RESULTS_DIR, SUPPORTED_MODELS, MODELS_DIR, USE_SEGMENTATION
from dataset_reorg import reorganize_all

def run_segmentation_ablation(datasets=None, models=None, skip_existing=True):
    """
    Run experiments with and without segmentation for all model/dataset combinations.
    Saves results to ablation_segmentation.csv
    """
    from config import RAW_DIR
    import os
    
    if datasets is None:
        datasets = sorted([d.name for d in os.scandir(RAW_DIR) if d.is_dir()])
    if models is None:
        models = SUPPORTED_MODELS
    
    results = []
    
    for dataset in datasets:
        ds_folder = os.path.join(RAW_DIR, dataset)
        if not os.path.exists(ds_folder):
            print(f"[WARN] Dataset folder not found: {ds_folder}, skipping.")
            continue
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"Running ablation: {model} on {dataset}")
            print(f"{'='*60}")
            
            # Run WITH segmentation
            print(f"\n--- WITH segmentation ---")
            prefix_with = f"{dataset}_{model}_with_seg"
            model_file_with = os.path.join(MODELS_DIR, f"{prefix_with}_{model}_best.h5")
            
            if skip_existing and os.path.exists(model_file_with):
                print(f"[SKIP] Model already exists: {model_file_with}")
                # Try to load metrics
                metrics_file = os.path.join(RESULTS_DIR, f"{prefix_with}_test_metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        rep_test_with = json.load(f)
                else:
                    continue
            else:
                try:
                    # Temporarily enable segmentation
                    from config import USE_SEGMENTATION as OLD_SEG
                    import config
                    config.USE_SEGMENTATION = True
                    
                    _, _, _, rep_test_with, _, _ = train_one(ds_folder, model, prefix_with)
                    
                    config.USE_SEGMENTATION = OLD_SEG
                except Exception as e:
                    print(f"[ERROR] With segmentation failed: {e}")
                    continue
            
            # Run WITHOUT segmentation
            print(f"\n--- WITHOUT segmentation ---")
            prefix_without = f"{dataset}_{model}_no_seg"
            model_file_without = os.path.join(MODELS_DIR, f"{prefix_without}_{model}_best.h5")
            
            if skip_existing and os.path.exists(model_file_without):
                print(f"[SKIP] Model already exists: {model_file_without}")
                metrics_file = os.path.join(RESULTS_DIR, f"{prefix_without}_test_metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        rep_test_without = json.load(f)
                else:
                    continue
            else:
                try:
                    # Temporarily disable segmentation
                    from config import USE_SEGMENTATION as OLD_SEG
                    import config
                    config.USE_SEGMENTATION = False
                    
                    _, _, _, rep_test_without, _, _ = train_one(ds_folder, model, prefix_without)
                    
                    config.USE_SEGMENTATION = OLD_SEG
                except Exception as e:
                    print(f"[ERROR] Without segmentation failed: {e}")
                    continue
            
            # Compare results
            acc_with = rep_test_with.get('accuracy', 0.0)
            acc_without = rep_test_without.get('accuracy', 0.0)
            f1_with = rep_test_with.get('macro avg', {}).get('f1-score', 0.0)
            f1_without = rep_test_without.get('macro avg', {}).get('f1-score', 0.0)
            rmse_with = rep_test_with.get('rmse', 0.0)
            rmse_without = rep_test_without.get('rmse', 0.0)
            
            acc_diff = acc_with - acc_without
            f1_diff = f1_with - f1_without
            rmse_diff = rmse_without - rmse_with  # Lower RMSE is better
            
            results.append({
                'dataset': dataset,
                'model': model,
                'acc_with_seg': acc_with,
                'acc_without_seg': acc_without,
                'acc_improvement': acc_diff,
                'f1_with_seg': f1_with,
                'f1_without_seg': f1_without,
                'f1_improvement': f1_diff,
                'rmse_with_seg': rmse_with,
                'rmse_without_seg': rmse_without,
                'rmse_improvement': rmse_diff,
                'segmentation_helps': acc_diff > 0.01  # >1% improvement
            })
            
            print(f"\nResults:")
            print(f"  Accuracy: {acc_with:.4f} (with) vs {acc_without:.4f} (without) | diff: {acc_diff:+.4f}")
            print(f"  F1-Score: {f1_with:.4f} (with) vs {f1_without:.4f} (without) | diff: {f1_diff:+.4f}")
            print(f"  RMSE:     {rmse_with:.4f} (with) vs {rmse_without:.4f} (without) | diff: {rmse_diff:+.4f}")
            print(f"  Segmentation helps: {acc_diff > 0.01}")
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, "ablation_segmentation.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'model', 'acc_with_seg', 'acc_without_seg', 'acc_improvement',
                      'f1_with_seg', 'f1_without_seg', 'f1_improvement',
                      'rmse_with_seg', 'rmse_without_seg', 'rmse_improvement', 'segmentation_helps']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Saved ablation results to {csv_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    if results:
        helps_count = sum(1 for r in results if r['segmentation_helps'])
        total = len(results)
        print(f"\nSummary: Segmentation helps in {helps_count}/{total} cases ({100*helps_count/total:.1f}%)")
        
        avg_acc_improvement = sum(r['acc_improvement'] for r in results) / len(results)
        print(f"Average accuracy improvement: {avg_acc_improvement:+.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation ablation study")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset keys to test")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to test")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing models")
    args = parser.parse_args()
    
    reorganize_all()
    run_segmentation_ablation(args.datasets, args.models, skip_existing=args.skip_existing)

