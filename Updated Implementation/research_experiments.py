# research_experiments.py
"""
Research experiment script to systematically test architectural improvements
and compare against paper baseline (DenseNet121: 99.94% accuracy).

This script runs experiments to find improvements that beat the baseline.
"""
import os, json, csv, argparse, time
from train import train_one
from config import RAW_DIR, RESULTS_DIR, MODELS_DIR, EXPERIMENT_MODELS
from dataset_reorg import reorganize_all

# Research models to test (improvements over baseline)
RESEARCH_MODELS = [
    "DenseNet121",  # Baseline (paper's best)
    "DenseNet121_Attention",  # Improvement 1: ECA Attention
    "DenseNet121_GAM",  # Improvement 2: GAM Attention (Pacal 2024 style)
    "DenseNet121_ContourFusion",  # Improvement 3: Contour Feature Fusion
    "DenseNet121_Hybrid",  # Improvement 4: Attention + Contour Fusion
    "MaxViT",  # Improvement 5: Vision Transformer
]

def run_research_experiment(dataset, models=None, epochs=40):
    """
    Run research experiments on a single dataset to compare improvements.
    
    Args:
        dataset: Dataset name (e.g., "leukemia")
        models: List of models to test (default: RESEARCH_MODELS)
        epochs: Number of epochs
    """
    models = models or RESEARCH_MODELS
    ds_folder = os.path.join(RAW_DIR, dataset)
    
    if not os.path.exists(ds_folder):
        print(f"[ERROR] Dataset not found: {ds_folder}")
        return
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"Research Experiment: {dataset}")
    print(f"{'='*70}")
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        
        prefix = f"research_{dataset}_{model_name}"
        model_file = os.path.join(MODELS_DIR, f"{prefix}_{model_name}_best.h5")
        
        try:
            t0 = time.time()
            _, history, rep_val, rep_test, cm_val, cm_test = train_one(
                ds_folder, model_name, prefix, epochs=epochs
            )
            t1 = time.time()
            time_min = (t1 - t0) / 60.0
            
            # Extract metrics
            test_acc = rep_test.get("accuracy", 0.0)
            test_rmse = rep_test.get("rmse", 0.0)
            test_f1 = rep_test.get("macro avg", {}).get("f1-score", 0.0)
            val_acc = history.history.get("val_accuracy", [-1])[-1]
            
            results.append({
                "dataset": dataset,
                "model": model_name,
                "test_accuracy": float(test_acc),
                "test_rmse": float(test_rmse),
                "test_f1": float(test_f1),
                "val_accuracy": float(val_acc),
                "time_min": time_min,
                "improvement_over_baseline": float(test_acc) - 0.9994  # Paper baseline: 99.94%
            })
            
            print(f"\nResults for {model_name}:")
            print(f"  Test Accuracy: {test_acc:.4f} ({'+' if test_acc > 0.9994 else ''}{test_acc - 0.9994:.4f} vs baseline)")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Training Time: {time_min:.2f} minutes")
            
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, f"research_{dataset}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["dataset", "model", "test_accuracy", "test_rmse", "test_f1", 
                      "val_accuracy", "time_min", "improvement_over_baseline"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*70}")
    print(f"Saved research results to {csv_path}")
    print(f"{'='*70}")
    
    # Print summary
    if results:
        baseline_result = next((r for r in results if r['model'] == 'DenseNet121'), None)
        if baseline_result:
            baseline_acc = baseline_result['test_accuracy']
            print(f"\nBaseline (DenseNet121): {baseline_acc:.4f}")
            print("\nImprovements:")
            for r in results:
                if r['model'] != 'DenseNet121':
                    diff = r['test_accuracy'] - baseline_acc
                    print(f"  {r['model']}: {r['test_accuracy']:.4f} ({'+' if diff > 0 else ''}{diff:.4f})")
        
        # Find best model
        best = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBest Model: {best['model']} with {best['test_accuracy']:.4f} accuracy")

def main():
    parser = argparse.ArgumentParser(description="Run research experiments to test improvements")
    parser.add_argument("--dataset", required=True, help="Dataset name to test")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to test")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    args = parser.parse_args()
    
    reorganize_all()
    run_research_experiment(args.dataset, args.models, args.epochs)

if __name__ == "__main__":
    main()

