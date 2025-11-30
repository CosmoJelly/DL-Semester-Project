# aggregate_results.py
"""
Aggregate results from all experiments into Tables 5-8 style summaries.
Generates per-cancer-type and per-model performance tables.
"""
import os, json, csv
import pandas as pd
import numpy as np
from config import RESULTS_DIR, SUPPORTED_MODELS

def load_all_metrics():
    """Load all metrics JSON files from results directory."""
    metrics_dict = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_metrics.json"):
            # Parse: {dataset}_{model}_{split}_metrics.json
            parts = filename.replace("_metrics.json", "").split("_")
            if len(parts) >= 3:
                # Find split (val or test)
                split = parts[-1]
                model = parts[-2]
                dataset = "_".join(parts[:-2])
                
                key = (dataset, model, split)
                filepath = os.path.join(RESULTS_DIR, filename)
                with open(filepath, 'r') as f:
                    metrics_dict[key] = json.load(f)
    return metrics_dict

def create_per_cancer_table(metrics_dict, split='test'):
    """
    Create Table 5-6 style: Per-cancer-type performance across all models.
    Columns: Cancer Type | Model | Accuracy | Precision | Recall | F1 | RMSE
    """
    rows = []
    datasets = set()
    for (dataset, model, s), metrics in metrics_dict.items():
        if s == split:
            datasets.add(dataset)
            rows.append({
                'cancer_type': dataset,
                'model': model,
                'accuracy': metrics.get('accuracy', 0.0),
                'precision_macro': metrics.get('macro avg', {}).get('precision', 0.0),
                'recall_macro': metrics.get('macro avg', {}).get('recall', 0.0),
                'f1_macro': metrics.get('macro avg', {}).get('f1-score', 0.0),
                'rmse': metrics.get('rmse', 0.0)
            })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No {split} metrics found")
        return df
    
    # Sort by cancer type, then model
    df = df.sort_values(['cancer_type', 'model'])
    return df

def create_per_model_table(metrics_dict, split='test'):
    """
    Create Table 7-8 style: Per-model average performance across all cancers.
    Columns: Model | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 | Avg RMSE
    """
    rows = []
    for (dataset, model, s), metrics in metrics_dict.items():
        if s == split:
            rows.append({
                'model': model,
                'accuracy': metrics.get('accuracy', 0.0),
                'precision_macro': metrics.get('macro avg', {}).get('precision', 0.0),
                'recall_macro': metrics.get('macro avg', {}).get('recall', 0.0),
                'f1_macro': metrics.get('macro avg', {}).get('f1-score', 0.0),
                'rmse': metrics.get('rmse', 0.0)
            })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No {split} metrics found")
        return df
    
    # Aggregate by model (average across all cancers)
    agg_df = df.groupby('model').agg({
        'accuracy': 'mean',
        'precision_macro': 'mean',
        'recall_macro': 'mean',
        'f1_macro': 'mean',
        'rmse': 'mean'
    }).reset_index()
    
    # Sort by accuracy descending
    agg_df = agg_df.sort_values('accuracy', ascending=False)
    return agg_df

def create_detailed_per_cancer_table(metrics_dict, split='test'):
    """
    Create detailed per-cancer table with per-class metrics (like Table 7-8 in paper).
    """
    all_rows = []
    for (dataset, model, s), metrics in metrics_dict.items():
        if s == split:
            # Get per-class metrics
            for class_name, class_metrics in metrics.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    all_rows.append({
                        'cancer_type': dataset,
                        'model': model,
                        'class': class_name,
                        'precision': class_metrics.get('precision', 0.0),
                        'recall': class_metrics.get('recall', 0.0),
                        'f1-score': class_metrics.get('f1-score', 0.0),
                        'support': class_metrics.get('support', 0)
                    })
    
    df = pd.DataFrame(all_rows)
    return df

def main():
    print("Loading all metrics...")
    metrics_dict = load_all_metrics()
    print(f"Loaded {len(metrics_dict)} metric files")
    
    # Create tables for test set (held-out)
    print("\n=== Creating per-cancer-type table (Table 5-6 style) ===")
    per_cancer_df = create_per_cancer_table(metrics_dict, split='test')
    if not per_cancer_df.empty:
        csv_path = os.path.join(RESULTS_DIR, "table_per_cancer_type.csv")
        per_cancer_df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
        print(per_cancer_df.to_string())
    
    print("\n=== Creating per-model average table (Table 7-8 style) ===")
    per_model_df = create_per_model_table(metrics_dict, split='test')
    if not per_model_df.empty:
        csv_path = os.path.join(RESULTS_DIR, "table_per_model_avg.csv")
        per_model_df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
        print(per_model_df.to_string())
    
    print("\n=== Creating detailed per-cancer per-class table ===")
    detailed_df = create_detailed_per_cancer_table(metrics_dict, split='test')
    if not detailed_df.empty:
        csv_path = os.path.join(RESULTS_DIR, "table_detailed_per_class.csv")
        detailed_df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
    
    # Also create validation set tables
    print("\n=== Creating validation set tables ===")
    per_cancer_val = create_per_cancer_table(metrics_dict, split='val')
    if not per_cancer_val.empty:
        csv_path = os.path.join(RESULTS_DIR, "table_per_cancer_type_val.csv")
        per_cancer_val.to_csv(csv_path, index=False)
        print(f"Saved validation table to {csv_path}")

if __name__ == "__main__":
    main()

