# evaluate.py
import os, argparse, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset_utils import make_eval_dataset_from_folder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model_path, dataset_dir, out_prefix):
    model = load_model(model_path)
    
    # Check if model expects two inputs (contour fusion)
    use_contour_fusion = len(model.inputs) == 2
    
    # Create evaluation dataset (all images, no split)
    eval_ds, class_names = make_eval_dataset_from_folder(dataset_dir, batch_size=8, return_contours=use_contour_fusion)
    
    y_true=[]; y_pred=[]
    for batch in eval_ds:
        if use_contour_fusion:
            # For contour fusion: batch is ((images, contours), labels)
            (img_batch, contour_batch), y_batch = batch
            preds = model.predict([img_batch, contour_batch], verbose=0)
        else:
            # For normal models: batch is (images, labels)
            x_batch, y_batch = batch
            preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    with open(f"{out_prefix}_eval_metrics.json","w") as f:
        json.dump(rep, f, indent=2)
    np.save(f"{out_prefix}_cm.npy", cm)
    plt.figure(figsize=(6,6))
    plt.imshow(cm); plt.colorbar(); plt.title("Confusion Matrix")
    plt.savefig(f"{out_prefix}_eval_cm.png")
    print("Saved eval results with prefix", out_prefix)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out_prefix", default="eval")
    args = p.parse_args()
    evaluate_model(args.model_path, args.dataset_dir, args.out_prefix)