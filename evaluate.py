# evaluate.py
import os, argparse, json
import numpy as np
from tensorflow.keras.models import load_model
from dataset_utils import make_tf_dataset_from_folder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model_path, dataset_dir, out_prefix):
    model = load_model(model_path)
    train_ds, val_ds, class_names = make_tf_dataset_from_folder(dataset_dir, batch_size=8, val_split=0.0, shuffle=False)
    # if val_split=0.0 then make_tf_dataset will put all in val (not ideal) - alternate approach
    y_true=[]; y_pred=[]
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch)
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