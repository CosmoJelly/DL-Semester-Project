# train.py
import os, json, argparse, time
import tensorflow as tf
from dataset_utils import make_tf_dataset_from_folder
from model_builder import build_model
from config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, MIXED_PRECISION
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def plot_history(history, out_prefix):
    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_loss.png"))
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_acc.png"))

def evaluate_and_save(model, val_ds, class_names, out_prefix):
    y_true=[]; y_pred=[]
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(RESULTS_DIR, f"{out_prefix}_metrics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, f"{out_prefix}_cm.npy"), cm)
    # confusion matrix plot
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.title(f"{out_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_cm.png"))
    return rep, cm

def train_one(dataset_dir, model_name, out_prefix, epochs=EPOCHS, batch_size=BATCH_SIZE):
    train_ds, val_ds, class_names = make_tf_dataset_from_folder(dataset_dir, batch_size=batch_size)
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Building model:", model_name)
    model = build_model(model_name, num_classes, base_trainable=False)
    model.summary()
    checkpoint = os.path.join(MODELS_DIR, f"{out_prefix}_{model_name}_best.h5")
    callbacks = [
        ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        CSVLogger(os.path.join(LOGS_DIR, f"{out_prefix}_{model_name}.csv")),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, f"{out_prefix}_{model_name}_tb"))
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    plot_history(history, f"{out_prefix}_{model_name}")
    rep, cm = evaluate_and_save(model, val_ds, class_names, f"{out_prefix}_{model_name}")
    return model, history, rep, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_prefix", default="run")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train_one(args.dataset_dir, args.model, args.out_prefix, epochs=args.epochs, batch_size=args.batch_size)