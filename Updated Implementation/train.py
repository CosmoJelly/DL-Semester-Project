# train.py
import os, json, argparse, time
import tensorflow as tf
from dataset_utils import make_tf_dataset_from_folder
from model_builder import build_model, build_model_with_contour_fusion
from config import (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, MIXED_PRECISION,
                   FINE_TUNE_EPOCHS_PHASE1, FINE_TUNE_EPOCHS_PHASE2, FINE_TUNE_LR_PHASE2, FINE_TUNE_UNFREEZE_LAST_N)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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

def compute_rmse(y_true_onehot, y_pred_prob):
    """Compute RMSE between true one-hot labels and predicted probabilities."""
    mse = np.mean((y_true_onehot - y_pred_prob) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def evaluate_and_save(model, val_ds, class_names, out_prefix, dataset_name=None, use_contour_fusion=False):
    y_true=[]; y_pred=[]; y_pred_prob=[]
    for batch in val_ds:
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
        y_pred_prob.extend(preds.tolist())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)
    
    # Convert true labels to one-hot for RMSE
    num_classes = len(class_names)
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    
    # Compute RMSE (matching paper)
    rmse = compute_rmse(y_true_onehot, y_pred_prob)
    
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Add RMSE to report
    rep['rmse'] = float(rmse)
    rep['rmse_per_class'] = {}
    for i, cls_name in enumerate(class_names):
        class_mask = y_true == i
        if np.any(class_mask):
            class_rmse = compute_rmse(y_true_onehot[class_mask], y_pred_prob[class_mask])
            rep['rmse_per_class'][cls_name] = float(class_rmse)
    
    with open(os.path.join(RESULTS_DIR, f"{out_prefix}_metrics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, f"{out_prefix}_cm.npy"), cm)
    
    # confusion matrix plot
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.title(f"{out_prefix} Confusion Matrix\nRMSE: {rmse:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_cm.png"))
    plt.close()
    
    print(f"RMSE: {rmse:.4f}")
    return rep, cm

def compute_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    return dict(zip(unique_labels, class_weights))

def unfreeze_last_n_blocks(model, n_blocks):
    """Unfreeze last N blocks of the base model for fine-tuning."""
    # This is model-dependent; for most models, we unfreeze the last N layers
    base_model = model.layers[0]  # First layer is the base model
    total_layers = len(base_model.layers)
    # Unfreeze last N layers (approximate, as blocks vary by architecture)
    layers_to_unfreeze = min(n_blocks * 10, total_layers)  # Rough estimate
    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    print(f"Unfroze last {layers_to_unfreeze} layers of base model")
    return model

def train_one(dataset_dir, model_name, out_prefix, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              use_fine_tuning=True, use_class_weight=True):
    """
    Train model with optional fine-tuning and class weighting.
    
    Args:
        dataset_dir: Path to dataset folder
        model_name: Model architecture name
        out_prefix: Output prefix for saved files
        epochs: Total epochs (if fine-tuning, split between phases)
        batch_size: Batch size
        use_fine_tuning: If True, use two-phase training
        use_class_weight: If True, compute and use class weights
    """
    # Extract dataset name from path for contour features
    dataset_name = os.path.basename(dataset_dir.rstrip('/\\'))
    
    # Check if this is a contour fusion or hybrid model
    use_contour_fusion = model_name.endswith("_ContourFusion") or model_name.endswith("_Hybrid")
    if model_name.endswith("_Hybrid"):
        base_model_name = model_name.replace("_Hybrid", "")
    elif model_name.endswith("_ContourFusion"):
        base_model_name = model_name.replace("_ContourFusion", "")
    else:
        base_model_name = model_name
    
    # Get three-way split (70/15/15)
    # For contour fusion, return contour features as second input
    train_ds, val_ds, test_ds, class_names = make_tf_dataset_from_folder(
        dataset_dir, batch_size=batch_size, return_test=True, 
        dataset_name=dataset_name, return_contours=use_contour_fusion
    )
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Building model:", model_name)
    if use_contour_fusion:
        print("Using contour feature fusion")
    
    # Phase 1: Frozen backbone
    if use_contour_fusion:
        if model_name.endswith("_Hybrid"):
            # Hybrid: Attention + Contour Fusion
            from model_builder import build_hybrid_attention_contour_model
            model = build_hybrid_attention_contour_model(base_model_name, num_classes, base_trainable=False)
        else:
            model = build_model_with_contour_fusion(base_model_name, num_classes, base_trainable=False)
    else:
        model = build_model(model_name, num_classes, base_trainable=False)
    model.summary()
    
    # Compute class weights
    class_weights = None
    if use_class_weight:
        # Collect all training labels to compute weights
        all_labels = []
        for batch in train_ds:
            if use_contour_fusion:
                # For contour fusion: batch is ((images, contours), labels)
                _, labels = batch
            else:
                # For normal models: batch is (images, labels)
                _, labels = batch
            all_labels.extend(labels.numpy().tolist())
        class_weights = compute_class_weights(all_labels)
        print("Class weights:", class_weights)
    
    checkpoint = os.path.join(MODELS_DIR, f"{out_prefix}_{model_name}_best.h5")
    callbacks = [
        ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        CSVLogger(os.path.join(LOGS_DIR, f"{out_prefix}_{model_name}.csv")),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, f"{out_prefix}_{model_name}_tb"))
    ]
    
    # Phase 1: Train with frozen backbone
    phase1_epochs = FINE_TUNE_EPOCHS_PHASE1 if use_fine_tuning else epochs
    print(f"\n=== Phase 1: Training with frozen backbone ({phase1_epochs} epochs) ===")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs, 
                        callbacks=callbacks, class_weight=class_weights, verbose=1)
    
    # Phase 2: Fine-tuning (unfreeze last N blocks)
    if use_fine_tuning and epochs > phase1_epochs:
        print(f"\n=== Phase 2: Fine-tuning (unfreezing last {FINE_TUNE_UNFREEZE_LAST_N} blocks) ===")
        model = unfreeze_last_n_blocks(model, FINE_TUNE_UNFREEZE_LAST_N)
        # Recompile with lower learning rate
        model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR_PHASE2),
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy",
                              tf.keras.metrics.Precision(name="precision"),
                              tf.keras.metrics.Recall(name="recall")])
        
        phase2_epochs = FINE_TUNE_EPOCHS_PHASE2
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs,
                           callbacks=callbacks, class_weight=class_weights, verbose=1)
        
        # Combine histories
        history = type('History', (), {})()
        history.history = {}
        for key in history1.history.keys():
            history.history[key] = history1.history[key] + history2.history[key]
    else:
        history = history1
    
    plot_history(history, f"{out_prefix}_{model_name}")
    
    # Evaluate on validation set
    rep_val, cm_val = evaluate_and_save(model, val_ds, class_names, f"{out_prefix}_{model_name}_val", dataset_name, use_contour_fusion)
    
    # Evaluate on test set (held-out)
    rep_test, cm_test = evaluate_and_save(model, test_ds, class_names, f"{out_prefix}_{model_name}_test", dataset_name, use_contour_fusion)
    
    return model, history, rep_val, rep_test, cm_val, cm_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_prefix", default="run")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no_fine_tuning", action="store_true", help="Disable fine-tuning")
    parser.add_argument("--no_class_weight", action="store_true", help="Disable class weighting")
    args = parser.parse_args()
    train_one(args.dataset_dir, args.model, args.out_prefix, 
              epochs=args.epochs, batch_size=args.batch_size,
              use_fine_tuning=not args.no_fine_tuning,
              use_class_weight=not args.no_class_weight)