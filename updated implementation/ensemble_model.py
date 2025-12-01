# ensemble_model.py
"""
Ensemble model combining predictions from multiple models.
Useful for research to achieve better performance than individual models.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_ensemble_model(model_paths, num_classes, weights=None):
    """
    Build ensemble model that averages predictions from multiple trained models.
    
    Args:
        model_paths: List of paths to saved model files
        num_classes: Number of output classes
        weights: Optional list of weights for each model (default: equal weights)
    
    Returns:
        Ensemble model that takes same input as individual models
    """
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    # Load all models
    loaded_models = []
    for path in model_paths:
        model = tf.keras.models.load_model(path, compile=False)
        loaded_models.append(model)
    
    # Get input shape from first model
    input_shape = loaded_models[0].input_shape[1:]
    
    # Create ensemble
    inputs = layers.Input(shape=input_shape)
    
    # Get predictions from each model
    predictions = []
    for i, model in enumerate(loaded_models):
        # Make model non-trainable
        model.trainable = False
        pred = model(inputs)
        predictions.append(pred * weights[i])
    
    # Average weighted predictions
    ensemble_pred = layers.Average()(predictions)
    outputs = layers.Activation('softmax', dtype='float32')(ensemble_pred)
    
    ensemble = models.Model(inputs, outputs, name='ensemble_model')
    ensemble.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    
    return ensemble

def evaluate_ensemble(model_paths, test_dataset, class_names, weights=None):
    """
    Evaluate ensemble model on test dataset.
    
    Args:
        model_paths: List of paths to saved model files
        test_dataset: TensorFlow dataset
        class_names: List of class names
        weights: Optional weights for each model
    """
    ensemble = build_ensemble_model(model_paths, len(class_names), weights)
    
    y_true = []
    y_pred = []
    y_pred_prob = []
    
    for batch in test_dataset:
        x_batch, y_batch = batch
        preds = ensemble.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_pred_prob.extend(preds.tolist())
    
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute RMSE
    from train import compute_rmse
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    rmse = compute_rmse(y_true_onehot, np.array(y_pred_prob))
    rep['rmse'] = float(rmse)
    
    return rep, cm

