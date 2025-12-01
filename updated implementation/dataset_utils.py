# dataset_utils.py
import os, random, csv
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import tensorflow as tf
from config import (IMG_SIZE, BATCH_SIZE, RANDOM_SEED, RAW_DIR,
                   GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
                   SAVE_CONTOUR_FEATURES, FEATURES_DIR, USE_AUGMENTATION,
                   AUGMENTATION_ROTATION, AUGMENTATION_ZOOM, AUGMENTATION_BRIGHTNESS)
import config  # Import module to access USE_SEGMENTATION dynamically
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Global dict to store contour features during dataset creation
_contour_features_cache = {}

def segment_and_crop_cv(img_bgr, min_size=500, image_path=None):
    """
    Segment and crop image with Gaussian smoothing and contour feature extraction.
    Returns (cropped_image, contour_features_dict) or (None, None) if segmentation fails.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Add explicit Gaussian smoothing (matching paper)
    img_gray_blur = cv2.GaussianBlur(img_gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)
    
    try:
        th = threshold_otsu(img_gray_blur)
    except Exception:
        th = 128
    bw = img_gray_blur > th
    bw = remove_small_objects(bw, min_size=min_size)
    
    # Check if binary image is empty
    if not np.any(bw):
        return None, None
    
    distance = ndi.distance_transform_edt(bw)
    try:
        local_maxi = peak_local_max(distance, indices=False, labels=bw)
    except Exception:
        try:
            local_maxi = peak_local_max(distance, labels=bw)
        except Exception:
            return None, None
    
    markers, _ = ndi.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=bw)
    props, counts = np.unique(labels_ws, return_counts=True)
    if len(counts) <= 1:
        return None, None
    label_vals = props[1:]; counts_vals = counts[1:]
    label = label_vals[np.argmax(counts_vals)]
    mask = (labels_ws == label).astype('uint8') * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    
    # Compute contour features (matching paper)
    perimeter = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, True)
    num_vertices = len(approx)
    
    contour_features = {
        'perimeter': float(perimeter),
        'area': float(area),
        'epsilon': float(epsilon),
        'num_vertices': int(num_vertices)
    }
    
    # Store features if image_path provided
    if image_path and SAVE_CONTOUR_FEATURES:
        _contour_features_cache[image_path] = contour_features
    
    x, y, w, h = cv2.boundingRect(c)
    pad = int(0.1 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None, None
    # Ensure crop has valid dimensions before resizing
    if len(crop.shape) != 3 or crop.shape[2] != 3:
        return None, None
    try:
        crop_resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
        return cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB), contour_features
    except Exception as e:
        print(f"[WARN] Resize failed in segmentation: {e}, crop shape: {crop.shape}")
        return None, None

def preprocess_path(path, do_seg=None):
    """Preprocess image path. Handles both string and bytes input."""
    if do_seg is None:
        do_seg = config.USE_SEGMENTATION
    
    # Handle both string and bytes input (for TensorFlow compatibility)
    if isinstance(path, bytes):
        p = path.decode("utf-8")
    else:
        p = str(path)
    
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Failed to load image: {p}")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    
    try:
        if do_seg:
            seg, _ = segment_and_crop_cv(img, image_path=p)
            if seg is None:
                # Fallback: center crop if segmentation fails
                h, w = img.shape[:2]
                c = min(h, w)
                startx = (w - c) // 2
                starty = (h - c) // 2
                crop = img[starty:starty+c, startx:startx+c]
                crop = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
                img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = seg
        else:
            resized = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        img_norm = img_rgb.astype("float32") / 255.0
        return img_norm
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {p}: {e}")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

def augment_image(img):
    """Apply data augmentation if enabled."""
    if not USE_AUGMENTATION:
        return img
    # Random horizontal and vertical flip
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Random rotation (90 degree increments)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    # Random zoom via resize (safer than central crop)
    zoom_factor = tf.random.uniform([], 1.0 - AUGMENTATION_ZOOM, 1.0 + AUGMENTATION_ZOOM)
    h, w = IMG_SIZE[0], IMG_SIZE[1]
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
    # Ensure minimum size
    new_h = tf.maximum(new_h, h // 2)
    new_w = tf.maximum(new_w, w // 2)
    img = tf.image.resize(img, [new_h, new_w])
    # Crop or pad to original size
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    # Random brightness
    img = tf.image.random_brightness(img, AUGMENTATION_BRIGHTNESS)
    # Clip to [0, 1]
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def tf_preprocess(path, label, augment=False):
    """TensorFlow wrapper for `preprocess_path` used in evaluation datasets.

    Uses `tf.numpy_function` so it works correctly inside tf.data pipelines
    without relying on `.numpy()` calls on tensors.
    """
    img = tf.numpy_function(
        func=preprocess_path,
        inp=[path],
        Tout=tf.float32,
    )
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))

    if augment:
        img = augment_image(img)

    # Ensure label is a scalar int32 tensor
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [])
    return img, label

def preprocess_path_with_contours(path):
    """Preprocess image and return both image and contour features."""
    try:
        # Handle both string and bytes input
        if isinstance(path, bytes):
            p = path.decode("utf-8")
        else:
            p = str(path)
        
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Failed to load image: {p}")
            img_norm = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            contour_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return img_norm, contour_vec
        
        # Get contour features from cache or compute
        feats = None
        if p in _contour_features_cache:
            feats = _contour_features_cache[p]
        else:
            # Compute if not cached (should be cached from dataset creation)
            _, feats = segment_and_crop_cv(img, image_path=p)
            if feats is None or not isinstance(feats, dict):
                feats = {'perimeter': 0.0, 'area': 0.0, 'epsilon': 0.0, 'num_vertices': 0.0}
        
        # Ensure feats is a dict with the right keys
        if not isinstance(feats, dict):
            feats = {'perimeter': 0.0, 'area': 0.0, 'epsilon': 0.0, 'num_vertices': 0.0}
        
        contour_vec = np.array([
            float(feats.get('perimeter', 0.0)),
            float(feats.get('area', 0.0)),
            float(feats.get('epsilon', 0.0)),
            float(feats.get('num_vertices', 0))
        ], dtype=np.float32)
        
        # Ensure contour_vec has the correct shape
        if contour_vec.shape != (4,):
            contour_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Process image
        if config.USE_SEGMENTATION:
            seg, _ = segment_and_crop_cv(img, image_path=p)
            if seg is None:
                h, w = img.shape[:2]
                c = min(h, w)
                startx = (w - c) // 2
                starty = (h - c) // 2
                crop = img[starty:starty+c, startx:startx+c]
                crop = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
                img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = seg
        else:
            resized = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Ensure image has correct shape
        if img_rgb.shape != (IMG_SIZE[0], IMG_SIZE[1], 3):
            img_rgb = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
            if len(img_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            elif img_rgb.shape[2] != 3:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        img_norm = img_rgb.astype("float32") / 255.0
        return img_norm, contour_vec
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {p}: {e}")
        # Return default values on error
        img_norm = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        contour_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return img_norm, contour_vec

def tf_preprocess_with_contours(path, label, augment=False):
    """Preprocess returning (image, contour_features), label."""
    # Use tf.numpy_function for better compatibility
    img, contours = tf.numpy_function(
        func=preprocess_path_with_contours,
        inp=[path],
        Tout=(tf.float32, tf.float32)
    )
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    contours.set_shape((4,))  # 4 contour features
    if augment:
        img = augment_image(img)
    # Ensure label is a scalar tensor with proper shape
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [])
    return (img, contours), label

def save_contour_features_csv(dataset_name, class_names):
    """Save accumulated contour features to CSV file."""
    if not SAVE_CONTOUR_FEATURES or not _contour_features_cache:
        return
    csv_path = os.path.join(FEATURES_DIR, f"{dataset_name}_contours.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'class', 'perimeter', 'area', 'epsilon', 'num_vertices'])
        writer.writeheader()
        for img_path, features in _contour_features_cache.items():
            # Extract class name from path
            class_name = None
            for cls in class_names:
                if cls in img_path:
                    class_name = cls
                    break
            if class_name is None:
                class_name = "unknown"
            row = {'image_path': img_path, 'class': class_name, **features}
            writer.writerow(row)
    print(f"Saved contour features to {csv_path}")
    _contour_features_cache.clear()

def make_tf_dataset_from_folder(folder, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT, 
                                val_split=VAL_SPLIT, test_split=TEST_SPLIT, shuffle=True, 
                                return_test=False, dataset_name=None, return_contours=False):
    """
    Create TensorFlow datasets with 70/15/15 split (matching paper).
    
    Args:
        folder: Dataset folder with class subdirectories
        batch_size: Batch size
        train_split: Training split (default 0.70)
        val_split: Validation split (default 0.15)
        test_split: Test split (default 0.15)
        shuffle: Whether to shuffle data
        return_test: If True, return (train, val, test, classes), else (train, val, classes)
        dataset_name: Name for saving contour features CSV
        return_contours: If True, return datasets with contour features as second input
    
    Returns:
        train_ds, val_ds, (test_ds if return_test), classes
        If return_contours=True, datasets return ((image, contours), label) instead of (image, label)
    """
    classes = sorted([d.name for d in os.scandir(folder) if d.is_dir()])
    filepaths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(folder, cls)
        for fn in os.listdir(cls_dir):
            if not fn.lower().endswith((".png",".jpg",".jpeg",".tiff",".bmp")):
                continue
            p = os.path.join(cls_dir, fn)
            filepaths.append(p)
            labels.append(idx)
    combined = list(zip(filepaths, labels))
    random.shuffle(combined)
    if len(combined) == 0:
        raise RuntimeError("No images found in folder: " + folder)
    filepaths, labels = zip(*combined)
    n = len(filepaths)
    
    # Three-way split: 70/15/15
    train_end = int(train_split * n)
    val_end = train_end + int(val_split * n)
    
    train_files = filepaths[:train_end]
    train_labels = labels[:train_end]
    val_files = filepaths[train_end:val_end]
    val_labels = labels[train_end:val_end]
    test_files = filepaths[val_end:]
    test_labels = labels[val_end:]
    
    def gen(files, labs):
        for p,l in zip(files, labs):
            yield p.encode('utf-8'), np.int32(l)
    
    train_ds = tf.data.Dataset.from_generator(lambda: gen(train_files, train_labels),
                                              output_types=(tf.string, tf.int32))
    val_ds = tf.data.Dataset.from_generator(lambda: gen(val_files, val_labels),
                                            output_types=(tf.string, tf.int32))
    test_ds = tf.data.Dataset.from_generator(lambda: gen(test_files, test_labels),
                                             output_types=(tf.string, tf.int32))
    
    
    # Create wrapper functions for mapping
    def map_train(path, label):
        img = tf.numpy_function(func=preprocess_path, inp=[path], Tout=tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        img = augment_image(img)
        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, [])
        return img, label
    
    def map_val_test(path, label):
        img = tf.numpy_function(func=preprocess_path, inp=[path], Tout=tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, [])
        return img, label
    
    autotune = tf.data.AUTOTUNE
    
    if return_contours:
        train_ds = train_ds.map(
            lambda p, l: tf_preprocess_with_contours(p, l, augment=True),
            num_parallel_calls=autotune
        )
        val_ds = val_ds.map(
            lambda p, l: tf_preprocess_with_contours(p, l, augment=False),
            num_parallel_calls=autotune
        )
        test_ds = test_ds.map(
            lambda p, l: tf_preprocess_with_contours(p, l, augment=False),
            num_parallel_calls=autotune
        )
    else:
        train_ds = train_ds.map(map_train, num_parallel_calls=autotune)
        val_ds = val_ds.map(map_val_test, num_parallel_calls=autotune)
        test_ds = test_ds.map(map_val_test, num_parallel_calls=autotune)
    
    if shuffle:
        train_ds = train_ds.shuffle(1024)
    
    train_ds = train_ds.batch(batch_size).prefetch(autotune)
    val_ds = val_ds.batch(batch_size).prefetch(autotune)
    test_ds = test_ds.batch(batch_size).prefetch(autotune)
    
    # Save contour features if enabled
    if dataset_name:
        save_contour_features_csv(dataset_name, classes)
    
    if return_test:
        return train_ds, val_ds, test_ds, classes
    return train_ds, val_ds, classes

def make_eval_dataset_from_folder(folder, batch_size=BATCH_SIZE, return_contours=False):
    """
    Create evaluation dataset from folder (no train/val/test split - uses all images).
    Useful for evaluating on a separate test set or full dataset.
    
    Args:
        folder: Dataset folder with class subdirectories
        batch_size: Batch size
        return_contours: If True, return datasets with contour features as second input
    
    Returns:
        eval_ds, classes
    """
    classes = sorted([d.name for d in os.scandir(folder) if d.is_dir()])
    filepaths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(folder, cls)
        for fn in os.listdir(cls_dir):
            if not fn.lower().endswith((".png",".jpg",".jpeg",".tiff",".bmp")):
                continue
            p = os.path.join(cls_dir, fn)
            filepaths.append(p)
            labels.append(idx)
    
    if len(filepaths) == 0:
        raise RuntimeError("No images found in folder: " + folder)
    
    def gen(files, labs):
        for p, l in zip(files, labs):
            yield p.encode('utf-8'), np.int32(l)
    
    eval_ds = tf.data.Dataset.from_generator(lambda: gen(filepaths, labels),
                                             output_types=(tf.string, tf.int32))
    
    autotune = tf.data.AUTOTUNE
    if return_contours:
        # Preprocess to cache contour features
        for p in filepaths:
            img = cv2.imread(p)
            if img is not None:
                segment_and_crop_cv(img, image_path=p)
        eval_ds = eval_ds.map(lambda p,l: tf_preprocess_with_contours(p, l, augment=False), num_parallel_calls=autotune)
    else:
        eval_ds = eval_ds.map(lambda p,l: tf_preprocess(p, l, augment=False), num_parallel_calls=autotune)
    
    eval_ds = eval_ds.batch(batch_size).prefetch(autotune)
    return eval_ds, classes