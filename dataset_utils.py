# dataset_utils.py
import os, random
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, RANDOM_SEED, RAW_DIR, USE_SEGMENTATION
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def segment_and_crop_cv(img_bgr, min_size=500):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        th = threshold_otsu(img_gray)
    except Exception:
        th = 128
    bw = img_gray > th
    bw = remove_small_objects(bw, min_size=min_size)
    distance = ndi.distance_transform_edt(bw)
    try:
        local_maxi = peak_local_max(distance, indices=False, labels=bw)
    except Exception:
        local_maxi = peak_local_max(distance, labels=bw)
    markers, _ = ndi.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=bw)
    props, counts = np.unique(labels_ws, return_counts=True)
    if len(counts) <= 1:
        return None
    label_vals = props[1:]; counts_vals = counts[1:]
    label = label_vals[np.argmax(counts_vals)]
    mask = (labels_ws == label).astype('uint8') * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = int(0.1 * max(w,h))
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
    return cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

def preprocess_path(path, do_seg=USE_SEGMENTATION):
    p = path.decode("utf-8")
    img = cv2.imread(p)
    if img is None:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    if do_seg:
        seg = segment_and_crop_cv(img)
        if seg is None:
            h,w = img.shape[:2]
            c = min(h,w)
            startx = (w-c)//2; starty = (h-c)//2
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

def tf_preprocess(path, label):
    img = tf.py_function(func=preprocess_path, inp=[path], Tout=tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label

def make_tf_dataset_from_folder(folder, batch_size=BATCH_SIZE, val_split=0.2, shuffle=True):
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
    split = int((1.0 - val_split) * n)
    train_files = filepaths[:split]; train_labels = labels[:split]
    val_files = filepaths[split:]; val_labels = labels[split:]
    def gen(files, labs):
        for p,l in zip(files, labs):
            yield p.encode('utf-8'), np.int32(l)
    train_ds = tf.data.Dataset.from_generator(lambda: gen(train_files, train_labels),
                                              output_types=(tf.string, tf.int32))
    val_ds = tf.data.Dataset.from_generator(lambda: gen(val_files, val_labels),
                                            output_types=(tf.string, tf.int32))
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda p,l: tf_preprocess(p,l), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda p,l: tf_preprocess(p,l), num_parallel_calls=autotune)
    if shuffle:
        train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.batch(batch_size).prefetch(autotune)
    val_ds = val_ds.batch(batch_size).prefetch(autotune)
    return train_ds, val_ds, classes