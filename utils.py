# utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def save_sample_predictions(model, dataset, class_names, out_prefix, max_images=8):
    i = 0
    for x,y in dataset.unbatch().batch(1):
        if i >= max_images: break
        pred = model.predict(x)
        pred_label = np.argmax(pred, axis=1)[0]
        true_label = int(y.numpy()[0])
        img = x[0].numpy()
        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.title(f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join("results", f"{out_prefix}_sample_{i}.png"))
        plt.close()
        i += 1