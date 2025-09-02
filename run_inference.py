# run_inference.py
import os
import numpy as np
import tensorflow as tf
from training.models.segnet import build_segnet
import matplotlib.pyplot as plt
from glob import glob
import cv2

# ----------------------------
# Paths
# ----------------------------
DATA_FILE = "data/processed/diaretdb1_processed.npz"
MODEL_WEIGHTS = "weights/segnet_best.h5"
OUTPUT_DIR = "inference_results"
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

# Create output directories
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ----------------------------
# Load preprocessed data
# ----------------------------
data = np.load(DATA_FILE)
X, Y = data["images"], data["masks"]

# Use last 5 images for demo or all for full inference
X_test, Y_test = X[-5:], Y[-5:]
print(f"Running inference on {len(X_test)} images...")

# ----------------------------
# Load model
# ----------------------------
model = build_segnet(input_shape=(512, 512, 3), n_classes=4)
model.load_weights(MODEL_WEIGHTS)
print("✅ Model loaded.")

# ----------------------------
# Predict
# ----------------------------
preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(np.uint8)  # threshold 0.5

# ----------------------------
# Save masks per lesion type
# ----------------------------
lesion_names = ["MA", "HE", "SE", "EX"]

for i, pred in enumerate(preds_binary):
    for ch, name in enumerate(lesion_names):
        mask_path = os.path.join(MASK_DIR, f"image_{i}_{name}.png")
        cv2.imwrite(mask_path, pred[:, :, ch] * 255)  # save binary mask
print(f"✅ Saved per-lesion masks to {MASK_DIR}")

# ----------------------------
# Save overlay images
# ----------------------------
for i, (img, pred) in enumerate(zip(X_test, preds_binary)):
    combined_mask = np.max(pred, axis=-1)
    overlay = (img * 255).astype(np.uint8).copy()
    overlay_mask = (combined_mask * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.7, cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
    overlay_path = os.path.join(OVERLAY_DIR, f"image_{i}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
print(f"✅ Saved overlay images to {OVERLAY_DIR}")

# ----------------------------
# Optional visualization
# ----------------------------
for i in range(len(X_test)):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
    plt.title("Image")
    plt.axis('off')

    combined_mask = np.max(preds_binary[i], axis=-1)
    plt.subplot(2, 5, i+6)
    plt.imshow(X_test[i])
    plt.imshow(combined_mask, alpha=0.5, cmap='Reds')
    plt.title("Prediction Overlay")
    plt.axis('off')

plt.show()

print("✅ Inference complete.")
