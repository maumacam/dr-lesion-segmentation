import os
import cv2
import numpy as np
from glob import glob

# Paths
TRAIN_IMG_DIR = "data/processed/train/images"
TRAIN_MASK_DIR = "data/processed/train/masks"
TEST_IMG_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

OUTPUT_DIR = "data/processed"
TARGET_SIZE = (512, 512)  # Model expects this size

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(image_path, mask_path, target_size):
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # Normalize

    # Load and resize mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Convert single-channel mask into 4 binary channels
    mask_multi = np.zeros((target_size[0], target_size[1], 4), dtype=np.uint8)
    for i in range(4):
        mask_multi[:, :, i] = (mask > 0).astype(np.uint8)

    return img, mask_multi.astype(np.float32)

def process_dataset(img_dir, mask_dir, target_size):
    images = sorted(glob(os.path.join(img_dir, "*.png")))
    masks = sorted(glob(os.path.join(mask_dir, "*.png")))

    X, Y = [], []

    for img_path, mask_path in zip(images, masks):
        img, mask = load_and_preprocess(img_path, mask_path, target_size)
        X.append(img)
        Y.append(mask)

    # Convert to arrays in float16 to save memory
    return np.array(X, dtype=np.float16), np.array(Y, dtype=np.float16)

if __name__ == "__main__":
    print("🔍 Preprocessing training data...")
    X_train, Y_train = process_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TARGET_SIZE)
    print(f"✔ Train set: {X_train.shape} images, {Y_train.shape} masks")

    print("🔍 Preprocessing test data...")
    X_test, Y_test = process_dataset(TEST_IMG_DIR, TEST_MASK_DIR, TARGET_SIZE)
    print(f"✔ Test set: {X_test.shape} images, {Y_test.shape} masks")

    # Save separately
    np.savez_compressed(os.path.join(OUTPUT_DIR, "X_train.npz"), images=X_train, masks=Y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "X_test.npz"), images=X_test, masks=Y_test)

    print(f"✅ Saved preprocessed train and test datasets separately in {OUTPUT_DIR}")
