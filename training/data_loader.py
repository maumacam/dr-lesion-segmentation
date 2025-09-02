import os
import shutil
import random

# Paths
ROOT_DIR = "data/diaretdb1/resources/images"
FUNDUS_DIR = os.path.join(ROOT_DIR, "diaretdb0_fundus_images")
MASK_DIR = os.path.join(ROOT_DIR, "diaretdb0_binary_masks")

OUTPUT_DIR = "data/processed"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Create output directories
for d in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "masks"), exist_ok=True)

# Collect all images
images = [f for f in os.listdir(FUNDUS_DIR) if f.endswith(".png")]
images.sort()

# Shuffle for randomness
random.shuffle(images)

# Split ratio
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
test_images = images[split_idx:]

def copy_files(image_list, split_dir):
    for img_file in image_list:
        img_id = img_file.replace(".png", "")
        img_src = os.path.join(FUNDUS_DIR, img_file)
        mask_file = img_id + "_valid.png"
        mask_src = os.path.join(MASK_DIR, mask_file)

        # Check if mask exists
        if os.path.exists(mask_src):
            img_dst = os.path.join(split_dir, "images", img_file)
            mask_dst = os.path.join(split_dir, "masks", mask_file)
            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"⚠️ No mask for {img_id}, skipping.")

# Copy train and test sets
copy_files(train_images, TRAIN_DIR)
copy_files(test_images, TEST_DIR)

print(f"✅ Dataset prepared!")
print(f"Training set: {len(train_images)} images")
print(f"Test set: {len(test_images)} images")
print(f"Output saved in {OUTPUT_DIR}")
