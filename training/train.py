# training/train.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to fix memory issues

"""
Training script for SegNet and U-Net++ on DIARETDB1 dataset.
Loads preprocessed data from preprocess.py output (diaretdb1_processed.npz).
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from models.segnet import build_segnet
from models.unetpp import build_unetpp

# ----------------------
# Loss functions & metrics
# ----------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

# ----------------------
# Training function
# ----------------------

def main(args):
    # Load preprocessed data
    data = np.load(args.data)
    
    if "X_train" in data and "X_test" in data:
        X_train, Y_train = data["X_train"], data["Y_train"]
        X_test, Y_test = data["X_test"], data["Y_test"]
    else:
        # fallback: single dataset, split manually
        X, Y = data["images"], data["masks"]
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Convert masks to float32
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    # Select model
    if args.model == 'segnet':
        model = build_segnet(input_shape=(512, 512, 3), n_classes=4)
    elif args.model == 'unetpp':
        model = build_unetpp(input_shape=(512, 512, 3), n_classes=4)
    else:
        raise ValueError("Invalid model name. Choose 'segnet' or 'unetpp'")

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=bce_dice_loss,
                  metrics=[dice_coefficient])

    model.summary()

    # Callbacks
    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, f"{args.model}_best.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save final model
    final_path = os.path.join(args.out, f"{args.model}_final.h5")
    model.save(final_path)
    print(f"✅ Training complete. Best model saved to {checkpoint_path}")
    print(f"✅ Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed .npz file')
    parser.add_argument('--model', type=str, default='segnet', choices=['segnet', 'unetpp'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)  # Reduced to 4 for memory safety
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='weights')
    args = parser.parse_args()

    main(args)