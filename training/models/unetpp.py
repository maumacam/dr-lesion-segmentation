"""
U-Net++ model implementation in TensorFlow/Keras.
Multi-label segmentation with 4 output channels (sigmoid activation).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def build_unetpp(input_shape=(512, 512, 3), n_classes=4):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x0_0 = conv_block(inputs, 64)
    x1_0 = layers.MaxPooling2D((2, 2))(x0_0)
    x1_0 = conv_block(x1_0, 128)
    x2_0 = layers.MaxPooling2D((2, 2))(x1_0)
    x2_0 = conv_block(x2_0, 256)
    x3_0 = layers.MaxPooling2D((2, 2))(x2_0)
    x3_0 = conv_block(x3_0, 512)
    x4_0 = layers.MaxPooling2D((2, 2))(x3_0)
    x4_0 = conv_block(x4_0, 1024)

    # Skip connections (simplified dense connections)
    x0_1 = conv_block(layers.concatenate([x0_0, layers.UpSampling2D((2, 2))(x1_0)]), 64)
    x0_2 = conv_block(layers.concatenate([x0_0, x0_1, layers.UpSampling2D((4, 4))(x2_0)]), 64)

    # Decoder
    x = layers.UpSampling2D((16, 16))(x4_0)
    x = layers.concatenate([x, x0_2])
    x = conv_block(x, 64)

    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(x)

    # ✅ Safe model name
    model = models.Model(inputs, outputs, name='UNetPP')  # Removed '++'
    return model

if __name__ == "__main__":
    model = build_unetpp()
    model.summary()
