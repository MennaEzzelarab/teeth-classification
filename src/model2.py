import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import normalization_layer, data_augmentation

def build_teeth_model(input_shape=(256, 256, 3), num_classes=7):
    # 1. Define Input
    inputs = tf.keras.Input(shape=input_shape)

    # 2. Add Preprocessing layers
    x = data_augmentation(inputs) 
    x = normalization_layer(x)

    # 3. Convolutional blocks 
    # Block 1: 64 Filters
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2: 128 Filters
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3: 256 Filters
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 4. Final Layers 
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model