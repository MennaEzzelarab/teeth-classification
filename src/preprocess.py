import tensorflow as tf
import os

def load_datasets(base_path, img_size=(256, 256), batch_size=32):
    # Construct paths carefully
    train_dir = os.path.join(base_path, "Teeth DataSet/Teeth_Dataset/Training")
    val_dir = os.path.join(base_path, "Teeth DataSet/Teeth_Dataset/Validation")
    
    # 1. Load data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size
    )
    test_dir = os.path.join(base_path, "Teeth DataSet/Teeth_Dataset/Testing")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size
    )
    
    # 2. Performance tuning (CRITICAL for tf.data)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

#Normalization Layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

#Augmentation Layer
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

