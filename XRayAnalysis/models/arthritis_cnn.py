import tensorflow as tf
from pathlib import Path

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32
NUM_CLASSES = 5

def load_datasets(data_dir="data"):
    data_dir = Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=False
    )

    return train_ds, val_ds

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])