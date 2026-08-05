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
        tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 1)),

        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(model, train_ds, val_ds, epochs=10):
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    return history

def save_model(model, model_dir="models/arthritis_grade_classifier.keras"):
    model.save(model_dir)