from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.python.keras.backend as K
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skin_vision.config import config
from skin_vision.logger import logger

tf.compat.v1.disable_eager_execution()


@lru_cache
def load_mobilenetv2(
        input_shape: tuple = (config.SHAPE[0], config.SHAPE[1], 3),
        include_top: bool = False,
        weights: str = "imagenet"
):
    """Load pre-trained MobileNetV2 with ImageNet weights."""
    return keras.applications.MobileNetV2(input_shape=input_shape, include_top=include_top, weights=weights)


@lru_cache
def load_efficientnetb7(
        input_shape: tuple = (config.SHAPE[0], config.SHAPE[1], 3),
        include_top: bool = False,
        weights: str = "imagenet"
):
    """Load pre-trained EfficientNetB7 with ImageNet weights."""
    return keras.applications.EfficientNetB7(input_shape=input_shape, include_top=include_top, weights=weights)


class SkinVisionModel:
    def __init__(self):
        self._base_model: keras.Model = load_efficientnetb7()
        self.model: Optional[keras.Model] = None

    def train(self, training_dataset, validation_dataset):

        value_counts = training_dataset[training_dataset].value_counts()
        class_weights = {k: v / len(training_dataset) for k, v in value_counts.items()}
        malignant, benign = class_weights[1], class_weights[0]

        print(f"Weight for benign cases = {class_weights[0]}")
        print(f"Weight for malignant cases = {class_weights[1]}")

        callback_early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=15,
            verbose=0,
            restore_best_weights=True
        )

        callbacks_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=10,
            verbose=0,
            min_lr=1e-6
        )

        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_WEIGHTS_PATH / "model_weights.h5",
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )

        bias = np.log(malignant / benign)
        bias = tf.keras.initializers.Constant(bias)
        self._base_model = tf.keras.applications.MobileNetV2(
            input_shape=(config.SHAPE[0], config.SHAPE[1], 3),
            include_top=False,
            weights="imagenet"
        )
        self._base_model.trainable = False
        self.model = tf.keras.Sequential([
            self._base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias)
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-2),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        self.model.summary()

        EPOCHS = 500
        history = self.model.fit(
            training_dataset,
            epochs=EPOCHS,
            steps_per_epoch=config.STEPS_PER_EPOCH_TRAIN,
            validation_data=validation_dataset,
            validation_steps=config.STEPS_PER_EPOCH_VAL,
            callbacks=[callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
            class_weight=class_weights
        )

        logger.info(f"Model trained for {len(history.history['loss'])} epochs")

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Resizing the image
        image = np.asarray(image.resize((224, 224)))[..., :3]
        image = np.expand_dims(image, 0)
        image = image / 127.5 - 1.0

        return keras.applications.imagenet_utils.decode_predictions(preds=self.model.predict(image), top=2)[0]


if __name__ == "__main__":

    model = SkinVisionModel()
    pass