from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.callbacks import Callback
from skin_vision.config import config
from skin_vision.logger import logger

tf.compat.v1.disable_eager_execution()


@lru_cache
def load_mobilenetv2(
        input_shape: tuple = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3),
        include_top: bool = False,
        weights: str = "imagenet"
):
    """Load pre-trained MobileNetV2 with ImageNet weights."""
    return keras.applications.MobileNetV2(input_shape=input_shape, include_top=include_top, weights=weights)


@lru_cache
def load_efficientnetb7(
        input_shape: tuple = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3),
        include_top: bool = False,
        weights: str = "imagenet"
):
    """Load pre-trained EfficientNetB7 with ImageNet weights."""
    return keras.applications.EfficientNetB7(input_shape=input_shape, include_top=include_top, weights=weights)


def get_model(bias: float):

    bias = tf.keras.initializers.Constant(bias)
    base_model = load_efficientnetb7(
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model


class SkinVisionModel:
    def __init__(self, model_bias: float):
        self.model: Optional[keras.Model] = get_model(bias=model_bias)

    @property
    def callback_early_stopping(self) -> Callback:
        return tf.keras.callbacks.EarlyStopping(
            patience=15,
            verbose=0,
            restore_best_weights=True
        )

    @property
    def callbacks_lr_reduce(self) -> Callback:
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=10,
            verbose=0,
            min_lr=1e-6
        )

    @property
    def callback_checkpoint(self) -> Callback:
        return tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_WEIGHTS_PATH / "model_weights.h5",
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )

    def fit(
            self,
            training_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            class_weights: Dict[int, float],
            steps_per_epoch_training,
            steps_per_epoch_validation,
    ) -> tf.keras.callbacks.History:

        history = self.model.fit(
            training_dataset,
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch_training,
            validation_data=validation_dataset,
            validation_steps=steps_per_epoch_validation,
            callbacks=[self.callback_early_stopping, self.callbacks_lr_reduce, self.callback_checkpoint],
            class_weight=class_weights
        )

        logger.info(f"Model trained for {len(history.history['loss'])} epochs")

        return history

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Resizing the image
        image = np.asarray(image.resize((224, 224)))[..., :3]
        image = np.expand_dims(image, 0)
        image = image / 127.5 - 1.0

        return keras.applications.imagenet_utils.decode_predictions(preds=self.model.predict(image), top=2)[0]
