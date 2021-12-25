import numpy as np
import tensorflow as tf
from functools import partial
from skin_vision.config import config

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(files: np.array, labeled=True):
    dataset = load_dataset(files, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset


def load_dataset(files: np.array, labeled: bool = True) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(files)
    return dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            'image': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.int64)
        }
        if labeled
        else {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*config.IMAGE_SIZE, 3])
    return image
