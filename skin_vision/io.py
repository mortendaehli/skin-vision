from functools import partial
from typing import List

from skin_vision.config import config
from skin_vision.utils import read_tfrecord, image_augmentation

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_dataset(filenames, labeled, ordered: bool = False):
    ignore_order = tf.data.Options()
    if not ordered:  # dataset is unordered, so we ignore the order to load data quickly.
        ignore_order.experimental_deterministic = False  # This disables the order and enhances the speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset


def get_training_dataset(training_files: List[str]):
    dataset = load_dataset(training_files, labeled=True, ordered=False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(validation_files: List[str]):
    dataset = load_dataset(validation_files, labeled=True, ordered=False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(testing_files: List[str]):
    dataset = load_dataset(testing_files, labeled=False, ordered=True)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
