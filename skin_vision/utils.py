import re

import cv2
import numpy as np
import tensorflow as tf

from skin_vision.config import config


def denoise_image(image: np.array) -> np.array:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def histogram_equalization(image: np.array) -> np.array:
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    y_channel = image_ycrcb[:, :, 0]  # apply local histogram processing on this channel
    cr_channel = image_ycrcb[:, :, 1]
    cb_channel = image_ycrcb[:, :, 2]

    # Local histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(y_channel)
    equalized_image = cv2.merge([equalized, cr_channel, cb_channel])
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)
    return equalized_image


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3])
    return image


def parse_function(example):
    feature_description = {"image": tf.io.FixedLenFeature([], tf.string),
                           "target": tf.io.FixedLenFeature([], tf.int64)}

    return tf.io.parse_single_example(example, feature_description)


def read_tfrecord(example, labeled):
    if labeled is True:
        tfrecord_format = {"image": tf.io.FixedLenFeature([], tf.string),
                           "target": tf.io.FixedLenFeature([], tf.int64)}
    else:
        tfrecord_format = {"image": tf.io.FixedLenFeature([], tf.string),
                           "image_name": tf.io.FixedLenFeature([], tf.string)}

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled is True:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    else:
        image_name = example["image_name"]
        return image, image_name


def image_augmentation(image, label):
    image = tf.image.resize(image, config.SHAPE)
    image = tf.image.random_flip_left_right(image)
    return image, label


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

