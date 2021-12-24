import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import List
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
import gc

from skin_vision.config import config
from skin_vision.logger import logger
from skin_vision.model import SkinVisionModel
from skin_vision.utils import count_data_items
from skin_vision.io import get_training_dataset, get_validation_dataset, get_test_dataset


def main():

    path_train_csv = config.ROOT_DATA_PATH / "train.csv"
    path_test_csv = config.ROOT_DATA_PATH / "test.csv"
    path_jpg_images = config.ROOT_DATA_PATH / "jpeg" / "train"

    tfrec_train_files = [x.as_posix() for x in (config.ROOT_DATA_PATH / "tfrecords").glob("train*.tfrec")]
    tfrec_test_files = [x.as_posix() for x in (config.ROOT_DATA_PATH / "tfrecords").glob("test*.tfrec")]

    training_files, validation_files = train_test_split(
        tf.io.gfile.glob(tfrec_train_files),
        test_size=0.1,
        random_state=42
    )

    testing_files = tf.io.gfile.glob(tfrec_test_files)

    print("Number of training files = ", len(training_files))
    print("Number of validation files = ", len(validation_files))
    print("Number of test files = ", len(testing_files))

    training_dataset = get_training_dataset(training_files=training_files)
    validation_dataset = get_validation_dataset(validation_files=validation_files)
    test_dataset = get_test_dataset(testing_files=testing_files)

    num_training_images = count_data_items(training_files)
    num_validation_images = count_data_items(validation_files)
    num_testing_images = count_data_items(testing_files)

    STEPS_PER_EPOCH_TRAIN = num_training_images // config.BATCH_SIZE
    STEPS_PER_EPOCH_VAL = num_validation_images // config.BATCH_SIZE

    print("Number of Training Images = ", num_training_images)
    print("Number of Validation Images = ", num_validation_images)
    print("Number of Testing Images = ", num_testing_images)
    print("\n")
    print("Numer of steps per epoch in Train = ", STEPS_PER_EPOCH_TRAIN)
    print("Numer of steps per epoch in Validation = ", STEPS_PER_EPOCH_VAL)

    gc.collect()

    model = SkinVisionModel()
    model.train(training_dataset=training_dataset, validation_dataset=validation_dataset)


if __name__ == "__main__":
    main()
