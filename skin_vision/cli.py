import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from functools import partial

from skin_vision.config import config
from skin_vision.logger import logger
from skin_vision.model import SkinVisionModel
from skin_vision.process_data import get_dataset
import typer
from skin_vision.transformations import augmentate_images


app = typer.Typer()


@app.command()
def train():
    """
    Note: we use the whole training dataset since we are not doing model selection ATM.
    :return:
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    augmentate = False

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        print("Device:", tpu.master())
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.get_strategy()

    print("Number of replicas:", strategy.num_replicas_in_sync)

    df_train = pd.read_csv(config.ROOT_DATA_PATH / 'train.csv')
    df_test = pd.read_csv(config.ROOT_DATA_PATH / 'test.csv')

    files_train = np.sort(np.array(tf.io.gfile.glob([x.as_posix() for x in (config.ROOT_DATA_PATH / "tfrecords").glob("train*.tfrec")])))
    files_test = np.sort(np.array(tf.io.gfile.glob([x.as_posix() for x in (config.ROOT_DATA_PATH / "tfrecords").glob("test*.tfrec")])))
    files_train, files_validation = train_test_split(files_train)

    logger.info(f"Number of training files: {len(files_train)}")
    logger.info(f"Number of validation files: {0}")
    logger.info(f"Number of test files: {len(files_test)}")

    value_counts = df_train["target"].value_counts().to_dict()
    class_weights = {k: v / len(df_train) for k, v in value_counts.items()}

    logger.info(f"Percent benign cases = {class_weights[0]:.2%}")
    logger.info(f"Percent malignant cases = {class_weights[1]:.2%}")

    ds_train = get_dataset(files_train, labeled=True)
    ds_validation = get_dataset(files_validation, labeled=True)

    print("\n Begin Training Models")
    steps_per_epoch_training = np.ceil(len(df_train) * 0.75 / config.BATCH_SIZE)
    steps_per_epoch_validation = np.ceil(len(df_train) * 0.25 / config.BATCH_SIZE)

    model = SkinVisionModel(model_bias=np.log(class_weights[0] / class_weights[1]))

    if augmentate:
        augmented_training_Data = ds_train.map(partial(augmentate_images, img_size=120), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    training_history = model.fit(
        training_dataset=ds_train,
        validation_dataset=ds_validation,
        class_weights=class_weights,
        steps_per_epoch_training=steps_per_epoch_training,
        steps_per_epoch_validation=steps_per_epoch_validation,
    )
    print("\n Done Training model_B6 \n")


@app.command()
def predict():
    pass


if __name__ == "__main__":
    typer.run(train)
