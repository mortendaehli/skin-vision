import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_array_as_image(data: np.array) -> None:
    i = tf.keras.preprocessing.image.array_to_img(data)
    plt.imshow(i, cmap='gray')


def plot_multiple_arrays_as_images(data: np.array, n_width: int, n_height: int) -> None:
    plt.figure(figsize=(7, 7))
    for i in range(int(n_width * n_height)):
        plt.subplot(n_width, n_height, i + 1)
        plot_array_as_image(data[i])


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            plt.title("MALIGNANT")
        else:
            plt.title("BENIGN")
        plt.axis("off")
