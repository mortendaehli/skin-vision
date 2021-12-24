import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook as tqdm
from scipy.stats import skew
from pathlib import Path
from typing import List
from skin_vision.config import Config


def get_image_names(df: pd.DataFrame) -> np.array:
    return df["image_name"].str.map(lambda x: x + ".jpg")


def extract_information(image_names: List[str], directory: Path):
    image_statistics = pd.DataFrame(index=np.arange(len(image_names)),
                                    columns=["image_name", "path", "rows", "columns", "channels",
                                             "image_mean", "image_standard_deviation", "image_skewness",
                                             "mean_red_value", "mean_green_value", "mean_blue_value"])
    i = 0
    for name in tqdm(image_names):
        path = directory / name
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_statistics.iloc[i]["image_name"] = name
        image_statistics.iloc[i]["path"] = path
        image_statistics.iloc[i]["rows"] = image.shape[0]
        image_statistics.iloc[i]["columns"] = image.shape[1]
        image_statistics.iloc[i]["channels"] = image.shape[2]
        image_statistics.iloc[i]["image_mean"] = np.mean(image.flatten())
        image_statistics.iloc[i]["image_standard_deviation"] = np.std(image.flatten())
        image_statistics.iloc[i]["image_skewness"] = skew(image.flatten())
        image_statistics.iloc[i]["mean_red_value"] = np.mean(image[:, :, 0])
        image_statistics.iloc[i]["mean_green_value"] = np.mean(image[:, :, 1])
        image_statistics.iloc[i]["mean_blue_value"] = np.mean(image[:, :, 2])

        i = i + 1
        del image

    return image_statistics


if __name__ == "__main__":
    df_train = pd.read_csv(Config.ROOT_DATA_PATH / "jpeg" / "train.csv")
    df_test = pd.read_csv(Config.ROOT_DATA_PATH / "jpeg" / "test.csv")

    image_names = get_image_names(df_train)
