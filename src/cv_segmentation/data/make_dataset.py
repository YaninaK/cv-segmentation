import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["load_and_clean_data"]


def load_data(config: Optional[dict] = None) -> Tuple[pd.DataFrame, list]:
    if config is None:
        config = CONFIG

    y_train = pd.read_csv(config["csv_file"], index_col=0)

    file_names = os.listdir(config["root_dir"])
    images_train = []
    for img_name in tqdm(file_names):
        img = np.load(os.path.join(config["root_dir"], img_name))
        images_train.append(img)

    return y_train, images_train


def clean_outliers(
    y_train: pd.DataFrame, images_train: list, config: Optional[float] = None
) -> Tuple[pd.DataFrame, np.array]:

    if config is None:
        config = CONFIG

    image_arr = np.reshape(np.asarray(images_train), (len(images_train), -1))
    image_arr_min = np.nan_to_num(image_arr).min(axis=1)

    y_train = y_train.iloc[image_arr_min > config["threshold"]].reset_index()

    return y_train, image_arr[image_arr_min > config["threshold"]]
