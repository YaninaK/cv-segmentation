import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["load_data"]


def clean_outliers(csv_file: str, root_dir: str) -> Tuple[pd.DataFrame, list]:
    y_train = pd.read_csv(csv_file, index_col=0)

    file_names = os.listdir(root_dir)
    images_train = []
    for img_name in tqdm(file_names):
        img = np.load(os.path.join(root_dir, img_name))
        images_train.append(img)

    image_arr = np.reshape(np.asarray(images_train), (len(images_train), -1))
    image_arr_min = np.nan_to_num(image_arr).min(axis=1)

    y_train = y_train.iloc[image_arr_min > -0.25].reset_index()

    return y_train, images_train
