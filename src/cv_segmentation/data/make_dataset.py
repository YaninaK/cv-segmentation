import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["load_and_clean_data"]


def load_data(config: Optional[dict] = None) -> Tuple[pd.DataFrame, list[np.array]]:
    if config is None:
        config = CONFIG

    y_train = pd.read_csv(config["csv_file"], index_col=0)

    images_train = []
    for index in tqdm(y_train.index):
        img = np.load(os.path.join(config["root_dir"], f"{index}.npy"))
        images_train.append(img)

    return y_train, images_train
