from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pca import pca

from . import CONFIG


class PreprocessData:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = CONFIG

        self.config = config
        self.to_delete = None
        self.fill_nan_list = None

    def fit_transform(
        self, y_train: pd.DataFrame, images_train: list[np.array]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        image_frame = self._fill_nan(y_train, images_train)
        image_frame = self._clean_univariate_outliers(image_frame)
        if self.to_delete:
            selected_idx = [i for i in y_train.index if i not in self.to_delete]
        else:
            selected_idx = y_train.index.tolist()
        self._clean_multivariate_outliers(image_frame.loc[selected_idx])

        return y_train.loc[self.selected_idx], image_frame.loc[self.selected_idx]

    def _fill_nan(
        self, y_train: pd.DataFrame, images_train: list[np.array]
    ) -> pd.DataFrame:
        image_arr = np.reshape(np.asarray(images_train), (len(images_train), -1))
        image_frame = pd.DataFrame(image_arr, y_train.index)

        n_missing = image_frame.isnull().sum(axis=1)
        self.to_delete = image_frame[
            n_missing >= self.config["fill_nan_threshold"]
        ].index.tolist()

        fill_nan_condition = (n_missing > 0) & (
            n_missing < self.config["fill_nan_threshold"]
        )
        self.fill_nan_list = image_frame[fill_nan_condition].index.tolist()
        image_frame.mask(
            fill_nan_condition, image_frame.median(axis=1), axis=0, inplace=True
        )
        return image_frame

    def _clean_univariate_outliers(self, image_frame: pd.DataFrame) -> pd.DataFrame:
        n_outliers = (image_frame < self.config["outlier_threshold"]).sum(axis=1)
        self.to_delete += n_outliers[
            n_outliers >= self.config["n_outlier_threshold"]
        ].index.tolist()
        image_frame.mask(
            image_frame < self.config["outlier_threshold"],
            image_frame.median(axis=1),
            axis=0,
            inplace=True,
        )
        return image_frame

    def _clean_multivariate_outliers(self, image_frame: pd.DataFrame) -> None:
        model = pca(alpha=self.config["alpha"], detect_outliers=["spe"])
        results = model.fit_transform(image_frame)
        delete_list = image_frame[results["outliers"]["y_bool_spe"]].index.tolist()
        self.to_delete += delete_list

        self.selected_idx = [i for i in image_frame.index if i not in delete_list]
