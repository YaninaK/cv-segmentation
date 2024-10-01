import logging
import os
from typing import Optional

import pandas as pd

from . import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["train_test_split"]


def split_y_to_train_val_test(
    y_train: pd.DataFrame,
    config: Optional[dict] = None,
) -> None:
    if config is None:
        config = CONFIG

    y_train["well"] = y_train["index"].apply(lambda x: int(x.split("_")[1]))

    for stage in ["train", "val", "test"]:
        y_train.loc[y_train["well"].isin(config[stage]), y_train.columns[:-1]].to_csv(
            os.path.join(config["save_dir"], f"y_{stage}.csv"), index=False
        )
