import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "cv_segmentation"))

import logging
from typing import Optional, Tuple

import torch
from data.make_dataset import clean_outliers, load_data
from data.validation import split_y_to_train_val_test
from features.get_dataset import SegmentationDataset
from features.transform import data_transform

from data import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["prepare_dataset"]


def data_preparation_pipeline(
    config: Optional[dict] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    if config is None:
        config = CONFIG

    y_train, images_train = load_data(config)
    y_train, _ = clean_outliers(y_train, images_train)
    split_y_to_train_val_test(y_train)

    stages = ["train", "val", "test"]

    transform = {"train": data_transform, "val": None, "test": None}
    segmentation_datasets = {
        stage: SegmentationDataset(
            csv_file=os.path.join(config["save_dir"], f"y_{stage}.csv"),
            root_dir=config["root_dir"],
            transform=transform[stage],
        )
        for stage in stages
    }
    dataset_loader = {
        stage: torch.utils.data.DataLoader(
            segmentation_datasets[stage], batch_size=4, shuffle=True, num_workers=4
        )
        for stage in stages
    }

    return segmentation_datasets, dataset_loader
