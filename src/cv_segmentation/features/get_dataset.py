import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "cv_segmentation"))

import logging
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["generate_dataset"]


class SegmentationDataset(Dataset):
    """Segmentation dataset."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        config: Optional[dict] = None,
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            config (dict): Contains outlier_threshold to detect outliers.
        """
        if config is None:
            config = CONFIG

        self.mask_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.mask_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, f"{self.mask_frame.iloc[idx, 0]}.npy")
        image = np.load(img_name)
        image = np.nan_to_num(image, nan=np.quantile(np.nan_to_num(image), 0.5))
        image = np.where(
            image < self.config["outlier_threshold"], np.quantile(image, 0.5), image
        )
        image = torch.from_numpy(image)
        mask = self.mask_frame.iloc[idx, 1:]
        mask = torch.from_numpy(np.array(mask, dtype=float).reshape(36, 36))

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
