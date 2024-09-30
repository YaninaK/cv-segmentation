import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset
from tqdm import tqdm


class SegmentationDataset(Dataset):
    """Segmentation dataset."""

    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask_frame = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.clean_outliers()
        self.train_val_test_ratio = [0.7, 0.2, 0.1]
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.get_train_val_test_idx()

    def __len__(self):
        return len(self.mask_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, f"{self.mask_frame.iloc[idx, 0]}.npy")
        image = np.nan_to_num(np.load(img_name))
        image = torch.from_numpy(image)
        mask = self.mask_frame.iloc[idx, 1:]
        mask = torch.from_numpy(np.array(mask, dtype=float).reshape(36, 36))

        # Label refinement
        mask = binary_fill_holes(mask)
        mask = self.apply_dilation(mask)

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def clean_outliers(self):
        file_names = os.listdir(self.root_dir)
        images_train = []
        for img_name in tqdm(file_names):
            img = np.load(os.path.join(self.root_dir, img_name))
            images_train.append(img)

        image_arr = np.reshape(np.asarray(images_train), (len(images_train), -1))
        image_arr_min = np.nan_to_num(image_arr).min(axis=1)

        self.mask_frame = self.mask_frame.iloc[image_arr_min > -0.25].reset_index()


    def get_train_val_test_idx(self):
        
        n = self.mask_frame.shape[0]
        n_train = int(self.train_val_test_ratio[0] * n)
        n_test = int(self.train_val_test_ratio[2] * n)

        dataset_idx = self.mask_frame.index.tolist()
        train_idx = random.sample(dataset_idx, n_train)
        val_idx = list(set(dataset_idx) - set(train_idx))
        test_idx = random.sample(val_idx, n_test)
        val_idx = list(set(val_idx) - set(test_idx))

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx 

    def apply_dilation(self, mask):
        """
        Structuring Element & Morphological Operations
        """
        kernel_size = (2, 2)
        kernel_shape = cv2.MORPH_RECT
        # Create the adaptive structuring element
        adaptive_kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
        # Apply morphological operations : dilation
        mask = cv2.dilate(np.ones((36, 36)) * mask, adaptive_kernel)

        return mask
