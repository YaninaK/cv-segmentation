CONFIG = {
    "csv_file": "../data/raw/y_train.csv",
    "root_dir": "/content/cv-segmentation/data/raw/images/",
    "save_dir": "../data/interim/",
    "fill_nan_threshold": 100,
    "outlier_threshold": -0.25,
    "n_outlier_threshold": 200,
    "alpha": 0.05,
    "train": [1, 6, 7, 8, 10, 11, 12, 13, 14],
    "val": [3, 9],
    "test": [2, 4, 5, 15],
    "batch_size": {
        "train": 16,
        "val": 16,
        "test": 4,
    },
    "num_workers": {
        "train": 4,
        "val": 4,
        "test": 4,
    },
}
