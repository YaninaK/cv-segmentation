import numpy as np


def data_transform(sample: dict) -> dict:
    """
    Apply horizontal and vertical flips with 50% probability
    """
    image = sample["image"]
    mask = sample["mask"]

    if np.random.random() > 0.5:
        image = np.fliplr(image)  # Horizontal flip
        mask = np.fliplr(mask)

    if np.random.random() > 0.5:
        image = np.flipud(image)  # Vertical flip
        mask = np.flipud(mask)

    sample = {"image": image, "mask": mask}

    return sample
