import torch


def dice(prediction: torch.tensor, target: torch.tensor, target_one_hot=True) -> torch.tensor:

    if not target_one_hot:
        target = torch.eye(len(target))[target]

    prediction_mask = (prediction > 0.5).int()

    TP = prediction_mask * target
    FP = prediction_mask - TP
    FN = target - TP

    dice_score = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum() + 1e-06)

    return dice_score
