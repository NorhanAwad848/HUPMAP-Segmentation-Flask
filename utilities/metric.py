import numpy as np
from utilities.image_handler import apply_threshold


# calculate IoU, and Dice
def calculate_metrics(y_true, y_pred):
    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Dice coefficient
    dice_denominator = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denominator if dice_denominator != 0 else 1

    # Intersection over Union (IoU)
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator != 0 else 1

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 1

    # Recall
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1

    # F1 Score
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 1

    return dice, iou, precision, recall, f1_score


def get_best_thresh(y, y_hat, metrics=calculate_metrics):
    thresh = 0
    best_score = -1
    for i in range(10, 100, 10):
        yhat = apply_threshold(y_hat, i/100)
        metric = metrics(y, yhat)[0]

        if metric > best_score:
            best_score = metric
            thresh = i / 100

    return thresh
