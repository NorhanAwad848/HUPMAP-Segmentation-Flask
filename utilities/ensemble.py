import numpy as np


def average_ensemble(*args):
    y_hat_ensemble = [arg for arg in args]

    # Stack the predictions to create a new dimension
    y_hat_stack = np.stack(y_hat_ensemble, axis=-1)

    # Compute the ensemble prediction
    ensemble_pred = np.mean(y_hat_stack, axis=-1)

    return ensemble_pred


def weighted_avg_ensemble(*args):
    y_hat_ensemble = [arg for arg in args]

    # Stack the predictions to create a new dimension
    y_hat_stack = np.stack(y_hat_ensemble, axis=-1)

    # optimal weights
    optimal_weights = np.array([0.25, 0.25, 0.25, 0.25])

    # Apply optimal weights to compute the final ensemble prediction
    weights = optimal_weights.reshape(1, 1, 1, 1, -1)
    ensemble_pred = np.sum(y_hat_stack * weights, axis=-1)[0]

    return ensemble_pred


def majority_voting_ensemble(*args):
    y_hat_ensemble = [arg for arg in args]

    # Convert probabilities to binary masks
    thresh = 0.5
    binary_masks = [(pred > thresh).astype(np.uint8) for pred in y_hat_ensemble]

    # Stack the binary masks to create a new dimension
    binary_masks_stack = np.stack(binary_masks, axis=-1)

    # Compute majority vote
    ensemble_pred = np.sum(binary_masks_stack, axis=-1) > (len(y_hat_ensemble) / 2)

    return ensemble_pred


def maximum_probability_ensemble(*args):
    y_hat_ensemble = [arg for arg in args]

    # Stack the predictions to create a new dimension
    y_hat_stack = np.stack(y_hat_ensemble, axis=-1)

    # Compute maximum probability
    ensemble_pred = np.max(y_hat_stack, axis=-1)

    return ensemble_pred
