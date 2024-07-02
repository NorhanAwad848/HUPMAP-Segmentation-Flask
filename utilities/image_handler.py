import cv2
import numpy as np
from globals import HOST, PORT


def apply_threshold(nparr, thresh):
    return (nparr > thresh).astype(np.uint8)


def overlay_image(image, mask, opacity=0.8):
    if np.max(mask) == 0:
        return image.astype(np.uint8)  # Return the original image if the mask is blank (all zeros)

    mask_normalized = mask.astype(float) / np.max(mask)
    alpha = mask_normalized[:, :, 0] * opacity  # Extract the single channel from the mask & Adjust the opacity by
    # multiplying with a factor
    alpha = alpha[:, :, np.newaxis]  # Add a third dimension to make it compatible with the image
    result = alpha * mask + (1 - alpha) * image

    return result.astype(np.uint8)


def overlay_masks(y_true, y_hat):
    true_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    true_mask[:, :, 1] = y_true[:, :, 0] * 200

    pred_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_mask[:, :, 2] = y_hat[:, :, 0] * 230

    overlay_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay_mask[:, :, 1] = y_true[:, :, 0] * 200  # Green for true mask
    overlay_mask[:, :, 2] = y_hat[:, :, 0] * 230  # Red for predicted mask

    return true_mask, pred_mask, overlay_mask


def save_images(static_path, kidney_image, true_mask, predicted_mask):
    # Save the kidney slide image
    biomedical_image_path = static_path + "_image.png"
    cv2.imwrite(biomedical_image_path, cv2.cvtColor(kidney_image, cv2.COLOR_RGB2BGR))

    # Getting our images overlaid by masks ready
    overlaid_image_true = overlay_image(kidney_image, true_mask)
    overlaid_image_pred = overlay_image(kidney_image, predicted_mask)

    # Save the overlaid images by masks
    biomedical_overlaid_image_true_path = static_path + "_overlaid_image_true.png"
    biomedical_overlaid_image_pred_path = static_path + "_overlaid_image_pred.png"

    cv2.imwrite(biomedical_overlaid_image_true_path, cv2.cvtColor(overlaid_image_true, cv2.COLOR_RGB2BGR))
    cv2.imwrite(biomedical_overlaid_image_pred_path, cv2.cvtColor(overlaid_image_pred, cv2.COLOR_RGB2BGR))

    # Getting our RGB masks ready
    true_mask, predicted_mask, overlaid_mask = overlay_masks(true_mask, predicted_mask)

    # Save the true mask, predicted mask, and overlaid masks
    biomedical_true_mask_path = static_path + "_true_mask.png"
    biomedical_predicted_mask_path = static_path + "_predicted_mask.png"
    biomedical_overlaid_mask_path = static_path + "_overlaid_mask.png"

    cv2.imwrite(biomedical_true_mask_path, true_mask)
    cv2.imwrite(biomedical_predicted_mask_path, predicted_mask)
    cv2.imwrite(biomedical_overlaid_mask_path, overlaid_mask)

    return dict(image="http://{}:{}/{}".format(HOST, PORT, biomedical_image_path),
                true_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_true_mask_path),
                predicted_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_predicted_mask_path),
                overlaid_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_overlaid_mask_path),
                overlaid_image_true="http://{}:{}/{}".format(HOST, PORT, biomedical_overlaid_image_true_path),
                overlaid_image_pred="http://{}:{}/{}".format(HOST, PORT, biomedical_overlaid_image_pred_path))
