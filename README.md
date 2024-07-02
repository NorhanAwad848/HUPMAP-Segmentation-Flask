# HuBMAP - Hacking the Human Vasculature (BackEnd implementation using Flask)

## Description
The goal is to segment instances of microvascular structures, including capillaries, arterioles, and venules, to automate the segmentation of microvasculature structures to improve researchers' understanding of how the blood vessels are arranged in human tissues.

## APIs

### `/predict/unet`
- **Description:** API for predicting using the UNet model.
- **Input:**
  - `image`: Kidney tissue image of shape `512x512x3`.
  - `mask`: Corresponding mask of shape `512x512x1`. **Note:** The mask's pixel values must be 0 or 1 for proper functioning.
- **Output:**
  - **Paths:**
    - `image`: Path to the input image.
    - `overlaid_image_true`: Path to the image overlaid with the true mask.
    - `overlaid_image_pred`: Path to the image overlaid with the predicted mask.
    - `true_mask`: Path to the true mask.
    - `predicted_mask`: Path to the predicted mask.
    - `overlaid_mask`: Path to the overlaid mask.
  - **Score Metrics:**
    - `IOU`: Intersection over Union score indicating the model's confidence.
    - `Dice`: The dice coefficient score indicates the model's confidence.
  - **Threshold:**
    - `threshold`: The threshold at which the `predicted_mask` was binarized.

### `/predict/unet_scratch`
- **Description:** API for predicting using defined UNet model layers.
- **Input:**
  - `image`: Kidney tissue image of shape `512x512x3`.
  - `mask`: Corresponding mask of shape `512x512x1`. **Note:** The mask's pixel values must be 0 or 1 for proper functioning.
- **Output:**
  - **Paths:**
    - `image`: Path to the input image.
    - `overlaid_image_true`: Path to the image overlaid with the true mask.
    - `overlaid_image_pred`: Path to the image overlaid with the predicted mask.
    - `true_mask`: Path to the true mask.
    - `predicted_mask`: Path to the predicted mask.
    - `overlaid_mask`: Path to the overlaid mask.
  - **Score Metrics:**
    - `IOU`: Intersection over Union score indicating the model's confidence.
    - `Dice`: The dice coefficient score indicates the model's confidence.
  - **Threshold:**
    - `threshold`: The threshold at which the `predicted_mask` was binarized.
  
### `/predict/linknet`
- **Description:** API for predicting using LinkNet model.
- **Input:**
  - `image`: Kidney tissue image of shape `512x512x3`.
  - `mask`: Corresponding mask of shape `512x512x1`. **Note:** The mask's pixel values must be 0 or 1 for proper functioning.
- **Output:**
  - **Paths:**
    - `image`: Path to the input image.
    - `overlaid_image_true`: Path to the image overlaid with the true mask.
    - `overlaid_image_pred`: Path to the image overlaid with the predicted mask.
    - `true_mask`: Path to the true mask.
    - `predicted_mask`: Path to the predicted mask.
    - `overlaid_mask`: Path to the overlaid mask.
  - **Score Metrics:**
    - `IOU`: Intersection over Union score indicating the model's confidence.
    - `Dice`: The dice coefficient score indicates the model's confidence.
  - **Threshold:**
    - `threshold`: The threshold at which the `predicted_mask` was binarized.

### `/predict/fcn`
- **Description:** API for predicting using FCN model.
- **Input:**
  - `image`: Kidney tissue image of shape `512x512x3`.
  - `mask`: Corresponding mask of shape `512x512x1`. **Note:** The mask's pixel values must be 0 or 1 for proper functioning.
- **Output:**
  - **Paths:**
    - `image`: Path to the input image.
    - `overlaid_image_true`: Path to the image overlaid with the true mask.
    - `overlaid_image_pred`: Path to the image overlaid with the predicted mask.
    - `true_mask`: Path to the true mask.
    - `predicted_mask`: Path to the predicted mask.
    - `overlaid_mask`: Path to the overlaid mask.
  - **Score Metrics:**
    - `IOU`: Intersection over Union score indicating the model's confidence.
    - `Dice`: The dice coefficient score indicates the model's confidence.
  - **Threshold:**
    - `threshold`: The threshold at which the `predicted_mask` was binarized.

### `/predict/ensemble`
- **Description:** API for predicting using an ensemble of models.
- **Input:**
  - `image`: Kidney tissue image of shape `512x512x3`.
  - `mask`: Corresponding mask of shape `512x512x1`. **Note:** The mask's pixel values must be 0 or 1 for proper functioning.
- **Output:**
  - **Paths:**
    - `image`: Path to the input image.
    - `overlaid_image_true`: Path to the image overlaid with the true mask.
    - `overlaid_image_pred`: Path to the image overlaid with the predicted mask.
    - `true_mask`: Path to the true mask.
    - `predicted_mask`: Path to the predicted mask.
    - `overlaid_mask`: Path to the overlaid mask.
  - **Score Metrics:**
    - `IOU`: Intersection over Union score indicating the model's confidence.
    - `Dice`: The dice coefficient score indicates the model's confidence.
  - **Threshold:**
    - `threshold`: The threshold at which the `predicted_mask` was binarized.

## Installation
To install the project dependencies, run the following command:

```
pip install -r requirements.txt
```

This command will install all the necessary packages listed in the `requirements.txt` file.

### Additional Setup

To ensure the project runs smoothly, please follow these steps:

1. **Weights Directory**: Create a directory named `weights` within the model directory. This directory should contain the weights of the four models (UNet, UNet Scratch, LinkNet, FCN). These weights are essential for the proper functioning of the project. Adjust the model weights file name to match the one in the `model_weights.json` file.

    - **UNet Model Weights**: Download the UNet model weights from [here](https://www.kaggle.com/datasets/ahmedmaherelsaeidy/unet-weights).
    - **UNet Scratch Model Weights**: Download the LinkNet model weights from [here](https://www.kaggle.com/datasets/norhanawad/hupmap-unet-oversample).
    - **LinkNet Model Weights**: Download the LinkNet model weights from [here](https://www.kaggle.com/datasets/ahmedmaherelsaeidy/hupmap-models).
    - **FCN Model Weights**: Download the FCN model weights from [here](https://www.kaggle.com/datasets/norhanawad/hupmap-fcn-oversample).

2. **Images Directory in Static**: Create a directory named `images` within the `static` directory. This directory will contain the images generated by the server to display them later.

Make sure to create these directories and add the necessary files before running the project.

## UI
Here you can find the implemented [User Interface](https://github.com/NorhanAwad848/HUPMAP-Segmentation-ReactJS).

