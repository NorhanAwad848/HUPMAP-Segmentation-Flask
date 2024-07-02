import os
import numpy as np
from utilities.utility import get_model_weights

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


class UnetSegmentationModel:
    def __init__(self, backbone='efficientnetb5', weights_path=get_model_weights('unet')):
        self.BACKBONE = backbone
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        self.model = sm.Unet(self.BACKBONE, encoder_weights=None)

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

    def preprocess_image(self, image):
        preprocessed_image = self.preprocess_input(image)
        return preprocessed_image

    def predict(self, image, threshold=0.5):
        # Preprocess the input image
        preprocessed_image = self.preprocess_image(image)

        # Ensure the input image has the right shape for prediction
        if len(preprocessed_image.shape) == 3:
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension if needed

        # Predict the probabilities for the input image
        prediction = self.model.predict(preprocessed_image)[0]

        # Process predictions as needed
        if threshold:
            prediction = (prediction > threshold).astype(np.uint8)

        return prediction
