import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utilities.utility import get_model_weights


class UNet:
    def __init__(self, input_shape=(512, 512, 3), initial_channels=16, kernel_size=3,
                 num_classes=1, weights_path=get_model_weights('unet_scratch')):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.initial_channels = initial_channels
        self.model = self.build_model()

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

    def LeftBlock(self, channel, X, ksize=3, downsample=True):
        if downsample:
            X = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

        X = layers.Conv2D(channel, kernel_size=ksize, strides=1, padding="same")(X)
        X = layers.BatchNormalization()(X)
        X = layers.ReLU()(X)

        # Add more convolutional layers to increase model capacity
        for _ in range(3):
            X = layers.Conv2D(channel, kernel_size=ksize, strides=1, padding="same")(X)
            X = layers.BatchNormalization()(X)
            X = layers.ReLU()(X)

        return X

    def RightBlock(self, channel, X, ksize=3, X_skip=None, upsample=True):
        if upsample:
            X = layers.Conv2DTranspose(channel, kernel_size=3, strides=2, padding="same")(X)

        if X_skip is not None:
            X = layers.Concatenate()([X, X_skip])

        X = layers.Conv2D(channel, kernel_size=ksize, strides=1, padding="same")(X)
        X = layers.BatchNormalization()(X)
        X = layers.ReLU()(X)

        # Add more convolutional layers to increase model capacity
        for _ in range(3):
            X = layers.Conv2D(channel, kernel_size=ksize, strides=1, padding="same")(X)
            X = layers.BatchNormalization()(X)
            X = layers.ReLU()(X)

        return X

    def build_model(self):
        Input = layers.Input(shape=self.input_shape)

        X0 = layers.Rescaling(scale=1. / 127.5, offset=-1)(Input)

        KS = self.kernel_size
        channel = self.initial_channels

        X1 = self.LeftBlock(channel, X0, ksize=KS, downsample=False)  # 512

        channel *= 2
        X2 = self.LeftBlock(channel, X1, ksize=KS, downsample=True)  # 256

        channel *= 2
        X3 = self.LeftBlock(channel, X2, ksize=KS, downsample=True)  # 128

        channel *= 2
        X4 = self.LeftBlock(channel, X3, ksize=KS, downsample=True)  # 64

        XR = self.RightBlock(channel, X4, ksize=KS, X_skip=X3, upsample=True)  # 128

        channel = int(channel / 2)
        XR = self.RightBlock(channel, XR, ksize=KS, X_skip=X2, upsample=True)  # 256

        channel = int(channel / 2)
        XR = self.RightBlock(channel, XR, ksize=KS, X_skip=X1, upsample=True)  # 512

        channel = self.num_classes
        XR = layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")(XR)
        model = Model(inputs=Input, outputs=XR)

        return model

    def predict(self, image, threshold=0.5):
        # Ensure the input image has the right shape for prediction
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension if needed

        # Predict the probabilities for the input image
        prediction = self.model.predict(image)[0]
        prediction = tf.keras.activations.sigmoid(prediction).numpy()

        # Process predictions as needed
        if threshold:
            prediction = (prediction > threshold).astype(np.uint8)

        return prediction
