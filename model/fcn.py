import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utilities.utility import get_model_weights


class FCN:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, weights_path=get_model_weights('fcn')):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

    def conv_block(self, channel, X, ksize=3, downsample=True):
        if downsample:
            X = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

        X = layers.Conv2D(channel, kernel_size=ksize, strides=1, padding="same")(X)
        X = layers.BatchNormalization()(X)
        X = layers.ReLU()(X)

        return X

    def upsample_block(self, channel, X, ksize, stride):
        X = layers.Conv2DTranspose(channel, kernel_size=ksize, strides=stride, padding="same")(X)
        return X

    def build_model(self):
        Input = layers.Input(shape=self.input_shape)
        X = layers.Rescaling(scale=1. / 127.5, offset=-1)(Input)

        KS = 3
        channel = 64
        X = self.conv_block(channel, X, ksize=KS, downsample=False)  # 512
        X = self.conv_block(channel, X, ksize=KS, downsample=False)
        X = self.conv_block(channel, X, ksize=KS, downsample=False)

        channel *= 2
        X = self.conv_block(channel, X, ksize=KS, downsample=True)  # 256
        X = self.conv_block(channel, X, ksize=KS, downsample=False)
        X = self.conv_block(channel, X, ksize=KS, downsample=False)

        channel *= 2
        X = self.conv_block(channel, X, ksize=KS, downsample=True)  # 128
        X = self.conv_block(channel, X, ksize=KS, downsample=False)
        X = self.conv_block(channel, X, ksize=KS, downsample=False)

        channel *= 2
        X = self.conv_block(channel, X, ksize=KS, downsample=True)  # 64
        X = self.conv_block(channel, X, ksize=KS, downsample=False)
        X = self.conv_block(channel, X, ksize=KS, downsample=False)

        X = self.upsample_block(self.num_classes, X, ksize=8, stride=8)
        model = Model(inputs=Input, outputs=X)

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
