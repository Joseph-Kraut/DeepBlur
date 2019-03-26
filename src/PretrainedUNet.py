import tensorflow as tf
import tensorflow.contrib.slim as slim
import  numpy as np

class PretrainedUNet():
    """
    This class implements a UNet from https://bit.ly/2UAvptW
    """

    def __init__(self, pretrained=True, learning_rate=1e-3):
        """
        Sets up the model for training and inference
        :param pretrained:
        :param learning_rate:
        """
        # Make a call to the build_model
        self.sess = tf.Session()

        # Printout
        if pretrained:
            print("Building saved UNet model...")
        else:
            print("Building new UNet model...")

        self.build_model(pretrained, learning_rate)


    def build_model(self, pretrained, learning_rate):
        """
        Builds the model, either from scratch or from pretrained
        :param learning_rate: The learning rate for training
        :param pretrained: Whether or not to pull from pretrained models
        :return: None
        """
        raise NotImplementedError()

    def evaluate(self, inputs, ground_truth):
        """
        Evaluates the model on a set of inputs and outputs
        :param inputs: The input to feed through the network
        :param ground_truth: The output to compare to the network prediction
        :return: A loss value
        """
        raise NotImplementedError()

    def predict(self, inputs):
        """
        Outputs the result of the forward pass of inputs
        :param inputs: The input to forward pass through the network
        :return: The output of the forward pass
        """
        raise NotImplementedError()
