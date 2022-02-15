import sys
import tensorflow as tf

from Modules.Model import Model
from tensorflow import keras


class MLP_Model(Model):
    def __init__(self):
        self.num_aspects = 6 if str(sys.argv[1])[:4] == "mebe" else 8
        self.models = [tf.keras.models.Sequential() for _ in range(self.num_aspects)]

    def train(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        return NotImplementedError

    def save(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def load(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        raise NotImplementedError
