import sys
import tensorflow as tf
import numpy as np
from Modules.models import Model
from Input_Output import Output
from tensorflow.keras import layers


class MLP_Model(Model):
    def __init__(self):
        self.num_aspects = 6 if str(sys.argv[1])[:4] == "mebe" else 8
        self.vocab = []
        with open(r"H:\DS&KT Lab\\NCKH\\Aquaman\\data\\data_mebe\\mebeshopee_vocab.txt", encoding='utf8') as f:
            for line in f:
                self.vocab.append(line.strip())

        self.models = [tf.keras.models.Sequential() for _ in range(self.num_aspects)]

    def represent(self, inputs):
        features = []
        for ip in inputs:
            _features = [1 if w in ip.text else 0 for w in self.vocab]
            features.append(_features)

        return np.array(features)

    def train(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        x = self.represent(inputs)
        ys = [np.array([output.scores[i] for output in outputs]) for i in range(self.num_aspects)]

        for i in range(self.num_aspects):
            self.models[i].add(layers.Dense(256, input_shape=(x.shape[1],)))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Activation('relu'))
            self.models[i].add(layers.Dense(256))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Activation('relu'))
            self.models[i].add(layers.Dense(1, activation='sigmoid'))

            self.models[i].compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            self.models[i].fit(x, ys[i], epochs=5, batch_size=128)
            # self.models[i].summary()

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        x = self.represent(inputs)

        outputs = []
        predicts = [self.models[i].predict_classes(x) for i in range(self.num_aspects)]
        for ps in zip(*predicts):
            aspects = list(range(self.num_aspects))
            scores = list(ps)
            outputs.append(Output(aspects, scores))

        return outputs

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
