import sys
import tensorflow as tf
import numpy as np
from Modules.models import Model
from Input_Output import Output
from tensorflow.keras import layers
from Modules.preprocess import load_chi2


class ModelMLP(Model):
    def __init__(self, embedding):
        if str(sys.argv[1])[:4] == "mebe":
            self.num_aspects = 6
            self.categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
        else:
            self.num_aspects = 8
            self.categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']
        self.embedding = embedding
        self.vocab = []
        with open(r"H:\DS&KT Lab\\NCKH\\Aquaman\\data\\data_{}\\{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), encoding='utf8') as f:
            for line in f:
                self.vocab.append(line.strip())
        self.chi2_dict = []
        for i in range(self.num_aspects):
            path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/{}_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
            self.chi2_dict.append(load_chi2(path))

        model = tf.keras.models.Sequential(
            [
                layers.Dense(256),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        self.models = [model for _ in range(self.num_aspects)]

    def represent_onehot(self, inputs):
        features = [[] for i in range(self.num_aspects)]
        for ip in inputs:
            _features = [1 if w in ip.text else 0 for w in self.vocab]
            for i in range(self.num_aspects):
                features[i].append(_features)

        return np.array(features)

    def represent_onehot_chi2(self, inputs):
        features = [[] for i in range(self.num_aspects)]
        for i in range(self.num_aspects):
            for ip in inputs:
                rep = np.zeros(len(self.chi2_dict[i]))
                text = ip.text.split(' ')
                for w in text:
                    if w in self.chi2_dict[i]:
                        rep[list(self.chi2_dict[i]).index(w)] = self.chi2_dict[i][w]
                features[i].append(np.array(rep))

        return np.array(features)

    def train(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        if self.embedding == 'onehot':
            x = self.represent_onehot(inputs)
        else:
            x = self.represent_onehot_chi2(inputs)
        ys = [np.array([output.scores[i] for output in outputs]) for i in range(self.num_aspects)]

        for i in range(self.num_aspects):
            self.models[i].compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            self.models[i].fit(x[i], ys[i], epochs=3, batch_size=128)
            # self.models[i].summary()

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        if self.embedding == 'onehot':
            x = self.represent_onehot(inputs)
        else:
            x = self.represent_onehot_chi2(inputs)
        outputs = []
        predicts = [self.models[i].predict_classes(x[i]) for i in range(self.num_aspects)]
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
