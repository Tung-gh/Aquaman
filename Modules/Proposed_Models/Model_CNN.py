import sys
import tensorflow as tf
import numpy as np
from Modules.models import Model
from Input_Output import Output
from tensorflow.keras import layers
from Modules.preprocess import load_chi2


class ModelCNN(Model):
    def __init__(self, embedding, text_len, pretrained):
        if str(sys.argv[1])[:4] == "mebe":
            self.num_aspects = 6
            self.categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
        else:
            self.num_aspects = 8
            self.categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']
        self.embedding = embedding
        self.text_len = text_len
        self.fasttext = pretrained

        # input = layers.Input(shape=(self.text_len, 300))

        model = tf.keras.models.Sequential(
            [
                layers.Dense(256),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        self.models = [model for _ in range(self.num_aspects)]

    def represent_fasttext(self, inputs, outputs):
        inputs_matrix, outputs_matrix = [], []
        _pos, pos = 0, []
        for ip, op in zip(inputs, outputs):
            if len(ip.text.split(' ')) <= self.text_len:
                text = ip.text.split(' ')
                sen_matrix = np.zeros((self.text_len, 300))
                for w, i in zip(text, range(len(text))):
                    if w in self.fasttext:
                        sen_matrix[i] = self.fasttext[w]
                inputs_matrix.append(sen_matrix)
                pos.append(_pos)
            _pos = _pos + 1

        for i in range(self.num_aspects):
            score = []
            for j in pos:
                score.append(outputs[j].scores[i])
            outputs_matrix.append(score)

        return np.array(inputs_matrix), np.array(outputs_matrix)

    def train(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        if self.embedding == 'fasttext':
            x, y = self.represent_fasttext(inputs, outputs)
        elif self.embedding == 'fasttext_chi2_attention':
            x, y = self.represent_fasttext_chi2_attention(inputs)
        # ys = [np.array(([output.scores[i] for output in outputs]), dtype='float32') for i in range(self.num_aspects)]
        print()

        for i in range(self.num_aspects):
            self.models[i].compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.models[i].fit(x, y[i], epochs=3, batch_size=128)
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