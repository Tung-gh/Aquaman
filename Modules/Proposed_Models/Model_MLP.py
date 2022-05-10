import sys
import tensorflow as tf
import numpy as np
from Modules.models import Model
from tensorflow.keras import layers
from Modules.preprocess import load_chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class ModelMLP(Model):
    def __init__(self, embedding):
        self.class_weight = None
        self.epochs = None
        self.threshold = None
        self.chi2_dict = []
        self.history = []
        self.models = []
        self.embedding = embedding
        if str(sys.argv[1])[:4] == "mebe":
            self.num_aspects = 6
            self.categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
            self.num = 0
        else:
            self.num_aspects = 8
            self.categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']
            self.num = 1

        self.vocab = []
        with open(r"H:\DS&KT Lab\\NCKH\\Aquaman\\data\\data_{}\\{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), encoding='utf8') as f:
            for line in f:
                self.vocab.append(line.strip())

    def represent_onehot(self, inputs):
        self.threshold = [
            [0.03, 0.01, 0.01, 0.02, 0.01, 0.5],
            []
        ]
        self.epochs = [
            [25, 25, 25, 25, 25, 25],   # [5, 5, 5, 5, 5, 5]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 3}, {0: 1, 1: 3}, {0: 1, 1: 3}],
            []
        ]

        features = [[] for i in range(self.num_aspects)]
        for ip in inputs:
            _features = [1 if w in ip else 0 for w in self.vocab]
            for i in range(self.num_aspects):
                features[i].append(_features)

        return np.array(features)

    def represent_onehot_chi2(self, inputs):
        self.threshold = [
            [0.1, 0.1, 0.2, 0.5, 0.08, 0.5],
            []
        ]
        self.epochs = [
            [25, 25, 25, 25, 25, 25],  # [6, 7, 10, 20, 10, 5]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 3}],
            []
        ]
        features = [[] for i in range(self.num_aspects)]
        for i in range(self.num_aspects):
            for ip in inputs:
                rep = np.zeros(len(self.chi2_dict[i]))
                text = ip.split(' ')
                for w in text:
                    if w in self.chi2_dict[i]:
                        rep[list(self.chi2_dict[i]).index(w)] = self.chi2_dict[i][w]
                features[i].append(np.array(rep))

        return np.array(features)

    def create_model(self):
        model = tf.keras.models.Sequential(
            [
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.Activation('tanh'),
                layers.Dropout(0.25),
                layers.Dense(128),
                layers.BatchNormalization(),
                layers.Activation('tanh'),
                # layers.Dense(1, activation='sigmoid')
                layers.Dense(2, activation='softmax')
            ]
        )
        return model

    def train(self, x_tr, x_val, y_tr, y_val):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        if self.embedding == 'onehot':
            xt = self.represent_onehot(x_tr)
            xv = self.represent_onehot(x_val)
        elif self.embedding == 'onehot_chi2':
            for i in range(self.num_aspects):
                path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/old/{}_fasttext_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
                self.chi2_dict.append(load_chi2(path))
            xt = self.represent_onehot_chi2(x_tr)
            xv = self.represent_onehot_chi2(x_val)

        yt = [np.array([output[i] for output in y_tr]) for i in range(self.num_aspects)]
        yv = [np.array([output[i] for output in y_val]) for i in range(self.num_aspects)]
        print()

        for i in range(self.num_aspects):
            model = self.create_model()
            self.models.append(model)
            print('Training aspect: {}'.format(self.categories[i]))

            es = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy', min_delta=0.005,
                                                  patience=3, restore_best_weights=True)
            reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_crossentropy', factor=0.2,
                                                            patience=2, min_delta=0.005, verbose=1)  # min_lr=0.0001,
            callbacks = [es, reducelr]  # val_binary_crossentropy val_sparse_categorical_crossentropy

            self.models[i].compile(loss='sparse_categorical_crossentropy',  # binary_crossentropy sparse_categorical_crossentropy
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                   metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(),    # BinaryCrossentropy SparseCategoricalCrossentropy
                                            'accuracy'
                                            ]
                                   )
            history = self.models[i].fit(xt[i], yt[i],
                                         epochs=self.epochs[self.num][i],
                                         batch_size=128,
                                         validation_data=(xv[i], yv[i]),
                                         callbacks=callbacks,
                                         class_weight=self.class_weight[self.num][i]
                                         )
            self.history.append(history)
            print('\n')
        return self.history

    def predict(self, x_te, y_te):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        if self.embedding == 'onehot':
            x = self.represent_onehot(x_te)
        elif self.embedding == 'onehot_chi2':
            x = self.represent_onehot_chi2(x_te)
        outputs = []
        predicts = []
        for i in range(self.num_aspects):
            # pred = self.models[i].predict(x[i]) > self.threshold[self.num][i]
            pred = np.argmax(self.models[i].predict(x[i]), axis=-1)
            _y_te = [y[i] for y in y_te]
            # print("Classification Report for aspect: {}".format(self.categories[i]))
            # print(classification_report(_y_te, list(pred)))
            print("Confusion Matrix for aspect: {}".format(self.categories[i]))
            print(confusion_matrix(_y_te, pred), '\n')
            predicts.append(pred)

        for ps in zip(*predicts):
            scores = list(ps)
            outputs.append(scores)

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
