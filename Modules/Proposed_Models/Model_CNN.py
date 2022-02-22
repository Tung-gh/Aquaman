import sys
import tensorflow as tf
import numpy as np
from Modules.models import Model
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
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
        self.epoches = [10, 10, 10, 10, 16, 10, 20, 10]

        self.chi2_dict = []
        for i in range(self.num_aspects):
            path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/{}_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
            self.chi2_dict.append(load_chi2(path))

        # Create model
        model_input = layers.Input(shape=(self.text_len, 300))

        output_cnn_1 = layers.Conv1D(128, 1, activation='relu')(model_input)
        output_cnn_1 = layers.MaxPool1D(self.text_len-1)(output_cnn_1)
        output_cnn_1 = layers.Flatten()(output_cnn_1)

        output_cnn_2 = layers.Conv1D(128, 2, activation='relu')(model_input)
        output_cnn_2 = layers.MaxPool1D(self.text_len-2)(output_cnn_2)
        output_cnn_2 = layers.Flatten()(output_cnn_2)

        output_cnn_3 = layers.Conv1D(64, 3, activation='relu')(model_input)
        output_cnn_3 = layers.MaxPool1D(self.text_len-3)(output_cnn_3)
        output_cnn_3 = layers.Flatten()(output_cnn_3)

        output_cnn = tf.concat([output_cnn_1, output_cnn_2, output_cnn_3], axis=-1)

        output_mlp = layers.Dense(1024)(output_cnn)
        output_mlp = layers.Dropout(0.5)(output_mlp)
        output_mlp = layers.Dense(512)(output_mlp)
        output_mlp = layers.Dropout(0.2)(output_mlp)

        final_output = layers.Dense(2, activation='softmax')(output_mlp)

        model = tf.keras.models.Model(inputs=model_input, outputs=final_output)
        self.models = [model for _ in range(self.num_aspects)]

    def represent_fasttext(self, inputs):
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for ip in inputs:
            text = ip.split(' ')
            text_matrix = np.zeros((self.text_len, 300))
            for w, i in zip(text, range(len(text))):
                if w in self.fasttext:
                    text_matrix[i] = self.fasttext[w]
            for i in range(self.num_aspects):
                inputs_matrix[i].append(text_matrix)

        return np.array(inputs_matrix)

    def represent_fasttext_chi2_attention(self, inputs):
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for i in range(self.num_aspects):
            med = max(list(self.chi2_dict[i].values())) - min(list(self.chi2_dict[i].values()))
            for ip in inputs:
                text_matrix = np.zeros((self.text_len, 300))
                text = ip.split(' ')
                for w, j in zip(text, range(len(text))):
                    if w in self.fasttext:
                        if w in self.chi2_dict[i]:
                            text_matrix[i] = self.fasttext[w] * ((self.chi2_dict[i][w] - min(list(self.chi2_dict[i].values()))) / med)
                        else:
                            text_matrix[i] = self.fasttext[w]
                inputs_matrix[i].append(text_matrix)

        return np.array(inputs_matrix)

    def train(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        if self.embedding == 'fasttext':
            x = self.represent_fasttext(inputs)
        elif self.embedding == 'fasttext_chi2_attention':
            x = self.represent_fasttext_chi2_attention(inputs)
        y = [np.array(([output[i] for output in outputs]), dtype='float32') for i in range(self.num_aspects)]
        print()

        for i in range(self.num_aspects):
            print("Training aspect: {}".format(self.categories[i]))

            class_weight = compute_class_weight('balanced', classes=np.unique(y[i]), y=y[i])
            class_weight[1] = class_weight[1] * 5

            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2)

            self.models[i].compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.models[i].fit(x[i], y[i], epochs=self.epoches[i], batch_size=128, callbacks=[callback])
            # self.models[i].summary()

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        if self.embedding == 'fasttext':
            x = self.represent_fasttext(inputs)
        elif self.embedding == 'fasttext_chi2_attention':
            x = self.represent_fasttext_chi2_attention(inputs)
        outputs = []
        # for i in range(self.num_aspects):
        #     pred = self.models[i].predict(x[i])
        #     pred = np.argmax(pred, axis=-1)
        #     outputs.append(pred)

        predicts = [np.argmax(self.models[i].predict(x[i]), axis=-1) for i in range(self.num_aspects)]
        # predicts = [self.models[i].predict(x[i]) for i in range(self.num_aspects)]

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