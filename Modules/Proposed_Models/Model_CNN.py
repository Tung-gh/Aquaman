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

        # Create model
        model_input = layers.Input(shape=(self.text_len, 300))
        output_cnn = layers.Conv1D(512, 1, activation='relu')(model_input)
        output_cnn = layers.MaxPool1D(self.text_len-1)(output_cnn)
        output_cnn = layers.Flatten()(output_cnn)

        output_mlp = layers.Dense(1024)(output_cnn)
        output_mlp = layers.Dense(512)(output_mlp)
        # output_mlp = layers.Dropout(0.5)(output_mlp)
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
            self.models[i].compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.models[i].fit(x[i], y[i], epochs=2, batch_size=128)
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