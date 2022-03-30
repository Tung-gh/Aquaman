import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

import sys
import pickle
import tensorflow as tf
import numpy as np

from Modules.models import Model
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from Modules.preprocess import load_chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class ModelCNN(Model):
    def __init__(self, embedding, text_len):
        self.class_weight = None
        self.epochs = None
        self.threshold = None
        self.chi2_dict = None
        self.dim = None
        self.history = []
        self.embedding = embedding
        self.text_len = text_len
        self.models = []

        with open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\fasttext.pkl", 'rb') as fasttext_emb:
            self.fasttext = pickle.load(fasttext_emb)

        if str(sys.argv[1])[:4] == "mebe":
            self.num_aspects = 6
            self.categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
            self.num = 0
        else:
            self.num_aspects = 8
            self.categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']
            self.num = 1

    def represent_fasttext(self, inputs):
        self.threshold = [
            [0.75, 0.6, 0.6, 0.75, 0.6, 0.5],    # [0.5, 0.45, 0.45, 0.01, 0.45, 0.45]
            []
        ]
        self.epochs = [
            [25, 25, 25, 25, 25, 25],    # [25, 25, 25, 25, 25, 25] [1, 1, 1, 1, 1, 1]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 3}],
            []
        ]
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for ip in inputs:
            text = ip.split(' ')
            text_matrix = np.zeros((self.text_len, self.dim))
            for w, i in zip(text, range(len(text))):
                if w in self.fasttext:
                    text_matrix[i] = self.fasttext[w]
            for j in range(self.num_aspects):
                inputs_matrix[j].append(text_matrix)

        return np.array(inputs_matrix)

    def represent_fasttext_chi2(self, inputs):
        self.threshold = [
            [0.75, 0.6, 0.6, 0.75, 0.75, 0.75],    # [0.5, 0.45, 0.45, 0.01, 0.45, 0.45]
            []
        ]
        self.epochs = [
            [20, 20, 20, 20, 20, 20],    # [10, 10, 10, 10, 10, 10] [5, 5, 9, 8, 7, 5]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 1}],
            []
        ]
        self.chi2_dict = []
        for i in range(self.num_aspects):
            path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/{}_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
            self.chi2_dict.append(load_chi2(path))
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for i in range(self.num_aspects):
            max_chi2_score = max(list(self.chi2_dict[i].values()))
            min_chi2_score = min(list(self.chi2_dict[i].values()))
            med = max_chi2_score - min_chi2_score
            for ip in inputs:
                text_matrix = np.zeros((self.text_len, self.dim))
                text = ip.split(' ')
                for w, j in zip(text, range(len(text))):
                    if w in self.fasttext:
                        if w in self.chi2_dict[i]:
                            text_matrix[j] = self.fasttext[w] * ((self.chi2_dict[i][w] - min_chi2_score)/med)
                        else:
                            text_matrix[j] = self.fasttext[w]
                inputs_matrix[i].append(text_matrix)

        return np.array(inputs_matrix)

    def represent_phobert(self, inputs):
        self.threshold = [
            [0.75, 0.7, 0.4, 0.5, 0.4, 0.45],  # [0.5, 0.45, 0.45, 0.01, 0.45, 0.45]
            []
        ]
        self.epochs = [
            [15, 15, 15, 15, 15, 15],  # [10, 10, 10, 10, 10, 10] [2, 2, 2, 2, 2, 2]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 7}, {0: 1, 1: 7}, {0: 1, 1: 7}, {0: 1, 1: 3}],
            []
        ]

        inputs_matrix = [[] for _ in range(self.num_aspects)]

        # Call for PhoBERT pretrained model from huggingface-transformers
        phoBert = AutoModel.from_pretrained('vinai/phobert-base', output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)

        # Load and use RDRSegmenter from VnCoreNLP as recommended by PhoBERT authors
        rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar',
                                 annotators='wseg', max_heap_size='-Xmx500m')

        segmented_texts = []  # Texts which are segmented
        # for inp in inputs[:10]:
        for inp in inputs:
            _inp = rdrsegmenter.tokenize(inp)
            for sen in _inp:
                segmented_texts.append(' '.join(sen))

        encoded_texts = []  # Texts are converted to indices vectors with 2 added token indices for <s>, </s>
        for st in segmented_texts:
            _st = tokenizer.encode(st)
            encoded_texts.append(_st)

        masked_pos = []
        for mp in encoded_texts:
            m = [int(token_id > 0) for token_id in mp]
            masked_pos.append(m)

        tensors, masks = [], [] # Convert list of indices to torch tensor
        for i in range(len(masked_pos)):
            tensors.append(torch.tensor([encoded_texts[i]]))
            masks.append(torch.tensor([masked_pos[i]]))

        lhs = []  # There are 13 tensors of 13 attention layers from PhoBERT <=> 1 word has 13 (768,)-tensor
        for i in range(len(tensors)):
            with torch.no_grad():
                f = phoBert(tensors[i], masks[i])
                hs = f[2]  # Len: 13 as 13 output tensors from 13 attention layers
                _hs = np.squeeze(np.array([x.detach().numpy() for x in hs]), axis=1)    # Reduce the dimension
                lhs.append(_hs)

        reshaped_lhs = []   # Shape: num_words * 13 * 768
        for rlhs in lhs:
            _rlhs = []
            for i in range(rlhs.shape[1]):
                a = np.array([x[i] for x in rlhs])
                _rlhs.append(a)
            reshaped_lhs.append(_rlhs)

        texts_token_emb = []    # Shape: num_words * 768
        for tte in reshaped_lhs:
            _tte = []
            for i in tte:
                emb = tf.reduce_sum(i[-4:], axis=0)
                _tte.append(emb)
            texts_token_emb.append(np.array(_tte[1:-1]))

        texts_matrix = []
        for te in texts_token_emb:
            emb = np.zeros((self.text_len, self.dim))
            for i in range(len(te)):
                emb[i] = te[i]
            texts_matrix.append(emb)

        for i in range(self.num_aspects):
            inputs_matrix[i] = texts_matrix
            inputs_matrix[i] = np.array(inputs_matrix[i])

        return np.array(inputs_matrix)

    def create_model(self):
        model_input = layers.Input(shape=(self.text_len, self.dim))
        # h = layers.Dense(units=1024, activation='relu')(model_input)
        # h = layers.Dense(units=512, activation='relu')(h)
        # a = layers.Dense(units=1, activation='relu')(h)
        # a = layers.Flatten()(a)
        # s = tf.math.sigmoid(a)
        # model_input = model_input * tf.expand_dims(s, axis=-1)
        # model_input = tf.keras.layers.Input(tensor=model_input)

        output_cnn_1 = layers.Conv1D(512, 1, activation='tanh')(model_input)
        output_cnn_1 = layers.MaxPool1D(self.text_len - 1)(output_cnn_1)
        output_cnn_1 = layers.Flatten()(output_cnn_1)

        output_cnn_2 = layers.Conv1D(512, 2, activation='tanh')(model_input)
        output_cnn_2 = layers.MaxPool1D(self.text_len - 2)(output_cnn_2)
        output_cnn_2 = layers.Flatten()(output_cnn_2)

        output_cnn_3 = layers.Conv1D(64, 3, activation='tanh')(model_input)
        output_cnn_3 = layers.MaxPool1D(self.text_len-3)(output_cnn_3)
        output_cnn_3 = layers.Flatten()(output_cnn_3)

        output_cnn = tf.concat([output_cnn_1, output_cnn_2, output_cnn_3], axis=-1)
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2], axis=-1)

        output_mlp = layers.Dense(512)(output_cnn)
        output_mlp = layers.BatchNormalization()(output_mlp)
        output_mlp = layers.Activation('tanh')(output_mlp)
        # output_mlp = layers.Dropout(0.25)(output_mlp)
        output_mlp = layers.Dense(256)(output_mlp)
        output_mlp = layers.BatchNormalization()(output_mlp)
        output_mlp = layers.Activation('tanh')(output_mlp)
        # output_mlp = layers.Dropout(0.2)(output_mlp)

        # final_output = layers.Dense(1, activation='sigmoid')(output_mlp)
        final_output = layers.Dense(2, activation='softmax')(output_mlp)

        model = tf.keras.models.Model(inputs=model_input, outputs=final_output)
        return model

    def train(self, x_tr, x_val, y_tr, y_val):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        if self.embedding == 'fasttext':
            self.dim = 300
            xt = self.represent_fasttext(x_tr)
            xv = self.represent_fasttext(x_val)
        elif self.embedding == 'fasttext_chi2':
            self.dim = 300
            xt = self.represent_fasttext_chi2(x_tr)
            xv = self.represent_fasttext_chi2(x_val)
        elif self.embedding == 'PhoBERT':
            self.dim = 768
            xt = self.represent_phobert(x_tr)
            xv = self.represent_phobert(x_val)
        yt = [np.array(([output[i] for output in y_tr]), dtype='int32') for i in range(self.num_aspects)]
        yv = [np.array(([output[i] for output in y_val]), dtype='int32') for i in range(self.num_aspects)]
        print()

        for i in range(self.num_aspects):
            # Create model
            model = self.create_model()
            self.models.append(model)

            print("Training aspect: {}".format(self.categories[i]))

            es = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy', min_delta=0.005,
                                                  patience=3, restore_best_weights=True)
            reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_crossentropy', factor=0.2,
                                                            patience=2, min_delta=0.005, verbose=1)  # min_lr=0.0001,
            callbacks = [es, reducelr]    # val_binary_crossentropy val_sparse_categorical_crossentropy

            print(self.models[i].weights[-1])

            self.models[i].compile(loss='sparse_categorical_crossentropy',  # binary_crossentropy sparse_categorical_crossentropy
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                   metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(),   # BinaryCrossentropy SparseCategoricalCrossentropy
                                            'accuracy'
                                            ]
                                   )
            history = self.models[i].fit(xt[i], yt[i],
                                         epochs=self.epochs[self.num][i],
                                         batch_size=128,
                                         validation_data=(xv[i], yv[i]),
                                         callbacks=callbacks,    # es callbacks
                                         class_weight=self.class_weight[self.num][i]
                                         )
            self.history.append(history)
            print()
        return self.history

    def predict(self, x_te, y_te):
        """
        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        if self.embedding == 'fasttext':
            x = self.represent_fasttext(x_te)
        elif self.embedding == 'fasttext_chi2':
            x = self.represent_fasttext_chi2(x_te)
        elif self.embedding == 'PhoBERT':
            x = self.represent_phobert(x_te)
        outputs = []
        predicts = []
        for i in range(self.num_aspects):
            # pred = (self.models[i].predict(x[i]) > self.threshold[self.num][i]).astype('int32')
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