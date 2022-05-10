import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

import sys
import pickle
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from Modules.models import Model
from tensorflow.keras import layers
from Modules.preprocess import load_chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from kmax import KMaxPooling



class ModelCNN(Model):
    def __init__(self, embedding, text_len):
        self.class_weight = None
        self.epochs = None
        self.threshold = None
        self.chi2_dict = None
        self.dim = None
        self.fasttext = None
        self.phoBert = None
        self.tokenizer = None
        self.rdrsegmenter = None
        self.history = []
        self.models = []
        self.embedding = embedding
        self.text_len = text_len

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
            [60, 60, 60, 60, 60, 60],    # [2, 2, 2, 2, 2, 2] [40, 40, 40, 40, 40, 40]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 3}],
            []
        ]
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for ip in inputs:
            text = ip.split(' ')
            text_matrix = np.zeros((self.text_len, self.dim), dtype='float32')
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
            [60, 60, 60, 60, 60, 60],    # [40, 40, 40, 40, 40, 40] [5, 5, 9, 8, 7, 5]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 3}],   # 1 5 5 5 5 2
            []
        ]
        self.chi2_dict = []
        for i in range(self.num_aspects):
            path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/old/{}_fasttext_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
            self.chi2_dict.append(load_chi2(path))
        inputs_matrix = [[] for _ in range(self.num_aspects)]
        for i in range(self.num_aspects):
            for ip in inputs:
                text_matrix = np.zeros((self.text_len, self.dim))
                text = ip.split(' ')
                for w, j in zip(text, range(len(text))):
                    if w in self.fasttext:
                        if w in self.chi2_dict[i]:
                            text_matrix[j] = self.fasttext[w] * self.chi2_dict[i][w]
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
            [60, 60, 60, 60, 60, 60],  # [2, 2, 2, 2, 2, 2] [40, 40, 40, 40, 40, 40]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 3}],
            []
        ]

        inputs_matrix = [[] for _ in range(self.num_aspects)]

        segmented_texts = []  # Texts which are segmented
        for inp in inputs:
            _inp = self.rdrsegmenter.tokenize(inp)
            for sen in _inp:
                segmented_texts.append(' '.join(sen))

        encoded_texts = []  # Texts are converted to indices vectors with 2 added token indices for <s>, </s>
        for st in segmented_texts:
            _st = self.tokenizer.encode(st)
            encoded_texts.append(_st)
        # maxdim = max([len(i) for i in encoded_texts])
        # print(maxdim)

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
                f = self.phoBert(tensors[i], masks[i])
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
            emb = np.zeros((self.text_len, self.dim))    # 75-2 self.text_len
            for i in range(len(te)):
                emb[i] = te[i]
            texts_matrix.append(emb)

        for i in range(self.num_aspects):
            inputs_matrix[i] = texts_matrix   # texts_matrix inputs
            inputs_matrix[i] = np.array(inputs_matrix[i])

        return np.array(inputs_matrix)

    def represent_phobert_chi2(self, inputs):
        self.threshold = [
            [0.75, 0.7, 0.4, 0.5, 0.4, 0.45],  # [0.5, 0.45, 0.45, 0.01, 0.45, 0.45]
            []
        ]
        self.epochs = [
            [60, 60, 60, 60, 60, 60],  # [2, 2, 2, 2, 2, 2] [40, 40, 40, 40, 40, 40]
            []
        ]
        self.class_weight = [
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 3}, {0: 1, 1: 3}],
            []
        ]

        inputs_matrix = [[] for _ in range(self.num_aspects)]
        self.chi2_dict = []
        for i in range(self.num_aspects):
            path = r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/old/{}_phobert_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), self.categories[i])
            self.chi2_dict.append(load_chi2(path))

        segmented_texts = []  # Texts which are segmented
        for i in range(len(inputs)):
            _inp = self.rdrsegmenter.tokenize(inputs[i])
            for sen in _inp:
                segmented_texts.append(' '.join(sen))
            assert len(segmented_texts) == i + 1

        encoded_texts = []  # Texts are converted to indices vectors with 2 added token indices for <s>, </s>
        for st in segmented_texts:
            _st = self.tokenizer.encode(st)
            encoded_texts.append(_st)

        masked_pos = []
        for mp in encoded_texts:
            m = [int(token_id > 0) for token_id in mp]
            masked_pos.append(m)

        tensors, masks = [], []  # Convert list of indices to torch tensor
        for i in range(len(masked_pos)):
            tensors.append(torch.tensor([encoded_texts[i]]))
            masks.append(torch.tensor([masked_pos[i]]))

        lhs = []  # There are 13 tensors of 13 attention layers from PhoBERT <=> 1 word has 13 (768,)-tensor
        for i in range(len(tensors)):
            with torch.no_grad():
                f = self.phoBert(tensors[i], masks[i])
                hs = f[2]  # Len: 13 as 13 output tensors from 13 attention layers
                _hs = np.squeeze(np.array([x.detach().numpy() for x in hs]), axis=1)  # Reduce the dimension
                lhs.append(_hs)

        reshaped_lhs = []  # Shape: num_words * 13 * 768
        for rlhs in lhs:
            _rlhs = []
            for i in range(rlhs.shape[1]):
                a = np.array([x[i] for x in rlhs])
                _rlhs.append(a)
            reshaped_lhs.append(_rlhs)

        texts_token_emb = []  # Shape: num_words * 768
        for tte in reshaped_lhs:
            _tte = []
            for i in tte:
                emb = tf.reduce_sum(i[-4:], axis=0)
                _tte.append(emb)
            texts_token_emb.append(np.array(_tte[1:-1]))
        print()

        for i in range(self.num_aspects):
            for j in range(len(segmented_texts)):
                te = self.tokenizer.tokenize(segmented_texts[j])
                text_matrix = np.zeros((self.text_len, self.dim))    # 75-2 self.text_len
                for h in range(len(te)):
                    if te[h] in self.chi2_dict[i]:
                        text_matrix[h] = texts_token_emb[j][h] * self.chi2_dict[i][te[h]]
                    else:
                        text_matrix[h] = texts_token_emb[j][h]
                inputs_matrix[i].append(text_matrix)

        return np.array(inputs_matrix)

    def create_model(self):
        model_input = layers.Input(shape=(self.text_len, self.dim))  # 75-2 self.text_len

        """
        PAPER'S METHOD 
        """
        # output_cnn = layers.Conv1D(300, 5, strides=1)(model_input)
        #
        # block1 = layers.Conv1D(128, 7, strides=1)(output_cnn)
        # block1 = layers.BatchNormalization()(block1)
        # block1 = layers.Activation('relu')(block1)
        #
        # block2 = layers.Conv1D(256, 7, strides=1)(block1)
        # block2 = layers.BatchNormalization()(block2)
        # block2 = layers.Activation('relu')(block2)
        #
        # kmax = KMaxPooling()(block2)
        #
        # dense1 = layers.Dense(128)(kmax)
        # act1 = layers.Activation('relu')(dense1)
        # drop1 = layers.Dropout(0.2)(act1)
        #
        # final_output = layers.Dense(2, activation='softmax')(drop1)

        """
        CNN MODEL
        """
        output_cnn_1 = layers.Conv1D(512, 1)(model_input)    # 512
        output_cnn_1 = layers.Activation('tanh')(output_cnn_1)
        output_cnn_1 = layers.MaxPool1D(self.text_len)(output_cnn_1)  # 75-2 self.text_len
        output_cnn_1 = layers.Flatten()(output_cnn_1)

        # output_cnn_2 = layers.Conv1D(512, 2)(model_input)
        # output_cnn_2 = layers.Activation('tanh')(output_cnn_2)
        # output_cnn_2 = layers.MaxPool1D(self.text_len)(output_cnn_2) # 75-2 self.text_len
        # output_cnn_2 = layers.Flatten()(output_cnn_2)

        # output_cnn_3 = layers.Conv1D(64, 3)(model_input)
        # output_cnn_3 = layers.Activation('tanh')(output_cnn_3)
        # output_cnn_3 = layers.MaxPool1D(self.text_len)(output_cnn_3)   # 75-2 self.text_len
        # output_cnn_3 = layers.Flatten()(output_cnn_3)
        #
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2, output_cnn_3], axis=-1)
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2], axis=-1)

        output_mlp = layers.Dense(256)(output_cnn_1)   # 256    # output_cnn output_cnn_1
        output_mlp = layers.BatchNormalization()(output_mlp)
        output_mlp = layers.Activation('relu')(output_mlp)
        output_mlp = layers.Dropout(0.1)(output_mlp)       # 0.25
        # output_mlp = layers.Dense(32)(output_mlp)
        # output_mlp = layers.BatchNormalization()(output_mlp)
        # output_mlp = layers.Activation('relu')(output_mlp)
        # output_mlp = layers.Dropout(0.25)(output_mlp)

        # final_output = layers.Dense(1, activation='sigmoid')(output_mlp)
        final_output = layers.Dense(2, activation='softmax')(output_mlp)

        model = tf.keras.models.Model(inputs=model_input, outputs=final_output)    # model_input m_input
        return model

    def create_model_attention(self):
        model_input = layers.Input(shape=(self.text_len, self.dim))  # 75-2 self.text_len
        h = layers.Dense(units=256, activation='tanh')(model_input)
        a = layers.Dense(units=1)(h)
        a = layers.Flatten()(a)
        s = tf.math.sigmoid(a)
        model_input = model_input * tf.expand_dims(s, axis=-1)
        model_input = layers.Input(tensor=model_input)

        output_cnn_1 = layers.Conv1D(512, 1)(model_input)  # 512
        output_cnn_1 = layers.Activation('tanh')(output_cnn_1)
        output_cnn_1 = layers.MaxPool1D(self.text_len)(output_cnn_1) # 75-2 self.text_len
        output_cnn_1 = layers.Flatten()(output_cnn_1)

        # output_cnn_2 = layers.Conv1D(512, 2)(model_input)
        # output_cnn_2 = layers.Activation('tanh')(output_cnn_2)
        # output_cnn_2 = layers.MaxPool1D(self.text_len)(output_cnn_2)   # 75-2 self.text_len
        # output_cnn_2 = layers.Flatten()(output_cnn_2)

        # output_cnn_3 = layers.Conv1D(64, 3)(model_input)
        # output_cnn_3 = layers.Activation('tanh')(output_cnn_3)
        # output_cnn_3 = layers.MaxPool1D(self.text_len)(output_cnn_3)   # 75-2 self.text_len
        # output_cnn_3 = layers.Flatten()(output_cnn_3)
        #
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2, output_cnn_3], axis=-1)
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2], axis=-1)

        output_mlp = layers.Dense(256)(output_cnn_1)    # 256   # output_cnn output_cnn_1
        output_mlp = layers.BatchNormalization()(output_mlp)
        output_mlp = layers.Activation('relu')(output_mlp)
        output_mlp = layers.Dropout(0.1)(output_mlp)    # 0.25
        # output_mlp = layers.Dense(32)(output_mlp)
        # output_mlp = layers.BatchNormalization()(output_mlp)
        # output_mlp = layers.Activation('relu')(output_mlp)
        # output_mlp = layers.Dropout(0.25)(output_mlp)

        # final_output = layers.Dense(1, activation='sigmoid')(output_mlp)
        final_output = layers.Dense(2, activation='softmax')(output_mlp)

        model = tf.keras.models.Model(inputs=model_input, outputs=final_output)
        return model

    def create_model_mca(self):
        model_input = layers.Input(shape=(self.text_len, self.dim))  # 75-2 self.text_len
        z = np.zeros([1, self.dim], dtype='float32')
        mi1 = tf.concat([z[np.newaxis, :, :], z[np.newaxis, :, :], model_input], axis=1)
        mi2 = tf.concat([model_input, z[np.newaxis, :, :], z[np.newaxis, :, :]], axis=1)
        mi2 = K.reverse(mi2, axes=1)
        lstm1 = layers.RNN(layers.LSTMCell(128, input_shape=(2, self.dim)), return_sequences=True)
        lstm2 = layers.RNN(layers.LSTMCell(128, input_shape=(2, self.dim)), return_sequences=True)
        prefix_context = lstm1(mi1)
        suffix_context = lstm2(mi2)
        prefix_context = prefix_context[:, 1:-1, :]
        suffix_context = suffix_context[:, 1:-1, :]
        micro_context = tf.concat([prefix_context, suffix_context, model_input], axis=-1)
        h = layers.Dense(units=256, activation='tanh')(micro_context)
        a = layers.Dense(units=1, activation='tanh')(h)
        a = layers.Flatten()(a)
        s = tf.math.sigmoid(a)
        model_input = model_input * tf.expand_dims(s, axis=-1)
        model_input = layers.Input(tensor=model_input)

        """
        CNN MODEL
        """
        output_cnn_1 = layers.Conv1D(512, 1)(model_input)  # 512
        output_cnn_1 = layers.Activation('tanh')(output_cnn_1)
        output_cnn_1 = layers.MaxPool1D(self.text_len)(output_cnn_1)  # 75-2 self.text_len
        output_cnn_1 = layers.Flatten()(output_cnn_1)

        # output_cnn_2 = layers.Conv1D(512, 2)(model_input)
        # output_cnn_2 = layers.Activation('tanh')(output_cnn_2)
        # output_cnn_2 = layers.MaxPool1D(75-2 - 1)(output_cnn_2) # 75-2 self.text_len
        # output_cnn_2 = layers.Flatten()(output_cnn_2)

        # output_cnn_3 = layers.Conv1D(64, 3)(model_input)
        # output_cnn_3 = layers.Activation('tanh')(output_cnn_3)
        # output_cnn_3 = layers.MaxPool1D(self.text_len)(output_cnn_3)   # 75-2 self.text_len
        # output_cnn_3 = layers.Flatten()(output_cnn_3)
        #
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2, output_cnn_3], axis=-1)
        # output_cnn = tf.concat([output_cnn_1, output_cnn_2], axis=-1)

        output_mlp = layers.Dense(256)(output_cnn_1)  # 256    # output_cnn output_cnn_1
        output_mlp = layers.BatchNormalization()(output_mlp)
        output_mlp = layers.Activation('relu')(output_mlp)
        output_mlp = layers.Dropout(0.1)(output_mlp)  # 0.25
        # output_mlp = layers.Dense(32)(output_mlp)
        # output_mlp = layers.BatchNormalization()(output_mlp)
        # output_mlp = layers.Activation('relu')(output_mlp)
        # output_mlp = layers.Dropout(0.25)(output_mlp)

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
        if self.embedding in ['fasttext', 'fasttext_attention', 'fasttext_mca']:
            with open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\fasttext.pkl", 'rb') as fasttext_emb:
                self.fasttext = pickle.load(fasttext_emb)
            self.dim = 300
            xt = self.represent_fasttext(x_tr)
            xv = self.represent_fasttext(x_val)

        elif self.embedding in ['fasttext_chi2', 'fasttext_chi2_attention', 'fasttext_chi2_mca']:
            with open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\fasttext.pkl", 'rb') as fasttext_emb:
                self.fasttext = pickle.load(fasttext_emb)
            self.dim = 300
            xt = self.represent_fasttext_chi2(x_tr)
            xv = self.represent_fasttext_chi2(x_val)

        elif self.embedding in ['PhoBERT', 'PhoBERT_attention', 'PhoBERT_mca']:
            # Call for PhoBERT pretrained model from huggingface-transformers
            self.phoBert = AutoModel.from_pretrained('vinai/phobert-base', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)

            # Load and use RDRSegmenter from VnCoreNLP as recommended by PhoBERT authors
            self.rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar',
                                     annotators='wseg', max_heap_size='-Xmx500m')
            self.dim = 768
            xt = self.represent_phobert(x_tr)
            xv = self.represent_phobert(x_val)

        elif self.embedding in ['PhoBERT_chi2', 'PhoBERT_chi2_attention', 'PhoBERT_chi2_mca']:
            # Call for PhoBERT pretrained model from huggingface-transformers
            self.phoBert = AutoModel.from_pretrained('vinai/phobert-base', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)

            # Load and use RDRSegmenter from VnCoreNLP as recommended by PhoBERT authors
            self.rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar',
                                          annotators='wseg', max_heap_size='-Xmx500m')
            self.dim = 768
            xt = self.represent_phobert_chi2(x_tr)
            xv = self.represent_phobert_chi2(x_val)

        yt = [np.array(([output[i] for output in y_tr]), dtype='int32') for i in range(self.num_aspects)]
        yv = [np.array(([output[i] for output in y_val]), dtype='int32') for i in range(self.num_aspects)]
        print()

        for i in range(self.num_aspects):
            # Create model
            if 'attention' in self.embedding:
                model = self.create_model_attention()
            elif 'mca' in self.embedding:
                model = self.create_model_mca()
            else:
                model = self.create_model()
            self.models.append(model)

            es = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy', min_delta=0.005,
                                                  patience=3, restore_best_weights=True)
            reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_crossentropy', factor=0.2,
                                                            patience=2, min_delta=0.003, verbose=1)  # min_lr=0.0001,
            callbacks = [es, reducelr]    # val_binary_crossentropy val_sparse_categorical_crossentropy

            print("Training aspect: {}".format(self.categories[i]))
            print(self.models[i].weights[-1])
            self.models[i].compile(loss='sparse_categorical_crossentropy',  # binary_crossentropy sparse_categorical_crossentropy
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                   metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(),   # BinaryCrossentropy SparseCategoricalCrossentropy
                                            'accuracy']
                                   )
            history = self.models[i].fit(xt[i], yt[i],
                                         epochs=self.epochs[self.num][i],
                                         batch_size=64,
                                         validation_data=(xv[i], yv[i]),
                                         callbacks=callbacks,    # es callbacks reducelr
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
        if self.embedding in ['fasttext', 'fasttext_attention', 'fasttext_mca']:
            x = self.represent_fasttext(x_te)
        elif self.embedding in ['fasttext_chi2', 'fasttext_chi2_attention', 'fasttext_chi2_mca']:
            x = self.represent_fasttext_chi2(x_te)
        elif self.embedding in ['PhoBERT', 'PhoBERT_attention', 'PhoBERT_mca']:
            x = self.represent_phobert(x_te)
        elif self.embedding in ['PhoBERT_chi2', 'PhoBERT_chi2_attention', 'PhoBERT_chi2_mca']:
            x = self.represent_phobert_chi2(x_te)
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