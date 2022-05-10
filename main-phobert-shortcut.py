import torch
import sys
import re
import numpy as np
import pickle
from Modules.Proposed_Models.Model_CNN import ModelCNN
from sklearn.model_selection import train_test_split
from Modules.evaluation import cal_aspect_prf

from test import nload_data, npreprocess_inputs

datasets = {'mebeshopee': [6, 0],
            'mebetiki': [6, 1],
            'techshopee': [8, 2],
            'techtiki': [8, 3]
            }
train_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_train.csv",
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebetiki_train.csv"
    ]
test_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_test.csv",
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebetiki_test.csv"
]
text_len = [60, ]     # 40, 38, 55, 52

if __name__ == '__main__':
    argv = sys.argv[1]
    train_path = train_paths[datasets[argv][1]]
    test_path = test_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file   # 2086 samples, 2086 samples
    tr_inputs, tr_outputs = nload_data(train_path, num_aspects)     # 2403
    te_inputs, te_outputs = nload_data(test_path, num_aspects)      # 603
    print()

    # Preprocess the inputs data: Remove samples which longer than text_len; Replace number; Remove punctuations
    # Then create a vocabulary from inputs data     # 2079 samples, 2079 samples, 1295 words
    # tr_inputs, tr_outputs, max_dim = npreprocess_inputs(tr_inputs, tr_outputs, text_len[datasets[argv][1]], num_aspects)   # 2401, 2401, 2137
    # te_inputs, te_outputs, max_dim = npreprocess_inputs(te_inputs, te_outputs, text_len[datasets[argv][1]], num_aspects)   # 603 603
    # print(len(tr_inputs), len(tr_outputs), len(te_inputs), len(te_outputs))

    tr_outp = []
    for ip, op in zip(tr_inputs, tr_outputs):
        text = ip.text
        _text = re.sub('_', ' ', text)
        if len(_text.split(' ')) <= text_len[0]:
            tr_outp.append(op.scores)

    te_outp = []
    for ip, op in zip(te_inputs, te_outputs):
        text = ip.text
        _text = re.sub('_', ' ', text)
        if len(_text.split(' ')) <= text_len[0]:
            te_outp.append(op.scores)

    with open(r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_train_phobert.pkl", 'rb') as tr:
        tr_inputs = pickle.load(tr)
    print()

    with open(r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_test_phobert.pkl", 'rb') as te:
        te_inputs = pickle.load(te)
    print()

    x_tr, x_val, y_tr, y_val = train_test_split(np.array(tr_inputs), tr_outp, test_size=0.125, random_state=14)

    MLP_embedding = ['onehot', 'onehot_chi2']
    CNN_embedding = ['fasttext', 'fasttext_chi2', 'PhoBERT', ]  # 'PhoBERT_chi2'
    attention = ['fasttext_attention', 'fasttext_chi2_attention', 'PhoBERT_attention', ]    # 'PhoBERT_chi2_attention'

    # Call a model
    # model = ModelMLP(MLP_embedding[1])
    model = ModelCNN(CNN_embedding[2], text_len[datasets[argv][1]])
    # model = ModelCNN(attention[3], text_len[datasets[argv][1]])

    # Train model
    history = model.train(x_tr, x_val, y_tr, y_val)
    # Predict
    predicts = model.predict(np.array(te_inputs), te_outp)

    """ _________________________________________________________________________________________________________ """
    # Print the results
    cal_aspect_prf(te_outp, predicts, history, num_aspects, verbal=True)