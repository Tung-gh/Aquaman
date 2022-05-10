import torch
import sys
import pickle

from test import nload_data, npreprocess_inputs
from sklearn.model_selection import train_test_split
from Modules.Proposed_Models.Model_CNN import ModelCNN
from Modules.evaluation import cal_aspect_prf

datasets = {'mebeshopee': [6, 0],
            'mebetiki': [6, 1],
            'techshopee': [8, 2],
            'techtiki': [8, 3]
            }
train_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\mebeshopee_train.csv",
    ]
test_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\mebeshopee_test.csv",
    ]
text_len = [40, ]     # 40, 38, 55, 52

if __name__ == '__main__':
    argv = sys.argv[1]
    train_path = train_paths[datasets[argv][1]]
    test_path = test_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file   # 2086 samples, 2086 samples
    tr_inputs, tr_outputs = nload_data(train_path, num_aspects)     # 1663
    te_inputs, te_outputs = nload_data(test_path, num_aspects)      # 416
    print()
    tr_inputs, tr_outputs = npreprocess_inputs(tr_inputs, tr_outputs, text_len[0])
    x_te, y_te = npreprocess_inputs(te_inputs, te_outputs, text_len[0])

    x_tr, x_val, y_tr, y_val = train_test_split(tr_inputs, tr_outputs, test_size=0.125, random_state=14)
    print()
    f_embedding = ['fasttext', 'fasttext_chi2', 'fasttext_attention', 'fasttext_chi2_attention', 'fasttext_mca',
                   'fasttext_chi2_mca']
    p_embedding = ['PhoBERT', 'PhoBERT_chi2', 'PhoBERT_attention', 'PhoBERT_chi2_attention', 'PhoBERT_mca',
                   'PhoBERT_chi2_mca']
    # Call a model
    f_model = ModelCNN(f_embedding[0], text_len[0])
    p_model = ModelCNN(p_embedding[0], text_len[0])
    # Train model
    f_history = f_model.train(x_tr, x_val, y_tr, y_val)

    # Predict the labels
    predicts = f_model.predict(x_te, y_te)

    # Print the results
    cal_aspect_prf(y_te, predicts, f_history, num_aspects, verbal=True)
