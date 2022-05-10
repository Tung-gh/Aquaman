import torch
import sys
from Modules.preprocess import load_data, preprocess_inputs
from Modules.evaluation import cal_aspect_prf
from Modules.Proposed_Models.Model_MLP import ModelMLP
from Modules.Proposed_Models.Model_CNN import ModelCNN
from sklearn.model_selection import train_test_split

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

    # Load inputs, outputs from data file
    tr_inputs, tr_outputs = load_data(train_path, num_aspects)     # 2403
    te_inputs, te_outputs = load_data(test_path, num_aspects)      # 603

    # Preprocess the inputs data: Remove samples which longer than text_len; Replace number; Remove punctuations
    # Then create a vocabulary from inputs data
    tr_inputs, tr_outputs = preprocess_inputs(tr_inputs, tr_outputs, text_len[datasets[argv][1]], num_aspects)   # 2401, 2401, 2137
    te_inputs, te_outputs = preprocess_inputs(te_inputs, te_outputs, text_len[datasets[argv][1]], num_aspects)   # 603 603

    """ _________________________________________________________________________________________________________ """
    # Divide into train_set, valid_set,
    x_tr, x_val, y_tr, y_val = train_test_split(tr_inputs, tr_outputs, test_size=0.125, random_state=14)

    # Embedding mode for MLP, CNN model
    MLP_embedding = ['onehot', 'onehot_chi2']
    CNN_embedding = ['fasttext', 'fasttext_chi2', 'PhoBERT', 'PhoBERT_chi2', 'fasttext_mca', 'fasttext_chi2_mca']
    attention = ['fasttext_attention', 'fasttext_chi2_attention', 'PhoBERT_attention', 'PhoBERT_chi2_attention']

    # Call a model
    # model = ModelMLP(MLP_embedding[0])
    model = ModelCNN(CNN_embedding[3], text_len[datasets[argv][1]])
    # model = ModelCNN(attention[0], text_len[datasets[argv][1]])

    # Train model
    history = model.train(x_tr, x_val, y_tr, y_val)

    # Predict the labels
    predicts = model.predict(te_inputs, te_outputs)

    """ _________________________________________________________________________________________________________ """
    # Print the results
    cal_aspect_prf(te_outputs, predicts, history, num_aspects, verbal=True)




