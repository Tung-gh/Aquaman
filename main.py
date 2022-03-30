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
data_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee.csv",
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebetiki.csv"
    ]
text_len = [40, ]     # 30, 38, 55, 52

if __name__ == '__main__':
    argv = sys.argv[1]
    data_path = data_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file
    inputs, outputs = load_data(data_path, num_aspects)     # 2086 samples, 2086 samples

    # Preprocess the inputs data: Remove samples which longer than text_len; Replace number; Remove punctuations
    # Then create a vocabulary from inputs data
    inputs, outputs, vocab = preprocess_inputs(inputs, outputs, text_len[datasets[argv][1]], num_aspects)   # 2079 samples, 2079 samples, 1295 words

    """ _________________________________________________________________________________________________________ """
    # Divide into train_set, test_set
    x_train, x_te, y_train, y_te = train_test_split(inputs, outputs, test_size=0.2, random_state=20)      # 1663, 416
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.125, random_state=20)

    # Embedding mode for MLP, CNN model
    MLP_embedding = ['onehot', 'onehot_chi2']
    CNN_embedding = ['fasttext', 'fasttext_chi2', 'PhoBERT']
    attention = ['fasttext_attention', 'fasttext_chi2_attention', 'PhoBERT_attention', 'PhoBERT_chi2_attention']

    # Call a model
    # model = ModelMLP(MLP_embedding[1])
    model = ModelCNN(CNN_embedding[0], text_len[datasets[argv][1]])

    # Train model
    history = model.train(x_tr, x_val, y_tr, y_val)
    # Predict the labels
    predicts = model.predict(x_te, y_te)

    """ _________________________________________________________________________________________________________ """
    # Print the results
    cal_aspect_prf(y_te, predicts, history, num_aspects, verbal=True)




