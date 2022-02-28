import sys
from Modules.preprocess import load_data, preprocess_inputs, make_vocab, load_fasttext
from Modules.evaluation import cal_aspect_prf
from Modules.Proposed_Models.Model_MLP import ModelMLP
from Modules.Proposed_Models.Model_CNN import ModelCNN
from sklearn.model_selection import train_test_split
from feature_extraction import Chi2

datasets = {'mebeshopee': [6, 0],
            'mebetiki': [6, 1],
            'techshopee': [8, 2],
            'techtiki': [8, 3]
            }
data_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee.csv",
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebetiki.csv"
    ]
e_path = r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\cc.vi.300.vec"

if __name__ == '__main__':
    argv = sys.argv[1]
    data_path = data_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file
    inputs, outputs = load_data(data_path, num_aspects)     # 2086 samples, 2086 samples

    text_len = [30, 38, 55, 52]

    # Preprocess the inputs data: Remove samples which longer than text_len; Replace number; Remove punctuations
    inputs, outputs = preprocess_inputs(inputs, outputs, text_len[datasets[argv][1]])   # 2051 samples, 2051 samples

    # # Make a vocabulary from the inputs data
    # vocab = make_vocab(inputs)      # 1225 words

    """ _________________________________________________________________________________________________________ """
    # # Make chi2 dictionary for every aspect
    # Chi2(inputs, outputs, num_aspects)
    # Load fastText embedding
    # fasttext = load_fasttext(e_path)

    # # Divide into train_set, test_set
    x_tr, x_te, y_tr, y_te = train_test_split(inputs, outputs, test_size=0.2, random_state=20)      # 1640, 411

    # Embedding mode for MLP, CNN model
    MLP_embedding = ['onehot', 'onehot_chi2']
    CNN_embedding = ['fasttext', 'fasttext_chi2_attention']

    # Call a model
    model = ModelMLP(MLP_embedding[0])
    # model = ModelCNN(CNN_embedding[1], text_len[datasets[argv][1]], fasttext)

    # Train model
    model.train(x_tr, y_tr)
    # Predict the labels
    predicts = model.predict(x_te, y_te)

    """ _________________________________________________________________________________________________________ """
    # Print the results
    # if num_aspects == 6:
    #     print("\t\t Ship \t\t Gia \t\t Chinh hang \t\t Chat luong \t\t Dich vu \t\t An toan")
    # else:
    #     print("\t\t Cau hinh \t\t Mau ma \t\t Hieu nang \t\t Ship \t\t Gia \t\t Chinh hang \t\t Dich vu \t\t Phu kien")
    cal_aspect_prf(y_te, predicts, num_aspects, verbal=True)




