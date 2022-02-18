import sys
from Modules.preprocess import load_data, preprocess_inputs, make_vocab
from Modules.evaluation import cal_aspect_prf
from Modules.Proposed_Models.MLP_Model import MLP_Model
from sklearn.model_selection import train_test_split
from feature_extraction import Chi2

datasets = {'mebeshopee': [6, 0],
              'mebetiki': [6, 1],
              'techshopee': [8, 2],
              'techtiki': [8, 3] }
data_paths = [
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee.csv",
    r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebetiki.csv"
]


if __name__ == '__main__':
    argv = sys.argv[1]
    # if argv not in list(datasets.keys()):
    #     print('There is no category', argv)
    #     exit
    # else:

    data_path = data_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]
    # Load inputs, outputs from data file
    inputs, outputs = load_data(data_path, num_aspects)     # 2086 samples, 2086 samples
    # Preprocess the inputs data
    inputs = preprocess_inputs(inputs)
    # Make a vocabulary from the inputs data
    vocab = make_vocab(inputs)      # 1354 words

    """ CREATE AND RUN THE MODEL """
    ips, ops = Chi2(inputs, outputs, num_aspects)
    # Divide into train_set, test_set
    x_tr, x_te, y_tr, y_te = train_test_split(inputs, outputs, test_size=0.2, random_state=20)
    # Create a model variable
    model = MLP_Model()
    model.train(x_tr, y_tr)
    # Predict the labels
    predicts = model.predict(x_te)
    # Print the results
    if num_aspects == 6:
        print("\t\t ship \t\t giá \t\t chính hãng \t\t chất lượng \t\t dịch vụ \t\t an toàn")
    else:
        print("\t\t cấu hình \t\t mẫu mã \t\t hiệu năng \t\t ship \t\t giá \t\t chính hãng \t\t dịch vụ \t\t phụ kiện")
    cal_aspect_prf(y_te, predicts, num_aspects, verbal=True)


