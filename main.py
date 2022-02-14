import sys

from Modules.Preprocess import load_data, make_corpus, preprocess_inputs, make_vocab

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
    if argv not in list(datasets.keys()):
        print('There is no category', argv)
        exit
    else:
        data_path = data_paths[datasets[argv][1]]
        num_aspects = datasets[argv][0]
        # Load inputs, outputs from data file
        inputs, outputs = load_data(data_path, num_aspects)     # 2086 samples, 2086 samples
        # Preprocess the inputs data
        inputs = preprocess_inputs(inputs)
        # Make a vocabulary from the inputs data
        vocab = make_vocab(inputs)                              #1354 words
        print(len(vocab))


