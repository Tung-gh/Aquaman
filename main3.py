import torch
import sys
import pickle

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
    tr_inputs, tr_outputs = npreprocess_inputs(tr_inputs, tr_outputs, text_len[datasets[argv][1]], num_aspects)   # 2401, 2401, 2137
    te_inputs, te_outputs = npreprocess_inputs(te_inputs, te_outputs, text_len[datasets[argv][1]], num_aspects)   # 603 603
    print(len(tr_inputs), len(tr_outputs), len(te_inputs), len(te_outputs))
    # fasttext = None
    # with open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\fasttext.pkl", 'rb') as fasttext_emb:
    #     fasttext = pickle.load(fasttext_emb)
    #
    # print()
    # dic, out_fast, out_vocab = [], [], []
    # count = 0
    # for ip in tr_inputs:
    #     text = ip.split(' ')
    #     for w in text:
    #         if w not in dic:
    #             dic.append(w)
    #         if w not in fasttext and w not in out_fast:
    #             out_fast.append(w)
            # if w not in vocab and w not in out_vocab:
            #     out_vocab.append(w)

    # print(len(dic), len(out_fast), len(out_vocab))  # dic: 1339, out_fast: 120, out_vocab: 44
    # print(out_fast)     # fasttext cover 91% words in train dataset. These out-of-fasttext words are all users' specific words that bigger fasttext dictionary still can't cover them
    # print(out_vocab)    # 44 words from train dataset not in vocab. These words are all special characters (96,6% remaining)

