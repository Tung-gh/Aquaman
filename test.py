import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
import pickle

from Input_Output import Input, Output
from feature_extraction import Chi2
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']


def is_nan(s):
    return s != s


def nload_data(data_path, num_aspects):
    inputs, outputs = [], []
    df = pd.read_csv(data_path,  encoding='utf-8')
    aspects = list(range(num_aspects))
    for index, row in df.iterrows():
        if is_nan(row['text']) == 0:
            text = row['text'].strip()
            inputs.append(text)     # Input(text)

            _scores = list(row['label'][1:-1].split(', '))
            scores = [int(i) for i in _scores[:num_aspects]]
            outputs.append(scores)  # Output(aspects, scores)

    return inputs, outputs


def make_vocab(inputs):
    cv = CountVectorizer()
    x = cv.fit_transform(inputs)
    vocab = cv.get_feature_names_out()
    with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab_new.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab


def npreprocess_inputs(inputs, outputs, text_len): # , num_aspects
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        text = ip.strip()      # ip.text
        _text = re.sub('_', ' ', text)
        if len(_text.split(' ')) <= text_len:
            _text = re.sub('\d+', '', _text)
            _text = re.sub('\W+', ' ', _text).strip()
            inp.append(_text)
            outp.append(op)   # op.scores
    print(len(inp), len(outp))

    # # Check length of all texts
    # le = []
    # for ip in inp:
    #     le.append(len(ip.split(' ')))
    # x = Counter(le).keys()
    # y = Counter(le).values()
    # print(max(x))
    # plt.bar(x, y)
    # plt.show()

    # # Plot word cloud for each aspect
    # for i in range(num_aspects):
    #     li = []
    #     for ip, op in zip(inp, outp):
    #         if op[i] == 1:
    #             li.append(ip)
    #     text = " ".join(i for i in li)
    #     wcl = WordCloud(background_color='white').generate(text)
    #     plt.imshow(wcl)
    #     plt.show()

    # vocab = make_vocab(inp)
    # Chi2(inp, outp, num_aspects)

    # phoBert = AutoModel.from_pretrained('vinai/phobert-base', output_hidden_states=True)
    # tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    # rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
    #
    # segmented_texts = []  # Texts which are segmented
    # for i in range(len(inp)):
    #     ip = rdrsegmenter.tokenize(inp[i])
    #     for s in ip:
    #         segmented_texts.append(' '.join(s))
    #     assert len(segmented_texts) == i+1
    # print('length seg, outp', len(segmented_texts), len(outp))
    # # Chi2(segmented_texts, outp, num_aspects)
    #
    # encoded_texts = []  # Texts are converted to indices vectors with 2 added token indices for <s>, </s>
    # for st in segmented_texts:
    #     _st = tokenizer.encode(st)
    #     encoded_texts.append(_st)
    # max_dim = max([len(i) for i in encoded_texts])
    # print(max_dim)
    #
    # masked_pos = []
    # for mp in encoded_texts:
    #     m = [int(token_id > 0) for token_id in mp]
    #     masked_pos.append(m)
    #
    # tensors, masks = [], []  # Convert list of indices to torch tensor
    # for i in range(len(masked_pos)):
    #     tensors.append(torch.tensor([encoded_texts[i]]))
    #     masks.append(torch.tensor([masked_pos[i]]))
    #
    # lhs = []  # There are 13 tensors of 13 attention layers from PhoBERT <=> 1 word has 13 (768,)-tensor
    # with torch.no_grad():
    #     for i in range(len(tensors)):
    #         f = phoBert(tensors[i], masks[i])
    #         hs = f[2]  # Len: 13 as 13 output tensors from 13 attention layers
    #         _hs = np.squeeze(np.array([x.detach().numpy() for x in hs]), axis=1)  # Reduce the dimension
    #         lhs.append(_hs)     # 13 * num_word * 768
    #
    # reshaped_lhs = []  # Shape: num_words * 13 * 768
    # for rlhs in lhs:
    #     _rlhs = []
    #     for i in range(rlhs.shape[1]):
    #         a = np.array([x[i] for x in rlhs])
    #         _rlhs.append(a)
    #     reshaped_lhs.append(_rlhs)
    #
    # texts_token_emb = []  # Shape: num_words * 768
    # for tte in reshaped_lhs:
    #     _tte = []
    #     for i in tte:
    #         emb = tf.reduce_sum(i[-4:], axis=0)
    #         _tte.append(emb)
    #     texts_token_emb.append(np.array(_tte[1:-1]))
    #
    # texts_matrix = []
    # for te in texts_token_emb:
    #     emb = np.zeros((75-2, 768))
    #     for i in range(len(te)):
    #         emb[i] = te[i]
    #     texts_matrix.append(emb)
    #
    # # with open(r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_train_phobert.pkl", 'wb') as pf:
    # with open(r"H:\DS&KT Lab\NCKH\Aquaman\data\data_mebe\mebeshopee_test_phobert.pkl", 'wb') as pf:
    #     pickle.dump(texts_matrix, pf, protocol=pickle.HIGHEST_PROTOCOL)

    return inp, outp #, max_dim    #, vocab
