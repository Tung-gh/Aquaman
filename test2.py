import torch
import numpy as np
import tensorflow as tf

from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

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
text_len = [30, 38, 55, 52]

if __name__ == '__main__':
    argv = sys.argv[1]
    data_path = data_paths[datasets[argv][1]]
    num_aspects = datasets[argv][0]

    # Load inputs, outputs from data file
    inputs, outputs = load_data(data_path, num_aspects)     # 2086 samples, 2086 samples

    # Preprocess the inputs data: Remove samples which longer than text_len; Replace number; Remove punctuations
    # Then create a vocabulary from inputs data
    inputs, outputs, vocab = preprocess_inputs(inputs, outputs, text_len[datasets[argv][1]], num_aspects)   # 2051 samples, 2051 samples, 1225 words


# from main import inputs
# if __name__ == '__main__':
# def make_phobert_embedding(inputs):
    # Call for PhoBERT pretrained model from huggingface-transformers
    phoBert = AutoModel.from_pretrained('vinai/phobert-base', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)

    # Load and use RDRSegmenter from VnCoreNLP as recommended by PhoBERT authors
    rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar',
                             annotators='wseg', max_heap_size='-Xmx500m')

    print('\n Segmented texts:')
    segmented_texts = []    # Texts which are segmented
    for inp in inputs[:10]:
    # for inp in inputs:
        _inp = rdrsegmenter.tokenize(inp)
        print(_inp)
        for sen in _inp:
            segmented_texts.append(' '.join(sen))
            print(len(' '.join(sen).split(' ')), ' '.join(sen))
    print('Total:', len(segmented_texts))

    print('\n Encoded texts:')
    encoded_texts = []      # Texts are converted to indices vectors with 2 added token indices for <s>, </s>
    for st in segmented_texts:
        _st = tokenizer.encode(st)
        encoded_texts.append(_st)
        print(int(len(_st)) - 2, _st)
    print('Total2:', len(encoded_texts))

    # padded_texts = pad_sequences(encoded_texts, maxlen=30, padding='post', truncating='post')
    # for pt in padded_texts:
    #     print(len(pt))

    print('Masked positions:')
    masked_pos = []
    for mp in encoded_texts:
        m = [int(token_id > 0) for token_id in mp]
        masked_pos.append(m)
    #     print(m)
    # print(masked_pos)

    tensors, masks = [], []
    for i in range(len(masked_pos)):
        tensors.append(torch.tensor([encoded_texts[i]]))
        masks.append(torch.tensor([masked_pos[i]]))
    print(len(tensors), len(masks))

    print('\n Lhs:')
    lhs = []    # There are 13 tensors of 13 attention layers from PhoBERT <=> 1 word has 13 (768,)-tensor
    for i in range(len(tensors)):
        with torch.no_grad():
            f = phoBert(tensors[i], masks[i])
            hs = f[2]   # Len: 13 as 13 output tensors from 13 attention layers
            _hs = np.squeeze(np.array([x.detach().numpy() for x in hs]), axis=1)
            print(type(_hs), _hs.shape, type(_hs[0]), _hs[0].shape)    # _hs.shape: (attention_head * no_words * dim)
            lhs.append(_hs)

    print('\n reshaped_lhs:')
    reshaped_lhs = []
    for rlhs in lhs:
        _rlhs = []
        print(rlhs.shape[1])
        for i in range(rlhs.shape[1]):
            a = np.array([x[i] for x in rlhs])
            print(a.shape)
            _rlhs.append(a)
        reshaped_lhs.append(_rlhs)
    print(len(reshaped_lhs))

    texts_token_emb = []
    for tte in reshaped_lhs:
        _tte = []
        for i in tte:
            emb = tf.reduce_sum(i[-4:], axis=0)
            _tte.append(emb)
        print(len(_tte))
        texts_token_emb.append(np.array(_tte[1:-1]))
    print(len(texts_token_emb))

    print()