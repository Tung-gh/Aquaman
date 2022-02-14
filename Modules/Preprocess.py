import sys

import pandas as pd
import string

from Input_Output import Input, Output


punctuations = list(string.punctuation)
useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']

def isNan(string):
    return string != string

def contains_punctuation(string):
    for c in string:
        if c in punctuations:
            return True
    return False

def contains_digit(w):
    for i in w:
        if i.isdigit():
            return True
    return False

def typo_trash_labeled(lst):
    for i in lst:
        if i in useless_labels:
            return True
    return False

def load_data(data_path, num_aspects):
    if num_aspects == 6:
        categories = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
    else:
        categories = ['cấu hình','mẫu mã','hiệu năng','ship','giá','chính hãng','dịch vụ','phụ kiện']

    inputs, outputs = [], []
    df = pd.read_csv(data_path, encoding='utf-8')
    aspects = list(range(num_aspects))
    for index, row in df.iterrows():
        annotation = row['annotations']
        _annotation = annotation.strip().split(', ')
        input_annotation = [str(_annotation[i][-3:]) for i in range(0, len(_annotation), 2)]

        if isNan(row['text']) == 0 and typo_trash_labeled(input_annotation) == 0:
            text = row['text'].strip()
            inputs.append(Input(text, input_annotation))

            scores = [0 if row[categories[i]] == 0 else 1 for i in range(0, num_aspects)]
            outputs.append(Output(aspects, scores))

    return inputs, outputs

def preprocess_inputs(inputs):
    for ip in inputs:
        text = ip.text.strip()
        _text = text.split(' ')
        for token in _text:
            if contains_punctuation(token) == 0:
                _text.remove(token)
        for i in range(len(_text)):
            if contains_digit(_text[i]):
                _text[i] = '0'
        ip.text = ' '.join(_text)

    return inputs

def make_vocab(inputs):
    vocab = []
    for ip in inputs:
        text = ip.text.split(' ')
        for token in text:
            vocab.append(token)
    #Make a non-duplicated vocabulary
    vocab = list(dict.fromkeys(vocab))

    with open(r"H:/DS&KT Lab/NCKH/OpinionMining/data_mebe/" + str(sys.argv[1]) + "_vocab.txt", 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab

