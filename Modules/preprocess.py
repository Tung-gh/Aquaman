import sys
import io
import pandas as pd
import numpy as np
import string

from Input_Output import Input, Output
from sklearn.feature_extraction.text import CountVectorizer

punctuations = list(string.punctuation)
useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']


def is_nan(s):
    return s != s


def contains_punctuation(s):
    for c in s:
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
    df = pd.read_csv(data_path,  encoding='utf-8')
    aspects = list(range(num_aspects))
    for index, row in df.iterrows():
        annotation = row['annotations']
        _annotation = annotation.strip().split(', ')
        text_annotation = [str(_annotation[i][-3:]) for i in range(0, len(_annotation), 2)]

        if is_nan(row['text']) == 0 and typo_trash_labeled(text_annotation) == 0:
            text = row['text'].strip()
            inputs.append(Input(text, text_annotation))

            scores = [0 if row[categories[i]] == 0 else 1 for i in range(0, num_aspects)]
            outputs.append(Output(aspects, scores))

    return inputs, outputs


def preprocess_inputs(inputs):
    for ip in inputs:
        text = ip.text.strip().replace('_', ' ').split(' ')
        for j in range(len(text)):
            if contains_digit(text[j].strip()):
                text[j] = '0'
        for token in text:
            if len(token) <= 1 or token.strip() in punctuations:
                text.remove(token)
        ip.text = ' '.join(text)

    return inputs


def make_vocab(inputs):

    """
    corpus = [ip.text for ip in inputs]
    cv = CountVectorizer()
    x = cv.fit_transform(corpus)
    vocab = x.get_feature_names_out()
    with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))
    """

    vocab = []
    for ip in inputs:
        text = ip.text.split(' ')
        for token in text:
            vocab.append(token)
    # Make a non-duplicated vocabulary
    vocab = list(dict.fromkeys(vocab))

    with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab


def load_chi2(path):
    dictionary = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            t = line.strip().split(' ')
            dictionary[t[0]] = t[2]

    return dictionary


def load_fasttext(path):
    num_words = 100000
    fasttext = {}
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    i = 0
    for line in fin:
        i += 1
        tokens = line.rstrip().split(' ')
        fasttext[tokens[0]] = np.array([float(val) for val in tokens[1:]])
        fasttext[tokens[0]] /= np.linalg.norm(fasttext[tokens[0]])

        if i > num_words:
            break

    return fasttext
