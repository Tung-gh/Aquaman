import sys
import pandas as pd
import string
import matplotlib.pyplot as plt

from Input_Output import Input, Output
from feature_extraction import Chi2
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator


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
        categories = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện']

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


def make_vocab(inputs):

    # """
    cv = CountVectorizer()
    x = cv.fit_transform(inputs)
    vocab = cv.get_feature_names_out()
    with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))
    # """

    # vocab = []
    # for ip in inputs:
    #     text = ip.text.split(' ')
    #     for token in text:
    #         vocab.append(token)
    # # Make a non-duplicated vocabulary
    # vocab = list(dict.fromkeys(vocab))
    #
    # with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
    #     for w in vocab:
    #         f.write('{}\n'.format(w))

    return vocab


def preprocess_inputs(inputs, outputs, text_len, num_aspects):
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        text = ip.text.strip().replace('_', ' ').split(' ')
        if len(text) <= text_len:
            for j in range(len(text)):
                if contains_digit(text[j].strip()):
                    text[j] = '0'
            for token in text:
                if len(token) <= 1 or token.strip() in punctuations:
                    text.remove(token)
            ip.text = ' '.join(text)
            inp.append(ip.text)
            outp.append(op.scores)

    # le = []
    # for ip in inp:
    #     le.append(len(ip.split(' ')))
    # x = Counter(le).keys()
    # y = Counter(le).values()
    # print(max(x))
    # plt.bar(x, y)
    # plt.show()

    # for i in range(6):
    #     li = []
    #     for ip, op in zip(inp, outp):
    #         if op[i] == 1:
    #             li.append(ip)
    #     text = " ".join(i for i in li)
    #     wcl = WordCloud(background_color='white').generate(text)
    #     plt.imshow(wcl)
    #     plt.show()

    vocab = make_vocab(inp)
    # Chi2(inp, outp, num_aspects)
    return inp, outp, vocab


def load_chi2(path):
    dictionary = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            t = line.strip().split(' ')
            dictionary[t[0]] = float(t[2])

    return dictionary
