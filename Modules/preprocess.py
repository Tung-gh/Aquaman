import sys
import pandas as pd
import re
import matplotlib.pyplot as plt
import string

from Input_Output import Input, Output
from feature_extraction import Chi2
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from vncorenlp import VnCoreNLP

punctuations = list(string.punctuation)
useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']


def is_nan(s):
    return s != s


def typo_trash_labeled(lst):
    for i in lst:
        if i in useless_labels:
            return True
    return False


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
    cv = CountVectorizer()
    x = cv.fit_transform(inputs)
    vocab = cv.get_feature_names_out()
    with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_vocab.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1])), 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab


def preprocess_inputs(inputs, outputs, text_len, num_aspects):
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        text = ip.text.strip()
        text = re.sub('_', ' ', text)
        if len(text.split(' ')) <= text_len:
            text = re.sub("\d+", '', text)
            text = re.sub('\W+', ' ', text).strip()
            inp.append(text)
            outp.append(op.scores)

    """
    OTHERS PROCESSES
    """
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
    #     plt.axis('off')
    #     plt.show()

    # Make vocabulary
    # vocab = make_vocab(inp)

    # # Make Chi2_dictionary for fasttext embedding
    # Chi2(inp, outp, num_aspects)

    # # Make Chi2_dictionary for PhoBERT embedding
    # rdrsegmenter = VnCoreNLP(r'H:\DS&KT Lab\NCKH\Aquaman\vncorenlp\VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
    # segmented_texts = []  # Texts which are segmented
    # for i in range(len(inp)):
    #     ip = rdrsegmenter.tokenize(inp[i])
    #     for s in ip:
    #         segmented_texts.append(' '.join(s))
    #     assert len(segmented_texts) == i+1
    # Chi2(segmented_texts, outp, num_aspects)

    return inp, outp    # , vocab


def load_chi2(path):
    dictionary = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            t = line.strip().split(' ')
            dictionary[t[0]] = float(t[2])

    return dictionary
