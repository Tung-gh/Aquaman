import pandas as pd
import string

from Input_Output import Input, Output


punctuations = list(string.punctuation)
useless_labels = [296, 315, 330, 349, 295, 314, 329, 348]

def isNan(string):
    return string != string

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
        if isNan(row['text']) == 0:
            text = row['text'].strip()
            inputs.append(Input(text))


            scores = [0 if row[categories[i]] == 0 else 1 for i in range(0, num_aspects)]
            outputs.append(Output(aspects, scores))

    return inputs, outputs