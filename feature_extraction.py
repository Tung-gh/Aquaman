import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

def Chi2(inputs, outputs, num_aspects):
    if num_aspects == 6:
        categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
    else:
        categories = ['Cau hinh','Mau ma','Hieu nang','Ship','Gia','Chinh hang','Dich vu','Phu kien']

    corpus = [ip.text for ip in inputs]

    cv = CountVectorizer()
    x = cv.fit_transform(corpus)

    y = []
    skb = [SelectKBest(chi2, k='all') for _ in range(num_aspects)]
    for i in range(num_aspects):
        y.append([op.scores[i] for op in outputs])
        _chi2 = skb[i].fit_transform(x, y[i])

        feature_names = cv.get_feature_names_out()
        _chi2_scores = skb[i].scores_
        _chi2_pvalues = skb[i].pvalues_

        chi2_dict = {}
        chi2_dict['word'] = feature_names
        chi2_dict['score'] = list(_chi2_scores)
        chi2_dict['pvalue'] = list(_chi2_pvalues)
        df = pd.DataFrame(chi2_dict, columns=['word', 'score', 'pvalue'])
        df = df.sort_values('score', ascending=False)

        with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_mebe/{}_chi2_dict/{}_{}.txt".format(str(sys.argv[1]), str(sys.argv[1]), categories[i]), 'w', encoding='utf8') as f:
            for w, s, p in zip(df['word'], df['score'], df['pvalue']):
                f.write('{} \t {} \t {}\n'.format(w, s, p))

    return inputs, outputs

