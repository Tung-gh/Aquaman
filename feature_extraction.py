import sys
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def Chi2(inputs, outputs, num_aspects):
    if num_aspects == 6:
        categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
    else:
        categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']

    cv = CountVectorizer()
    x = cv.fit_transform(inputs)

    y = []
    skb = []
    for i in range(num_aspects):
        _skb = SelectKBest(chi2, k='all')
        skb.append(_skb)
        y.append([op[i] for op in outputs])
        _chi2 = _skb.fit_transform(x, y[i])

        feature_names = cv.get_feature_names_out()
        _chi2_scores = _skb.scores_
        _chi2_pvalues = _skb.pvalues_

        chi2_dict = {}
        chi2_dict['word'] = feature_names
        chi2_dict['score'] = list(_chi2_scores)
        chi2_dict['pvalue'] = list(_chi2_pvalues)
        df = pd.DataFrame(chi2_dict, columns=['word', 'score', 'pvalue'])
        df = df.sort_values('score', ascending=False)

        with open(r"H:/DS&KT Lab/NCKH/Aquaman/data/data_{}/{}_chi2_dict/old/{}_phobert_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), categories[i]), 'w', encoding='utf8') as f:
            for w, s, p in zip(df['word'], df['score'], df['pvalue']):
                f.write('{} \t {} \t {}\n'.format(w, s, p))
