import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

def Chi2(inputs, outputs, num_aspects):
    corpus = [ip.text for ip in inputs]

    cv = CountVectorizer()
    x = cv.fit_transform(corpus) #, dtype='int32'

    y = []
    for i in range(num_aspects):
        y.append([op.scores[i] for op in outputs])

    print()
