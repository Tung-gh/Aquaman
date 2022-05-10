import io
import numpy as np
import pickle

if __name__ == '__main__':
    # MAKING FASTTEXT FOR EMBEDDING
    num_words = 500000
    fasttext = {}
    fin = io.open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\cc.vi.300.vec", 'r', encoding='utf-8', newline='\n', errors='ignore')
    i = 0
    for line in fin:
        i += 1
        tokens = line.rstrip().split(' ')
        fasttext[tokens[0]] = np.array([float(val) for val in tokens[1:]])
        fasttext[tokens[0]] /= np.linalg.norm(fasttext[tokens[0]])

        if i > num_words:
            break

    with open(r"H:\DS&KT Lab\NCKH\Aquaman\vn_fasttext\fasttext.pkl", 'wb') as fasttext_emb:
        pickle.dump(fasttext, fasttext_emb, protocol=pickle.HIGHEST_PROTOCOL)


    # MAKING PHOBERT FOR EMBEDDING





