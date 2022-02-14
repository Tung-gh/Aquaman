import sys

from Modules.Preprocess import load_data

datasets = {'mebeshopee': [6, 0],
              'mebetiki': [6, 1],
              'techshopee': [8, 2],
              'techtiki': [8, 3] }
data_paths = [
    r"H:\DS&KT Lab\NCKH\OpinionMining\data_mebe\mebe_shopee.csv",
    r"H:\DS&KT Lab\NCKH\OpinionMining\data_mebe\mebe_tiki.csv"
]


if __name__ == '__main__':
    set = sys.argv[1]
    if set not in list(datasets.keys()):
        print('There is no category', set)
        exit
    else:
        data_path = data_paths[datasets[set][1]]
        num_aspects = datasets[set][0]

        inputs, outputs = load_data(data_path, num_aspects)