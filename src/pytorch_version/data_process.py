import os
from pprint import pprint
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import json
import re
import os
import pandas as pd
import shutil
from process_tools import create_stride_test


def split_sents(text):
    sentences = re.split(r"([,，。 ！？?])", text)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    if len(sentences[-1]) == 0:  #
        sentences = sentences[:-1]
    return sentences


def get_train(base_dir='./data'):
    train_data_dir = os.path.join(base_dir, 'raw_data/train/data/')
    train_label_dir = os.path.join(base_dir, 'raw_data/train/label/')
    files = os.listdir(train_data_dir)  # 得到文件夹下的所有文件
    all_train = []
    for file in tqdm(files):
        fpath = os.path.join(train_data_dir, file)
        csv_index = file.split('.')[0] + '.csv'
        fcsv = os.path.join(train_label_dir, csv_index)
        with open(fpath, 'r') as f:
            data = f.read()
        df = pd.read_csv(os.path.join(fcsv))
        data_map = {}
        if '前明骏女孩' in data:
            print(file)
        data_map['text'] = data
        data_map['label'] = {}

        for i, row in df.iterrows():
            category = row['Category']
            privacy = row['Privacy']
            if file == '2162.txt' and privacy == '前明骏女孩组合队长"':
                privacy = '前明骏女孩组合队长'
            if category not in data_map['label']:
                data_map['label'][category] = {}
            if privacy not in data_map['label'][category]:
                data_map['label'][category][privacy] = []
            data_map['label'][category][privacy].append([row['Pos_b'], row['Pos_e']])

        all_train.append(data_map)
    return all_train


def get_add_data(base_dir='./data'):
    train_path = os.path.join(base_dir, 'user_data/cluener/train.json')
    dev_path = os.path.join(base_dir, 'user_data/cluener/dev.json')
    all_cluener = []
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            all_cluener.append(line)
    with open(dev_path, 'r') as f:
        for i, line in enumerate(f):
            all_cluener.append(line)
    return all_cluener


def get_test(base_dir='./data'):
    train_data_dir = os.path.join(base_dir, 'raw_data/test/')
    files = os.listdir(train_data_dir)  # 得到文件夹下的所有文件
    # print(len(files))
    all_test = {}
    for i in tqdm(range(3956)):
        fpath = os.path.join(train_data_dir, str(i) + '.txt')
        with open(fpath, 'r') as f:
            data = f.read()
        data_map = {}
        data_map['id'] = i
        data_map['text'] = data
        all_test[i] = data_map
    json_test = [json.dumps(data, ensure_ascii=False) + '\n' for i, data in all_test.items()]

    base_fold = os.path.join(base_dir, 'user_data/test')
    if not os.path.exists(base_fold):
        os.makedirs(base_fold)
    test_path = os.path.join(base_fold, f'goldtest.json')
    with open(test_path, 'w') as w:
        w.writelines(json_test)
    for i in range(5):
        source_file = test_path
        target_file = os.path.join(base_dir, f'user_data/5fold_mix/fold_{i}/goldtest.json')
        shutil.copyfile(source_file, target_file)
    return all_test


def get_ori_5fold_dataset(all_data, base_fold='./data'):
    indexs = [i for i in range(len(all_data))]
    kf = KFold(n_splits=5, shuffle=True, random_state=42).split(indexs)
    fmap = lambda x: json.dumps(all_data[x], ensure_ascii=False) + '\n'
    D = {}
    for i, (train_fold, test_fold) in enumerate(kf):
        # print(1)
        train = list(map(fmap, train_fold))
        test = list(map(fmap, test_fold))
        D[i] = {
            'train': train,
            'dev': test
        }
        # base_fold = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        # print(base_fold)
        folder = os.path.join(base_fold, f'user_data/5fold_ori/fold_{i}/')
        # print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, 'train.json'), 'w') as f:
            f.writelines(train)
        with open(os.path.join(folder, 'dev.json'), 'w') as f:
            f.writelines(test)
    return D


def get_mix_5fold_dataset(all_data, all_cluener, base_fold='./data'):
    for i in range(5):
        all_data[i]['train'].extend(all_cluener)
        # base_fold = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        # print(base_fold)
        folder = os.path.join(base_fold, f'user_data/5fold_mix/fold_{i}/')
        # print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, 'train.json'), 'w') as f:
            f.writelines(all_data[i]['train'])
        with open(os.path.join(folder, 'dev.json'), 'w') as f:
            f.writelines(all_data[i]['dev'])


if __name__ == '__main__':
    print(os.getcwd())
    all_train = get_train()
    all_data = get_ori_5fold_dataset(all_train)
    all_cluener = get_add_data()
    get_mix_5fold_dataset(all_data, all_cluener)
    get_test()
    create_stride_test('./data/user_data/test', './data/user_data/test')
