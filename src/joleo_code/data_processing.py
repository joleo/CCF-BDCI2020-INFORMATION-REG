#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/16 22:00
# @Author : aomo
# @Site : 
# @File : data_processing.py
import pandas as pd
import numpy as np
from tqdm import tqdm
# from joleo_code.config import *
import os

def gen_bio_data(df):
    D = list()
    for value in df:
        d, last_flag = list(), ''
        for i in value:
            char, this_flag = i.split(' ')
            if this_flag == 'O' and last_flag == 'O':
                d[-1][0] += char
            elif this_flag == 'O' and last_flag != 'O':
                d.append([char, 'O'])
            elif this_flag[:1] == 'B':
                d.append([char, this_flag[2:]])
            else:
                d[-1][0] += char
            last_flag = this_flag
        D.append(d)
    return D

def gen_new_data(train_path='./data/user_data/cluener/train.json', val_path='./data/user_data/cluener/dev.json'):
    tr = pd.read_json(train_path, lines=True)
    dev = pd.read_json(val_path, lines=True)
    new_data = pd.concat([tr, dev], axis=0).reset_index(drop=True)
    all_df = list()
    for i in tqdm(range(len(new_data))):
        df_content = new_data.iloc[i].values
        df_enty = pd.DataFrame(df_content[1]).stack().reset_index()
        df_enty.columns = ['entity', 'val', 'position']
        df_enty['start'] = df_enty['position'].apply(lambda x: x[0][0])
        df_enty['end'] = df_enty['position'].apply(lambda x: x[0][1])
        df_enty = df_enty.sort_values(['start', 'end'], ascending=True).reset_index(drop=True)
        content = list(df_content[0])
        df = pd.DataFrame(content).reset_index()
        df.columns = ['id', 'content']
        entity_ = df_enty.values
        part_map = {}
        for j in range(len(entity_)):
            num_ = np.arange(entity_[j][3], entity_[j][4] + 1)
            for h, k in enumerate(num_):
                if h == 0:
                    part_map[k] = 'B-' + entity_[j][1]
                else:
                    part_map[k] = 'I-' + entity_[j][1]
        df['val'] = df['id'].map(part_map).fillna('O')
        df.columns = ['id', 'content', 'val']
        df = list(map(lambda x, y: x + ' ' + y, df['content'], df['val']))
        all_df.append(df)

    D = gen_bio_data(all_df)
    return D

def gen_train_data(train_label_path='./data/raw_data/train/label/', train_data_path='./data/raw_data/train/data/'):
    filelist = os.listdir(train_label_path)  # [1000:1002]
    filelist = list(set([i.split('.')[0] for i in filelist]))
    label_list = list()
    all_train = list()
    for i in tqdm(filelist):
        entity = pd.read_csv(train_label_path + i + '.csv')
        entity.columns = ['uid', 'val', 'start', 'end', 'entity']
        entity = entity.sort_values(['start', 'end'], ascending=True).reset_index(drop=True)
        content = pd.read_table(train_data_path + i + '.txt', header=None, names=['content'])
        content = list(content.values[0][0])
        ct = pd.DataFrame(content).reset_index(drop=True).reset_index()
        ct.columns = ['uid', 'content']
        entity_ = entity.values
        part_map = {}
        for j in range(len(entity_)):
            num_ = np.arange(entity_[j][2], entity_[j][3] + 1)
            for h, k in enumerate(num_):
                if h == 0:
                    part_map[k] = 'B-' + entity_[j][1]
                else:
                    part_map[k] = 'I-' + entity_[j][1]
        ct['val'] = ct['uid'].map(part_map).fillna('O')
        ct.columns = ['uid', 'content', 'val']
        ct = list(map(lambda x, y: x + ' ' + y, ct['content'], ct['val']))
        all_train.append(ct)
    D = gen_bio_data(all_train)
    return D

def gen_sub_test(test_path='./data/raw_data/test/'):
    test = pd.DataFrame()
    for i in tqdm(os.listdir(test_path)):
        content = pd.read_table(test_path+i,header = None,names = ['content'])
        content['ID'] = int(i.split('.')[0])
        test = pd.concat([test,content],axis=0 ,sort= True).reset_index(drop = True)
        test['length'] = test['content'].apply(len)
    return test

def gen_test_data(test_path='./data/raw_data/test/'):
    test = pd.DataFrame()
    for i in tqdm(os.listdir(test_path)):
        content = pd.read_table(test_path + i, header=None, names=['content'])
        content['id'] = i
        test = pd.concat([test, content], axis=0).reset_index(drop=True)
    return test



