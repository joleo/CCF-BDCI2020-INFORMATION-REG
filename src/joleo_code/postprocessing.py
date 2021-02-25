#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/16 22:13
# @Author : aumo
# @Site : 
# @File : postprocessing.py

from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *

def _split(x):
    try:
        return x.split('_')[1]
    except:
        return 0

def get_most_common(arr):
    return Counter(arr).most_common(50)

def drop_dup(x):
    set_list = list()
    for i in x:
        if i not in set_list:
            set_list.append(i)
    return set_list

def get_top_cnt(x):
    all_list = list()
    for i in x:
        if i[1] > cnt:
            all_list.append(i[0])
    return [list(i) for i in all_list]

def get_position(x, i):
    try:
        if i == 1:
            return int(x.split('_')[i]) + 1
        elif i == 0:
            return x.split('_')[0]
        else:
            return int(x.split('_')[i])
    except:
        return []


def get_lb(x, i):
    try:
        if i == 1:
            return int(x.split('_')[i]) + 1
        elif i == 0:
            return x.split('_')[0]
        else:
            return int(x.split('_')[i])
    except:
        return []


def submit(data_test):
    sub = data_test.copy()
    for i in tqdm(range(n_folds)):
        sub['key_' + str(i)] = sub['submit_' + str(i)]
        if i == 0:
            sub['all_key'] = sub['key_0']
        else:
            sub['all_key'] = sub['all_key'] + sub['key_' + str(i)]

    sub['keys'] = sub['all_key'].apply(get_most_common).apply(get_top_cnt)
    submit = sub[['id', 'keys']].copy()
    submit['keys'] = submit['keys'].apply(lambda x: np.array(x))
    submit = pd.concat([pd.Series(row['id'], row['keys']) for _, row in submit.iterrows()]).reset_index()
    submit.columns = ['items', 'id']
    submit['entity'] = submit['items'].apply(lambda x: x[0])
    submit['position'] = submit['items'].apply(lambda x: x[1])
    submit = submit[['id', 'position', 'entity']].drop_duplicates()
    submit['entity_type'] = submit['position'].apply(lambda x: get_position(x, 0))
    submit['start_pos'] = submit['position'].apply(lambda x: get_position(x, 1))
    submit['end_pos'] = submit['position'].apply(lambda x: get_position(x, 2))
    submit['res'] = list(
        map(lambda x, y, z: x + " " + str(y) + " " + str(z), submit['entity_type'], submit['start_pos'],
            submit['end_pos']))

    return submit[['id', 'res', 'entity']]