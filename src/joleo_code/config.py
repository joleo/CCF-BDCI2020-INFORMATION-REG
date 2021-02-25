#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/16 22:03
# @Author : aumo
# @Site : 
# @File : config.py
import argparse

# 超参数
parser = argparse.ArgumentParser()
parser.add_argument("--maxlen", default=512, type=int, required=False)
parser.add_argument("--epoch", default=5, type=int, required=False)
parser.add_argument("--batch_size", default=6, type=int, required=False)
parser.add_argument("--learning_rate", default=5e-5, type=float, required=False)
parser.add_argument("--min_learning_rate", default=1e-5, type=float, required=False)
parser.add_argument("--crf_lr", default=5000, type=int, required=False)
parser.add_argument("--fold", default=5, type=int, required=False)
parser.add_argument("--cnt", default=3, type=int, required=False)
parser.add_argument("--model_name", default='nezha', type=str, required=False)
parser.add_argument("--model_path", default='./data/user_data/model_data/nezha_large_', type=str, required=False)
parser.add_argument("--result_path", default='', type=str, required=False)
parser.add_argument("--final_result_path", default='', type=str, required=False)
parser.add_argument("--config_path", default='', type=str, required=False)
parser.add_argument("--checkpoint_path", default='', type=str, required=False)
parser.add_argument("--dict_path", default='', type=str, required=False)
parser.add_argument("--random_seed", default=2020, type=int, required=False)

args = parser.parse_args()
maxlen = args.maxlen
epoch = args.epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
min_learning_rate = args.min_learning_rate
crf_lr = args.crf_lr
n_folds = args.fold
cnt = args.cnt
model_name = args.model_name
model_path = args.model_path
result_path = args.result_path
final_result_path = args.final_result_path
config_path = args.config_path
checkpoint_path = args.checkpoint_path
dict_path = args.dict_path
random_seed = args.random_seed



label_map = ['company', 'organization', 'name', 'address', 'position', 'scene',
             'government', 'game', 'mobile', 'email','movie', 'book', 'QQ', 'vx']

id2label = dict(enumerate(label_map))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(id2label) * 2 + 1