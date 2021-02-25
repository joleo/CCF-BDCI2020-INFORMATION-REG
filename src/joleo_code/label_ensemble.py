import os
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
from config import *

def vote(df_list, theta):
    """

    Args:
        df_list: pandas.DataFrame列表
        theta: 投票阈值，保留大于等于该阈值的样本

    Returns:

    """
    collate = OrderedDict()
    for df in tqdm(df_list, desc='统计中..'):
        for idx, row in tqdm(df.iterrows(), desc='一行行统计中...'):
            id = row['ID']
            label = '$#$^$&$'.join([row['Category'], str(row['Pos_b']), str(row['Pos_e']), row['Privacy']])
            collate[id] = collate.get(id, dict())
            collate[id][label] = collate[id].get(label, 0)
            collate[id][label] += 1

    ids = []
    types = []
    pos_bs = []
    pos_es = []
    entities = []
    counts = []

    for id in collate.keys():
        for label in collate[id].keys():
            type, pos_b, pos_e, entity = label.split('$#$^$&$')
            count = collate[id][label]

            ids.append(id)
            types.append(type)
            pos_bs.append(pos_b)
            pos_es.append(pos_e)
            entities.append(entity)
            counts.append(count)

    df = pd.DataFrame(
        {'ID': ids, 'Category': types, 'Pos_b': pos_bs, 'Pos_e': pos_es, 'Privacy': entities, 'count': counts})
    print(df)

    # print(three_df)

    new_df = df[df['count'] >= theta].drop('count', axis=1)
    print(new_df)

    return new_df

def vote_and_output(input_dir, output_name, theta=3):
    # data_dir = [os.path.join(input_dir, dir) for dir in os.listdir(input_dir)]
    # data_dir = glob(input_dir + "*/bert/predict.csv")  ##获取所有需要融合的结果文件
    # data_dir = ['data/sumit/bert_ner.csv','data/sumit/submission_1119_v1.csv', 'data/sumit/submission_1125_v1.csv','data/sumit/bert_base_span_other_submit_predict.csv']


    df_list = [pd.read_csv(file, sep=',') for file in input_dir]
    output_df = vote(df_list, theta)  ##投票并返回大于等于theta的结果

    output_df.to_csv(output_name, sep=',', index=False, encoding='utf-8')

def vote_and_output2(input_dir, output_name, theta=3):
    # data_dir = [os.path.join(input_dir, dir) for dir in os.listdir(input_dir)]
    data_dir = glob(input_dir + "*/bert/predict.csv")  ##获取所有需要融合的结果文件

    df_list = [pd.read_csv(file, sep=',') for file in data_dir]
    output_df = vote(df_list, theta)  ##投票并返回大于等于theta的结果

    output_df = output_df.sort_values(by=['ID', 'Pos_b'])
    output_df.to_csv(output_name, sep=',', index=False, encoding='utf-8')

if __name__ == '__main__':

    data_dir = [
        './data/user_data/models/nezha_large_1211.csv'
        , './data/user_data/models/nezha_base_1211.csv'
        , './data/user_data/models/roberta_large_1211.csv'
    ]
    vote_and_output(input_dir=data_dir, output_name=result_path, theta=2)
