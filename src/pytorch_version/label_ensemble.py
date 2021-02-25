import os
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from glob import glob


def vote(df_list, theta):
    """

    Args:
        df_list: pandas.DataFrame列表
        theta: 投票阈值，保留大于等于该阈值的样本

    Returns:

    """
    collate = OrderedDict()
    if len(df_list) > 0:
        print('Strting voting ')
    else:
        print('file cant not be find')
    for df in tqdm(df_list, desc='统计中..'):
        for idx, row in (df.iterrows()):
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

    new_df = df[df['count'] >= theta].drop('count', axis=1)

    return new_df


def vote_and_output(input_dir, output_name, theta=3):
    # data_dir = [os.path.join(input_dir, dir) for dir in os.listdir(input_dir)]
    data_dir = glob(input_dir + "*/bert/predict.csv")  ##获取所有需要融合的结果文件

    df_list = [pd.read_csv(file, sep=',') for file in data_dir]
    output_df = vote(df_list, theta)  ##投票并返回大于等于theta的结果

    output_df = output_df.sort_values(by=['ID', 'Pos_b'])
    output_df.to_csv(output_name, sep=',', index=False, encoding='utf-8')


def vote_and_output_by_files(data_list, output_name, theta=3):
    # print(data_list)
    df_list = [pd.read_csv(file, sep=',') for file in data_list]
    output_df = vote(df_list, theta)  ##投票并返回大于等于theta的结果

    output_df = output_df.sort_values(by=['ID', 'Pos_b'])
    output_df.to_csv(output_name, sep=',', index=False, encoding='utf-8')


if __name__ == '__main__':
    # ## bert -span crf
    # data_list = [
    #     '../../data/user_data/outputs_5fold_mix_span_bert2/fold_0/bert/post2_predict.csv',  #
    #     '../../data/user_data/outputs_5fold_mix_span_bert2/fold_1/bert/post2_predict.csv',  #
    #     '../../data/user_data/outputs_5fold_mix_span_bert2/fold_2/bert/post2_predict.csv',  #
    #     '../../data/user_data/outputs_5fold_mix_span_bert2/fold_3/bert/post2_predict.csv',  #
    #     '../../data/user_data/outputs_5fold_mix_span_bert2/fold_4/bert/post2_predict.csv',  #
    # ]
    # vote_and_output_by_files(
    #     data_list=data_list,
    #     output_name='../../data/user_data/outputs_5fold_mix_span_bert2/5post_predict.csv',  #
    #     theta=5)

    # data_list = [
    #     '../../data/user_data/models/3-3model/blending_crf_result.csv',  #
    #     '../../data/user_data/models/3-3model/post-5-3_predict.csv',  #
    #     '../../data/user_data/models/3-3model/orig_rob5v3.csv',  #
    # ]
    data_list2 = [
        './data/user_data/models/3-3model/blending_crf_result.csv',  #
        './data/user_data/models/outputs_5fold_mix_crf_robert/5-3_predict.csv',  #
        './data/user_data/models/outputs_5fold_mix_crf_macbert/5-3_predict.csv',  #
    ]
    vote_and_output_by_files(
        data_list=data_list2,
        output_name='./data/user_data/models/3-3model/new_3-3vote.csv',  #
        theta=3)
else:
    pass
