import os
import pandas as pd
import datacompy
import pickle
from pprint import pprint
import json
import re
import shutil


def text_clean(text):
    return text.replace(',', '，')


def split_sents(text):
    sentences = re.split(r"([,，。 ！？?])", text)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    if len(sentences[-1]) == 0:  #
        sentences = sentences[:-1]
    return sentences


def jdata_character_clean(jdata):
    jdata_clean = {}
    jdata_clean['text'] = text_clean(jdata['text'])
    jdata_clean['label'] = {}
    for ca, ps in jdata['label'].items():
        jdata_clean['label'][ca] = {}
        for p, ens in ps.items():
            p = text_clean(p)
            jdata_clean['label'][ca][p] = []
            for en in ens:
                jdata_clean['label'][ca][p].append([en[0], en[1]])
    return jdata_clean


def adjust_label_offset(sent, labels, offset):
    '''
    根据偏移修正id
    '''
    data_map = {}
    data_map['text'] = sent
    data_map['label'] = {}
    for category, privacys in labels.items():
        for privacy, indlices in privacys.items():
            for indlice in indlices:
                index = sent.find(privacy)
                if index != -1 and (indlice[0] >= offset) and (indlice[1] < len(sent) + offset):
                    if category not in data_map['label']:
                        data_map['label'][category] = {}
                    if privacy not in data_map['label'][category]:
                        data_map['label'][category][privacy] = []
                    data_map['label'][category][privacy].append([indlice[0] - offset, indlice[1] - offset])
    return data_map


def get_stride_train(D, window_size=512):
    all_data = []
    change_count = 0
    for data in D:
        sents = split_sents(data['text'])
        sub_len = 0
        #  check
        for sent in sents:
            sub_len += len(sent)
        assert sub_len == len(data['text'])
        # operate
        labels = data['label']
        if len(data['text']) < window_size:
            all_data.append(data)
        else:
            change_count += 1
            s_id = 0
            s_id_max = len(sents)
            offset = 0
            while True:
                cur_sent = ''
                # 拼接
                while s_id < s_id_max and len(sents[s_id]) + len(cur_sent) < window_size:
                    cur_sent += sents[s_id]
                    s_id += 1
                data_map = adjust_label_offset(cur_sent, labels, offset)
                offset += len(cur_sent)
                all_data.append(data_map)
                if s_id >= s_id_max:
                    break
    print(change_count)
    return all_data


def get_add_data2(base_dir='./data'):
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


def data_character_clean(line):
    jdata = json.loads(line)
    jdata_clean = {}
    jdata_clean['text'] = text_clean(jdata['text'])
    jdata_clean['label'] = {}
    for ca, ps in jdata['label'].items():
        jdata_clean['label'][ca] = {}
        for p, ens in ps.items():
            p = text_clean(p)
            jdata_clean['label'][ca][p] = []
            for en in ens:
                jdata_clean['label'][ca][p].append([en[0], en[1]])
    return jdata_clean


def train_window_slide_combine_data():
    add_data_list = []
    add_data = get_add_data2()
    for i, line in enumerate(add_data):
        data = data_character_clean(line)
        add_data_list.append(data)
    return add_data_list


def adjust_sents_offset(datas, id):
    data_map = {'id': id, 'text': '', 'label': {}}
    offset = 0
    for data in datas:
        data_map['text'] += data['text']
        for category, privacys in data['label'].items():
            if category not in data_map['label']:
                data_map['label'][category] = {}
            for privacy, indlices in privacys.items():
                if privacy not in data_map['label'][category]:
                    data_map['label'][category][privacy] = []
                for indlice in indlices:
                    data_map['label'][category][privacy].append([indlice[0] + offset, indlice[1] + offset])
        offset += len(data['text'])
    return data_map


def create_stride_test(data_dir='./data/user_data/test', out_dir='./data/user_data/test'):
    data_list = []
    window_size = 512
    old2newid = {}
    newid2old = {}
    newidoffset = {}
    change_count = 0
    input_test_file = os.path.join(data_dir, 'goldtest.json')
    output_test_file = os.path.join(out_dir, 'test.json')
    output_test_dict_file = os.path.join(out_dir, 'test_dict.pkl')

    with open(input_test_file, 'r') as f:
        count = 0
        for line in f:
            jdata = json.loads(line)
            text = jdata['text']
            id = jdata['id']
            sents = split_sents(text)
            sub_len = 0
            # 检测拆分
            for sent in sents:
                sub_len += len(sent)
            assert sub_len == len(jdata['text'])

            offset = 0
            old2newid[id] = []
            # 长度符合要求
            if len(jdata['text']) < window_size:
                sub_data = {'id': count, 'text': text}
                old2newid[id] = [count]
                newid2old[count] = id
                newidoffset[count] = offset
                count += 1
                data_list.append(json.dumps(sub_data, ensure_ascii=False) + '\n')
            else:
                change_count += 1
                s_id = 0
                s_id_max = len(sents)
                offset = 0
                while True:
                    cur_sent = ''
                    # 拼接自居
                    while s_id < s_id_max and len(sents[s_id]) + len(cur_sent) < window_size:
                        cur_sent += sents[s_id]
                        s_id += 1
                    # 生成自居
                    sub_data = {'id': count, 'text': cur_sent}
                    old2newid[id].append(count)
                    newid2old[count] = id
                    newidoffset[count] = offset
                    count += 1
                    data_list.append(json.dumps(sub_data, ensure_ascii=False) + '\n')
                    offset += len(cur_sent)

                    if s_id >= s_id_max:
                        break
    # print(f'改变原文数  {change_count} 记录数 {len(data_list)}')
    with open(output_test_file, 'w') as f:
        f.writelines(data_list)
    origin_offset = restore_test_window_slide_offset(input_test_file, train_window_slide_combine_data())
    test_dict = (newid2old, newidoffset, origin_offset)
    pickle.dump(test_dict, open(output_test_dict_file, 'wb'))
    for i in range(5):
        source_file = output_test_file
        target_file = os.path.join('./data/user_data/', f'5fold_mix/fold_{i}/test.json')
        shutil.copyfile(source_file, target_file)
        source_file = output_test_dict_file
        target_file = os.path.join('./data/user_data/', f'5fold_mix/fold_{i}/test_dict.pkl')
        shutil.copyfile(source_file, target_file)


def post_process_raw(data_dir, result_dir):
    ids = []
    categorys = []
    posbs = []
    poses = []
    privacys = []
    test_submit_file = os.path.join(result_dir, 'test_submit.json')
    final_predict_file = os.path.join(result_dir, 'raw_predict.csv')
    test_dict_file = os.path.join(data_dir, 'test_dict.pkl')
    (newid2old, newidoffset, origin_offset) = pickle.load(open(test_dict_file, 'rb'))
    with open(test_submit_file, 'r') as f:
        for line in f:
            jdata = json.loads(line)
            newid = jdata['id']
            oldid = newid2old[newid]
            offset = newidoffset[newid]
            jdata = origin_offset[oldid] if oldid in origin_offset else jdata
            offset = 0 if oldid in origin_offset else offset
            for label, entitys in jdata['label'].items():
                for privacy, spans in entitys.items():
                    for span in spans:
                        if len(privacy) == 0 or span[1] == -1 or span[0] == -1 or '\n' in privacy:
                            continue

                        ids.append(oldid)
                        categorys.append(label)
                        posbs.append(span[0] + offset)
                        poses.append(span[1] + offset)
                        privacys.append(privacy)
    pd_map = {'ID': ids, "Category": categorys, "Pos_b": posbs, "Pos_e": poses, 'Privacy': privacys}
    df = pd.DataFrame(pd_map)
    df.drop_duplicates(keep='first', inplace=True)
    df = df.sort_values(by=['ID', 'Pos_b'])
    # print('test id restore done !')
    df.to_csv(final_predict_file, index=False, encoding='utf-8')


def compare_two_csv(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    compare = datacompy.Compare(
        df1=df1,
        df2=df2,
        join_columns=['ID', 'Category', 'Pos_b', 'Pos_e'],
        abs_tol=0.00001
    )
    result = {
        'df1_size': len(compare.df1),
        'df2_size': len(compare.df2),
        'df1_unq_rows': len(compare.df1_unq_rows),
        'df2_unq_rows': len(compare.df2_unq_rows),
        'equal_rows': len(compare.df1) - len(compare.df1_unq_rows),
    }
    pprint(result)
    return compare


def get_index(sub_df):
    index = sub_df.groupby('Pos_e').apply(
        lambda d: d.index.tolist() if len(d.index) > 1 else None
    ).dropna()
    indexs = index.to_list()[0]
    l = sub_df.Pos_b[indexs[0]]
    r = sub_df.Pos_e[indexs[0]]
    for index in indexs:
        l = min(l, sub_df.Pos_b[index])
    return indexs, [l, r]


def get_filter_by_min_len(df, indexs):
    '''
    返回除最短实体外的实体index，用于删除
    '''
    min_i = indexs[0]
    min_len = indexs[0]
    for index in indexs:
        len_ = len(df.Privacy.loc[index])
        if len_ < min_len:
            min_i = index
            min_len = len_
    indexs.remove(min_i)
    return indexs


def restore_test_window_slide_offset(comp_test_file, new_test_data_list):
    id2test_data = {}
    test_offset_r3 = {}
    with open(comp_test_file, 'r') as f:
        for line in f:
            jdata = json.loads(line)
            text = jdata['text']
            id2test_data[jdata['id']] = jdata
            for i, a_data in enumerate(new_test_data_list):
                if text == a_data['text']:
                    a_data['id'] = jdata['id']
                    test_offset_r3[jdata['id']] = a_data
                elif a_data['text'] in text:
                    datas = []
                    if new_test_data_list[i + 1]['text'] in text:
                        j = i
                        datas.append(new_test_data_list[j])
                        while j + 1 < len(new_test_data_list) and new_test_data_list[j + 1]['text'] in text:
                            j += 1
                            datas.append(new_test_data_list[j])
                        test_offset_r3[jdata['id']] = adjust_sents_offset(datas, jdata['id'])
                        break
    return test_offset_r3


def overlap_id_check(df):
    start_map = {}
    end_map = {}
    count_start = 0
    count_end = 0
    df2 = df.copy()
    error_case_ids = set()
    for idx, row in df2.iterrows():
        ID = row['ID']
        if ID not in start_map:
            start_map[ID] = set()
        if ID not in end_map:
            end_map[ID] = set()
        if row['Pos_b'] in start_map[ID]:
            count_start += 1
        if row['Pos_e'] in end_map[ID]:
            count_end += 1
            error_case_ids.add(ID)
        start_map[ID].add(row['Pos_b'])
        end_map[ID].add(row['Pos_e'])
    return error_case_ids, count_start, count_end


def post2(span_predict_file, crf_file, output_file=None):
    span_df = pd.read_csv(span_predict_file)
    crf_df = pd.read_csv(crf_file)
    error_case_ids, count_start, count_end = overlap_id_check(span_df)
    result = {'origin_size': len(span_df)}
    add_df = pd.DataFrame(columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    delect_indexs = []
    crf_delect_indexs = []
    remain = 0

    for ID in error_case_ids:
        indexs, span = get_index(span_df[span_df.ID == ID])
        ctg = span_df.Category[indexs[0]]
        temp = crf_df[
            (crf_df.ID == ID) & (crf_df.Category == ctg) & (crf_df.Pos_e <= span[1]) & (crf_df.Pos_e >= span[0])]
        if len(temp) == 0:
            # 保留最短实体
            remain += 1
            delect_indexs.extend(get_filter_by_min_len(span_df, indexs))
        else:
            # 根据crf中的结果对span方式进行直接替换
            add_df = add_df.append(temp)
            crf_delect_indexs.extend(indexs)

    delete_crf_df = span_df.iloc[crf_delect_indexs]
    delete_sgl_df = span_df.iloc[delect_indexs]
    delect_indexs.extend(crf_delect_indexs)
    # delete_df = span_df.iloc[delect_indexs]
    delete_df = delete_crf_df.append(delete_sgl_df)
    # span_df = span_df.drop(delect_indexs)
    span_df = span_df.drop(delete_df.index)
    span_df = span_df.append(add_df)
    span_df.reset_index(drop=True, inplace=True)
    span_df = span_df.sort_values(by=['ID', 'Pos_b'])

    if output_file:
        span_df.to_csv(output_file, index=False)
        print('saving post2  result')
    return span_df


def post4(span_predict_file, crf_file, output_file=None):
    span_df = pd.read_csv(span_predict_file)
    result = {'origin_size': len(span_df)}
    crf_df = pd.read_csv(crf_file)
    error_case_ids, count_start, count_end = overlap_id_check(span_df)

    add_df = pd.DataFrame(columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    delect_indexs = []
    crf_delect_indexs = []
    remain = 0

    # 解决重叠实体，方案为crf填充，结合保留最短实体
    for ID in error_case_ids:
        indexs, span = get_index(span_df[span_df.ID == ID])
        ctg = span_df.Category[indexs[0]]
        temp = crf_df[
            (crf_df.ID == ID) & (crf_df.Category == ctg) & (crf_df.Pos_e <= span[1]) & (crf_df.Pos_e >= span[0])]
        if len(temp) == 0:
            # 保留最短实体
            remain += 1
            delect_indexs.extend(get_filter_by_min_len(span_df, indexs))
        # else:
        #     根据crf中的结果对span方式进行直接替换
        # add_df = add_df.append(temp)
        # crf_delect_indexs.extend(indexs)

    # delete_crf_df = span_df.iloc[crf_delect_indexs]
    # delete_sgl_df = span_df.iloc[delect_indexs]
    # delect_indexs.extend(crf_delect_indexs)
    # delete_df = delete_crf_df.append(delete_sgl_df)
    delete_df = span_df.iloc[delect_indexs]
    # 去除实体
    span_df = span_df.drop(delete_df.index)
    # 增加crf实体替换
    # span_df = span_df.append(add_df)
    span_df.reset_index(drop=True, inplace=True)
    span_df = span_df.sort_values(by=['ID', 'Pos_b'])

    compare = datacompy.Compare(
        df1=delete_df,
        df2=add_df,
        join_columns=['ID', 'Category', 'Pos_b', 'Pos_e'],
        abs_tol=0.00001
    )
    result['crf_add'] = len(compare.df2_unq_rows)
    result['min_delete'] = len(compare.df1_unq_rows)

    # # 处理标点符号问题,暂时直接去除
    ch_df = chararcter_check(span_df)
    span_df = span_df.drop(ch_df.index)
    span_df.reset_index(drop=True, inplace=True)
    result['ch_delete'] = len(ch_df)

    # 空缺id检测
    forget_ID = check_null_answer_IDs(span_df)
    null_df = crf_df[crf_df.ID.isin(forget_ID)]
    result['null_ID'] = len(forget_ID)
    result['null_ID_entity'] = len(null_df)
    result['cur_size'] = len(span_df)

    # print(compare.report())
    # diff = get_df1_sub_df2(delete_df,add_df,['ID', 'Category', 'Pos_b', 'Pos_e'])
    if output_file:
        span_df.to_csv(output_file, index=False)
        print('writted  post2 result')
    # return delete_df, add_df, compare, span_df, delete_sgl_df, delete_crf_df, result
    return delete_df, None, compare, span_df, None, None, result


def post1(span_predict_file, crf_file, output_file=None):
    span_df = pd.read_csv(span_predict_file)
    crf_df = pd.read_csv(crf_file)
    error_case_ids, count_start, count_end = overlap_id_check(span_df)
    result = {'origin_size': len(span_df)}
    add_df = pd.DataFrame(columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    delect_indexs = []
    crf_delect_indexs = []
    remain = 0

    for ID in error_case_ids:
        indexs, span = get_index(span_df[span_df.ID == ID])
        ctg = span_df.Category[indexs[0]]
        temp = crf_df[
            (crf_df.ID == ID) & (crf_df.Category == ctg) & (crf_df.Pos_e <= span[1]) & (crf_df.Pos_e >= span[0])]
        if len(temp) == 0:
            # 保留最短实体
            remain += 1
            delect_indexs.extend(get_filter_by_min_len(span_df, indexs))

    # print(
    #     f"实体重叠数 {count_end} | 删除实体数 {len(delect_indexs)}| crf 补充实体数 {len(add_df)} |  crf 删除实体数 {len(crf_delect_indexs)} |         总删除数 {len(add_df) - len(delect_indexs) - len(crf_delect_indexs)} ")

    delete_crf_df = span_df.iloc[crf_delect_indexs]
    delete_sgl_df = span_df.iloc[delect_indexs]
    delect_indexs.extend(crf_delect_indexs)
    delete_df = span_df.iloc[delect_indexs]
    # delete_df = delete_crf_df.append(delete_sgl_df)
    # span_df = span_df.drop(delect_indexs)
    span_df = span_df.drop(delete_df.index)
    span_df = span_df.append(add_df)
    span_df.reset_index(drop=True, inplace=True)
    span_df = span_df.sort_values(by=['ID', 'Pos_b'])

    compare = datacompy.Compare(
        df1=delete_df,
        df2=add_df,
        join_columns=['ID', 'Category', 'Pos_b', 'Pos_e'],
        abs_tol=0.00001
    )
    print(compare.report())
    if output_file:
        span_df.to_csv(output_file, index=False)
        print('saving  post1 result')
    return delete_df, add_df, compare, span_df, delete_sgl_df, delete_crf_df, result


def post0(span_predict_file, output_file=None):
    span_df = pd.read_csv(span_predict_file)

    result = {'origin_size': len(span_df)}

    # # 处理标点符号问题,暂时直接去除
    ch_df = chararcter_check(span_df)
    result['ch_delete'] = len(ch_df)
    span_df = span_df.drop(ch_df.index)
    span_df.reset_index(drop=True, inplace=True)
    span_df = span_df.sort_values(by=['ID', 'Pos_b'])

    if output_file:
        span_df.to_csv(output_file, index=False)
        print('saving post0 result')
    return span_df


def filter_ch(x):
    if '《' == x[0] or '》' == x[-1]:
        return False
    elif '，' in x:
        return True
    if '。' in x:
        return True
    # elif '、' in x:
    #     return True
    elif '；' in x:
        return True
    return False


def get_df1_sub_df2(df1, df2, columns):
    diff_df = df1.copy()
    diff_df = diff_df.append(df2)
    diff_df = diff_df.append(df2)
    diff_df = diff_df.drop_duplicates(subset=columns, keep=False)
    return diff_df


def get_add_data():
    train_path = './data/user_data/cluener/train.json'
    dev_path = './data/user_data/cluener/dev.json'
    all_cluener = []
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            all_cluener.append(line)
    with open(dev_path, 'r') as f:
        for i, line in enumerate(f):
            all_cluener.append(line)
    return all_cluener


def get_slide_train(D, window_size=512):
    all_data = []
    change_count = 0
    for data in D:
        sents = split_sents(data['text'])
        sub_len = 0
        #  check
        for sent in sents:
            sub_len += len(sent)
        assert sub_len == len(data['text'])
        # operate
        labels = data['label']
        if len(data['text']) < window_size:
            all_data.append(data)
        else:
            change_count += 1
            s_id = 0
            s_id_max = len(sents)
            offset = 0
            while True:
                cur_sent = ''
                # 拼接
                while s_id < s_id_max and len(sents[s_id]) + len(cur_sent) < window_size:
                    cur_sent += sents[s_id]
                    s_id += 1
                data_map = adjust_label_offset(cur_sent, labels, offset)
                offset += len(cur_sent)
                all_data.append(data_map)
                if s_id >= s_id_max:
                    break
    print(change_count)
    return all_data


def check_long_entity(df):
    sub_df = df[df.apply(lambda x: x['Category'] == 'position' and len(x['Privacy']) > 30, axis=1)]
    return sub_df


def chararcter_check(df):
    sub_df = df[df.Privacy.map(filter_ch)]
    return sub_df


def check_null_answer_IDs(df):
    '''
    针对空的ID
    :param df:
    :return:
    '''
    whiole_IDs = set([i for i in range(3956)])
    return whiole_IDs - set(df.ID.tolist())


def pad_unfind_ID(df, crf_df):
    forget_ID = check_null_answer_IDs(df)
    return crf_df[crf_df.ID.isin(forget_ID)]


if __name__ == '__main__':
    comp = compare_two_csv(
        '../../data/user_data/models/submitv2_zy_category_1128_fill_leakv3_crf_high_model_4_3v2_multi.csv',
        '../../data/prediction_result/result.csv')
else:
    pass
