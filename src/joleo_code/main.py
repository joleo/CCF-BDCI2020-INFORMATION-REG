#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/16 22:06
# @Author : XXX
# @Site :
# @File : main.py
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.layers import ConditionalRandomField
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.backend import keras, K
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import KFold
import gc
print('开始')
from postprocessing import _split
from postprocessing import *
from data_processing import *
from config import *
import os
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior
# tf.disable_v2_behavior()
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# 数据加载
all_train = gen_train_data(train_label_path='./data/raw_data/train/label/', train_data_path='./data/raw_data/train/data/')
add_df = gen_new_data(train_path='./data/user_data/cluener/train.json', val_path='./data/user_data/cluener/dev.json')
all_train = all_train + add_df
test = gen_test_data()

test1 = test.copy()
test2 = test.copy()
test3 = test.copy()
test4 = test.copy()
test1['content'] = test1['content'].apply(lambda x: x[0:512])
test2['content'] = test2['content'].apply(lambda x: x[512:1024])
test3['content'] = test3['content'].apply(lambda x: x[1024:1024 + 512])
test4['content'] = test4['content'].apply(lambda x: x[1024 + 512:1024 + 1024])
test1 = test1[test1['content'] != ''].copy()
test2 = test2[test2['content'] != ''].copy()
test3 = test3[test3['content'] != ''].copy()
test4 = test4[test4['content'] != ''].copy()
test1['id'] = test1['id'].apply(lambda x: x + '_1')
test2['id'] = test2['id'].apply(lambda x: x + '_2')
test3['id'] = test3['id'].apply(lambda x: x + '_3')
test4['id'] = test4['id'].apply(lambda x: x + '_4')
test = pd.concat([test1, test2, test3, test4], axis=0).reset_index(drop=True)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1"""
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def viterbi_decode(nodes, trans):
    """Viterbi算法"""
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text, model, CRF):
    """arguments抽取函数 """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    entities, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                entities[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False
    try:
        return [
            (text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities
        ]
    except:
        return {}


def extract_test(text, model, CRF):
    """抽取函数"""
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    entities, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                entities[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False
    try:
        return [
            (text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1],
             l + '_' + str(mapping[w[0]][0]) + '_' + str(mapping[w[-1]][-1] + 1)) for w, l in entities
        ]
    except:
        return []


def evaluate(data, model, CRF):
    """评测函数"""
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(extract_arguments(text, model, CRF))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型 """

    def __init__(self, valid_data, model, CRF, file_path):
        self.best_val_f1 = 0.
        self.valid_data = valid_data
        self.model = model
        self.CRF = CRF
        self.passed = 0
        self.file_path = file_path

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate(self.valid_data, self.model, self.CRF)

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.file_path)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

    def evaluate(self, data, model, CRF):
        """评测函数"""
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in tqdm(data):
            text = ''.join([i[0] for i in d])
            #             print(text)
            R = set(extract_arguments(text, model, CRF))
            #             print(R)
            T = set([tuple(i) for i in d if i[1] != 'O'])
            #             print(T)
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


def get_model(num_labels):

    model = build_transformer_model(config_path, checkpoint_path, model=model_name)
    for layer in model.layers:
        layer.trainable = True
    output = Dense(num_labels)(model.output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr)
    output = CRF(output)

    model = Model(model.input, output)

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learning_rate),
        metrics=[CRF.sparse_accuracy]
    )
    return model, CRF


def train(nfolds, data, test, val, epochs, weight_path):
    """模型训练"""
    skf = KFold(n_splits=nfolds, shuffle=True, random_state=random_seed).split(data)

    for k, (train_fold, test_fold) in enumerate(skf):
        print('--------------Fold---------------------: ', k)
        train_data, valid_data, = list(np.array(data)[train_fold]), list(np.array(data)[test_fold])

        model, CRF = get_model(num_labels)

        file_path = weight_path + str(k) + '_.weights'

        evaluator_val = Evaluator(valid_data, model, CRF, file_path)
        # evaluator_tr = Evaluator(train_data, model, CRF, file_path)
        if not os.path.exists(file_path):
            train_generator = data_generator(train_data, batch_size)
            valid_generator = data_generator(valid_data, batch_size)

            model.fit_generator(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                validation_data=valid_generator.forfit(),
                validation_steps=len(valid_generator),
                epochs=epochs,
                callbacks=[evaluator_val],
                verbose=1
            )
            model.load_weights(file_path)
        else:
            model.load_weights(file_path)
        tqdm.pandas(desc="submit")
        test['submit_' + str(k)] = test['content'].progress_apply(lambda x: extract_test(x, model, CRF))
        del model
        del CRF
        gc.collect()
        K.clear_session()
    return test, val

test_res, val_res = train(n_folds, all_train, test, all_train, epochs=epoch, weight_path=model_path)

test_sub = gen_sub_test()

result = submit(test_res)
result.columns = ['id', 'values', 'entity']
result['ids'] = result['id'].apply(_split)
result['ID'] = result['id'].apply(lambda x: x.split('.')[0]).astype(int)
result['Category'] = result['values'].apply(lambda x: x.split(' ')[0])
result['Pos_b'] = result['values'].apply(lambda x: int(x.split(' ')[1]) - 1)
result['Pos_e'] = result['values'].apply(lambda x: int(x.split(' ')[2]) - 1)
result = result.merge(test_sub, on='ID', how='left')
result['Privacy'] = result['entity']
position = result['ids'].unique()
for i in position:
    if i != position[0]:
        result['Pos_b'] = np.where(result['ids'] == i, result['Pos_b'] + int(i) - int(position[0]), result['Pos_b'])
        result['Pos_e'] = np.where(result['ids'] == i, result['Pos_e'] + int(i) - int(position[0]), result['Pos_e'])

sub_col = ['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy']
result = result[sub_col]
result = result[~result['Privacy'].isnull()]
result = result.sort_values(by='ID')
result.to_csv(result_path, index=False)

