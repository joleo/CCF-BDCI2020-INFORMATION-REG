# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/25 20:01
@Auth ： aumo
@File ：bert_embedding.py
"""
from bert4keras.models import build_transformer_model
from keras.models import Model
from keras.layers import Add, Dropout
import tensorflow as tf

graph = None
model = None

class KerasBertEmbedding():
    def __init__(self, config_name, ckpt_name, dict_path, layer_num, bert_name,layer_indexes=[]):
        self.config_path = config_name
        self.checkpoint_path = ckpt_name
        self.dict_path = dict_path
        # self.max_seq_len = max_seq_len
        self.layer_num = layer_num
        self.layer_indexes = layer_indexes
        self.bert_name = bert_name

    def bert_encode(self):
        global graph
        graph = tf.get_default_graph()
        global model
        # model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=200)
        model = build_transformer_model(self.config_path, self.checkpoint_path, self.bert_name,
                                        with_pool=False, return_keras_model=False)

        # for l in model.layers:
        #     l.trainable = True

        print(model.output)
        print(len(model.layers))
        layer_dict = [7]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        elif len(self.layer_indexes) == 1:  # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
            if self.layer_indexes[0] in [i + 1 for i in range(self.layer_num)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0] - 1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        else:
            all_layers = [
                model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(self.layer_num)]
                else model.get_layer(index=layer_dict[-1]).output  # 如果给出不正确，就默认输出最后一层
                for lay in self.layer_indexes]
            print(self.layer_indexes)
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)
        print("KerasBertEmbedding:")
        print(encoder_layer.shape)
        model = Model(model.inputs, encoder_layer)
        # model.summary(120)
        return model.inputs, model.output


