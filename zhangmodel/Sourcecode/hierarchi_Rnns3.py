#coding=utf-8
'''
Created on 2017.5.3

@author: DUTIRLAB
'''

import pickle as pkl
import gzip
import sys
import time
import os

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Embedding
from keras.layers.core import Reshape, Merge
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

np.random.seed(345)

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = 'floatX=float32,device=cuda,gcc.cxxflags=-Wno-narrowing'

#evaluation of DDI extraction results. 4 DDI tpyes
def result_evaluation(y_test,pred_test):

    pred_matrix = np.zeros_like(pred_test, dtype=np.int8)

    y_matrix = np.zeros_like(y_test, dtype=np.int8)

    pred_indexs = np.argmax(pred_test, 1)
    pred_indexs = np.clip(pred_indexs, 3, 4)
    y_indexs = np.argmax(y_test, 1)
    y_indexs = np.clip(y_indexs, 3, 4)
    for i in range(len(pred_indexs)):
        pred_matrix[i][pred_indexs[i]] = 1
        y_matrix[i][y_indexs[i]] = 1


    count_matrix=np.zeros((4,3))
    for class_idx in range(4):
        count_matrix[class_idx][0] = np.sum(np.array(pred_matrix[:, class_idx]) * np.array(y_matrix[:, class_idx]))#tp
        count_matrix[class_idx][1] = np.sum(np.array(pred_matrix[:, class_idx]) * (1 - np.array(y_matrix[:, class_idx])))#fp
        count_matrix[class_idx][2] = np.sum((1 - np.array(pred_matrix[:, class_idx])) * np.array(y_matrix[:, class_idx]))#fn

    sumtp=sumfp=sumfn=0

    for i in range(4):
        sumtp+=count_matrix[i][0]
        sumfp+=count_matrix[i][1]
        sumfn+=count_matrix[i][2]

        precision=recall=f1=0

    if (sumtp + sumfp) == 0:
        precision = 0.
    else:
        precision = float(sumtp) / (sumtp + sumfp)

    if (sumtp + sumfn) == 0:
        recall = 0.
    else:
        recall = float(sumtp) / (sumtp + sumfn)

    if (precision + recall) == 0.:
        f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision,recall,f1

#embedding entities attention
class emb_AttentionLayer(Layer):
    def __init__(self, **kwargs):

        super(emb_AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input shape", input_shape)
        self.scaler=input_shape[0][1]
        print("scaler", (self.scaler))

        super(emb_AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):

        sentence = x[0]
        entity_0=x[1]
        entity_1=x[2]

        eij_0=K.batch_dot(sentence,entity_0,axes=[2,2])
        K.sqrt(sentence)
        K.sqrt(entity_0)
        eij_0/=2000
        print("shapes", eij_0.shape[0], eij_0.shape[1], x[0].shape[0], x[0].shape[1])
        eij_0=K.reshape(eij_0,(x[0].shape[0],x[0].shape[1]))
        ai_0 = K.exp(eij_0)
        weights_0=ai_0 / K.sum(ai_0, axis=1).dimshuffle(0, 'x')

        eij_1 = K.batch_dot(sentence, entity_1, axes=[2, 2])
        eij_1 /= 2000
        eij_1 = K.reshape(eij_1, (x[0].shape[0], x[0].shape[1]))
        ai_1 = K.exp(eij_1)
        weights_1 = ai_1 / K.sum(ai_1, axis=1).dimshuffle(0, 'x')


        weights=((weights_0+weights_1)/2.0)*self.scaler

        weighted_input = x[3] * weights.dimshuffle(0, 1, 'x')

        return weighted_input

    def get_output_shape_for(self, input_shape):
        return (input_shape[3])


def load_data(s):
    f_Train = gzip.open(s['train_file'], 'rb')

    train_inputs = pkl.load(f_Train)
    train_labels_vec = to_categorical(train_inputs["train_labels_vec"].astype(int))
    train_word = train_inputs["train_word"]
    train_POS = train_inputs["train_POS"]
    train_distances = train_inputs["train_distances"]
    train_distances2 = train_inputs["train_distances2"]

    train_shortest_word = train_inputs["train_shortest_word"]
    train_shortest_pos = train_inputs["train_shortest_pos"]
    train_shortest_dis1 = train_inputs["train_shortest_dis1"]
    train_shortest_dis2 = train_inputs["train_shortest_dis2"]
    train_entity = train_inputs["train_entity"]
    if s["use_ontology"]:
        train_common_ancestors = train_inputs["train_ancestors"]
        train_concat_ancestors = train_inputs["train_subpaths"]

    f_Train.close()

    f_Test = gzip.open(s['test_file'], 'rb')
    test_inputs = pkl.load(f_Test)
    test_labels_vec = to_categorical(test_inputs["train_labels_vec"].astype(int))
    test_word = test_inputs["train_word"]
    test_POS = test_inputs["train_POS"]
    test_distances = test_inputs["train_distances"]
    test_distances2 = test_inputs["train_distances2"]

    test_shortest_word = test_inputs["train_shortest_word"]
    test_shortest_pos = test_inputs["train_shortest_pos"]
    test_shortest_dis1 = test_inputs["train_shortest_dis1"]
    test_shortest_dis2 = test_inputs["train_shortest_dis2"]

    test_entity = test_inputs["train_entity"]
    if s["use_ontology"]:
        test_common_ancestors = test_inputs["train_ancestors"]
        test_concat_ancestors = test_inputs["train_subpaths"]
    f_Test.close()

    f_word2vec = gzip.open(s['wordvecfile'], 'rb')
    vec_table = pkl.load(f_word2vec, encoding='bytes')
    pos_vec_table = pkl.load(f_word2vec, encoding='bytes')
    dis_vec_table = pkl.load(f_word2vec, encoding='bytes')
    if s["use_ontology"]:
        onto_vec_table = pkl.load(f_word2vec, encoding='bytes')
    f_word2vec.close()

    inputs = {"train_labels_vec": train_labels_vec,
              "train_entity": train_entity,
              "train_word": train_word,
              "train_POS": train_POS,
              "train_distances": train_distances,
              "train_distances2":train_distances2,
              "train_shortest_word": train_shortest_word,
              "train_shortest_pos": train_shortest_pos,
              "train_shortest_dis1": train_shortest_dis1,
              "train_shortest_dis2": train_shortest_dis2,
              "test_labels_vec": test_labels_vec,
              "test_entity": test_entity,
              "test_word": test_word,
              "test_POS": test_POS,
              "test_distances": test_distances,
              "test_distances2": test_distances2,
              "test_shortest_word": test_shortest_word,
              "test_shortest_pos": test_shortest_pos,
              "test_shortest_dis1": test_shortest_dis1,
              "test_shortest_dis2": test_shortest_dis2,
              "vec_table": vec_table,
              "pos_vec_table": pos_vec_table,
              "dis_vec_table": dis_vec_table}

    if s["use_ontology"]:
              inputs.update({"train_common_ancestors": train_common_ancestors,
              "train_concat_ancestors": train_concat_ancestors,
              "test_common_ancestors": test_common_ancestors,
              "test_concat_ancestors": test_concat_ancestors,
               "onto_vec_table": onto_vec_table
              })

    return inputs

def main():

    s = {
         'emb_dimension':300, # dimension of word embedding
         'mini_batch':32,
         'shortest_part_length':12,
         'epochs':100,
         'class_type':5,
        'first_hidden_layer':150,
        'lr':0.001,
        'emb_dropout':0.7,
        'dense_dropout':0.5,
        'train_file': "../data/train.pkl.gz",
        'test_file': "../data/test.pkl.gz",
        'wordvecfile': "../data/vec.pkl.gz",
        'use_ontology': True

        }


    outpath = '../data/'
    nb_class=s['class_type']

    #read train and test data which are pkl files
    inputs = load_data(s)
    print(len(inputs["train_labels_vec"]))
    print(len(inputs["test_labels_vec"]))
    print(len(inputs["vec_table"]))
    if s["use_ontology"]:
        print(len(inputs["onto_vec_table"]))
    #print test_labels_vec

    answer_y = np.array(inputs["test_labels_vec"], dtype=np.int8)
    train_y = np.array(inputs["train_labels_vec"], dtype=np.int8)
    test_word_bak=inputs["test_word"]

    word_dep_max = 0
    word_max_num = 0
    temp_max = 0


    new_train_word=[]
    new_train_pos=[]
    new_train_dis=[]
    new_train_dis2=[]
    new_test_word=[]
    new_test_pos=[]
    new_test_dis=[]
    new_test_dis2=[]

    word_length = []

    #calculate the max length of each subsequence
    for i in range(3):

        temp_max = 0
        temp_train_word = []
        temp_train_pos = []
        temp_train_dis = []
        temp_train_dis2 = []
        for j in range(len(inputs["train_word"])):
            #print(len(train_word[j][0]), len(train_word[j][1]), len(train_word[j][2]))
            #print(i, j, train_word[j][i], train_POS[j][i], train_distances[j][i], train_distances2[j][i])
            assert len(inputs["train_word"][j][i])==len(inputs["train_POS"][j][i])
            assert len(inputs["train_POS"][j][i])==len(inputs["train_distances"][j][i])
            assert len(inputs["train_distances"][j][i])==len(inputs["train_distances2"][j][i])
            temp_train_word.append(inputs["train_word"][j][i])
            temp_train_pos.append(inputs["train_POS"][j][i])
            temp_train_dis.append(inputs["train_distances"][j][i])
            temp_train_dis2.append(inputs["train_distances2"][j][i])
            if len(inputs["train_word"][j][i]) > temp_max:
                temp_max = len(inputs["train_word"][j][i])

        new_train_word.append(temp_train_word)
        new_train_pos.append(temp_train_pos)
        new_train_dis.append(temp_train_dis)
        new_train_dis2.append(temp_train_dis2)

        temp_test_word = []
        temp_test_pos = []
        temp_test_dis = []
        temp_test_dis2 = []
        for j in range(len(inputs["test_word"])):
            assert len(inputs["test_word"][j][i]) == len(inputs["test_POS"][j][i])
            assert len(inputs["test_POS"][j][i]) == len(inputs["test_distances"][j][i])
            assert len(inputs["test_distances"][j][i]) == len(inputs["test_distances2"][j][i])
            temp_test_word.append(inputs["test_word"][j][i])
            temp_test_pos.append(inputs["test_POS"][j][i])
            temp_test_dis.append(inputs["test_distances"][j][i])
            temp_test_dis2.append(inputs["test_distances2"][j][i])
            if len(inputs["test_word"][j][i]) > temp_max:
                temp_max = len(inputs["test_word"][j][i])

        new_test_word.append(temp_test_word)
        new_test_pos.append(temp_test_pos)
        new_test_dis.append(temp_test_dis)
        new_test_dis2.append(temp_test_dis2)
        word_length.append(temp_max)

    train_word = new_train_word

    train_POS=new_train_pos
    train_distances=new_train_dis
    train_distances2=new_train_dis2

    test_word = new_test_word

    test_POS=new_test_pos
    test_distances=new_test_dis
    test_distances2=new_test_dis2

    train_entity_0=[]
    train_entity_1=[]
    test_entity_0=[]
    test_entity_1=[]

    for i in range(len(inputs["train_entity"])):
        #temp_list=[]
        #temp_list.append()
        train_entity_0.append([inputs["train_entity"][i][0]])
        train_entity_1.append([inputs["train_entity"][i][1]])

    for i in range(len(inputs["test_entity"])):
        temp_list = []
        temp_list.append(inputs["test_entity"][i][0])
        test_entity_0.append(temp_list)
        temp_list = []
        temp_list.append(inputs["test_entity"][i][1])
        test_entity_1.append(temp_list)


    print('Pad sequences (samples x time)')

    train_word_list = []
    test_word_list = []
    train_POS_list=[]
    test_POS_list=[]

    train_distances_list=[]
    test_distances_list=[]
    train_distances2_list=[]
    test_distances2_list=[]

    for i in range(len(train_word)):
        train_word_list.append(
            sequence.pad_sequences(train_word[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_word)):
        test_word_list.append(sequence.pad_sequences(test_word[i], maxlen=word_length[i], truncating='post',
                                                       padding='post'))

    for i in range(len(train_POS)):
        train_POS_list.append(
            sequence.pad_sequences(train_POS[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances)):
        test_POS_list.append(sequence.pad_sequences(test_POS[i], maxlen=word_length[i], truncating='post',
                                                       padding='post'))

    for i in range(len(train_distances)):
        train_distances_list.append(
            sequence.pad_sequences(train_distances[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances)):
        test_distances_list.append(sequence.pad_sequences(test_distances[i], maxlen=word_length[i], truncating='post',
                                                       padding='post'))

    for i in range(len(train_distances2)):
        train_distances2_list.append(
            sequence.pad_sequences(train_distances2[i], maxlen=word_length[i], truncating='post', padding='post'))
    for i in range(len(test_distances2)):
        test_distances2_list.append(sequence.pad_sequences(test_distances2[i], maxlen=word_length[i], truncating='post',
                                                       padding='post'))

    print(inputs["train_shortest_word"][0])
    train_shortest_word= sequence.pad_sequences(inputs["train_shortest_word"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    test_shortest_word = sequence.pad_sequences(inputs["test_shortest_word"], maxlen=s['shortest_part_length'], truncating='post',
                                                 padding='post')

    train_shortest_pos = sequence.pad_sequences(inputs["train_shortest_pos"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    test_shortest_pos = sequence.pad_sequences(inputs["test_shortest_pos"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    train_shortest_dis1 = sequence.pad_sequences(inputs["train_shortest_dis1"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    test_shortest_dis1 = sequence.pad_sequences(inputs["test_shortest_dis1"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    train_shortest_dis2 = sequence.pad_sequences(inputs["train_shortest_dis2"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
    test_shortest_dis2 = sequence.pad_sequences(inputs["test_shortest_dis2"], maxlen=s['shortest_part_length'], truncating='post', padding='post')

    train_entity_0 = sequence.pad_sequences(train_entity_0, maxlen=1, truncating='post',padding='post')
    train_entity_1 = sequence.pad_sequences(train_entity_1, maxlen=1, truncating='post', padding='post')
    test_entity_0 = sequence.pad_sequences(test_entity_0, maxlen=1, truncating='post',padding='post')
    test_entity_1 = sequence.pad_sequences(test_entity_1, maxlen=1, truncating='post', padding='post')

    if s["use_ontology"]:
        train_common = sequence.pad_sequences(inputs["train_common_ancestors"], maxlen=s['shortest_part_length'], truncating='post', padding='post')
        test_common = sequence.pad_sequences(inputs["test_common_ancestors"], maxlen=s['shortest_part_length'], truncating='post', padding='post')

        train_concat0 = sequence.pad_sequences(inputs["train_concat_ancestors"][0], maxlen=s['shortest_part_length'],
                                             truncating='post', padding='post')
        train_concat1 = sequence.pad_sequences(inputs["train_concat_ancestors"][1], maxlen=s['shortest_part_length'],
                                              truncating='post', padding='post')
        test_concat0 = sequence.pad_sequences(inputs["test_concat_ancestors"][0], maxlen=s['shortest_part_length'],
                                              truncating='post', padding='post')
        test_concat1 = sequence.pad_sequences(inputs["test_concat_ancestors"][1], maxlen=s['shortest_part_length'],
                                             truncating='post', padding='post')

    print(train_shortest_word.shape)

    ############ MODEL DEFINITION
    #embedding layer

    wordembedding = Embedding(inputs["vec_table"].shape[0],
                              inputs["vec_table"].shape[1],
                         weights=[inputs["vec_table"]])

    disembedding = Embedding(inputs["dis_vec_table"].shape[0],
                             inputs["dis_vec_table"].shape[1],
                             weights=[inputs["dis_vec_table"]]
                             )
    #print(pos_vec_table.shape[0], pos_vec_table.shape[1])
    posembedding = Embedding(inputs["pos_vec_table"].shape[0],
                             inputs["pos_vec_table"].shape[1],
                        weights=[inputs["pos_vec_table"]]
                        )
    if s["use_ontology"]:
        ontoembedding = Embedding(inputs["onto_vec_table"].shape[0],
                                 inputs["onto_vec_table"].shape[1],
                                 weights=[inputs["onto_vec_table"]]
                                 )

    input_entity_0 = Input(shape=(1,), dtype='int32', name='input_entity_0')
    entity_fea_0 = wordembedding(input_entity_0)
    input_entity_1 = Input(shape=(1,), dtype='int32', name='input_entity_1')
    entity_fea_1 = wordembedding(input_entity_1)

    input_word_0 = Input(shape=(word_length[0],), dtype='int32', name='input_word_0')
    word_fea_0 = wordembedding(input_word_0)  # trainable=False

    input_pos_0 = Input(shape=(word_length[0],), dtype='int32', name='input_pos_0')
    pos_fea_0 = posembedding(input_pos_0)

    input_dis1_0 = Input(shape=(word_length[0],), dtype='int32', name='input_dis1_0')
    dis_fea1_0 = disembedding(input_dis1_0)

    input_dis2_0 = Input(shape=(word_length[0],), dtype='int32', name='input_dis2_0')
    dis_fea2_0 = disembedding(input_dis2_0)

    input_word_1 = Input(shape=(word_length[1],), dtype='int32', name='input_word_1')
    word_fea_1 = wordembedding(input_word_1)


    input_pos_1 = Input(shape=(word_length[1],), dtype='int32', name='input_pos_1')
    pos_fea_1 = posembedding(input_pos_1)

    input_dis1_1 = Input(shape=(word_length[1],), dtype='int32', name='input_dis1_1')
    dis_fea1_1 = disembedding(input_dis1_1)

    input_dis2_1 = Input(shape=(word_length[1],), dtype='int32', name='input_dis2_1')
    dis_fea2_1 = disembedding(input_dis2_1)

    input_word_2 = Input(shape=(word_length[2],), dtype='int32', name='input_word_2')
    word_fea_2 = wordembedding(input_word_2)


    input_pos_2 = Input(shape=(word_length[2],), dtype='int32', name='input_pos_2')
    pos_fea_2 = posembedding(input_pos_2)

    input_dis1_2 = Input(shape=(word_length[2],), dtype='int32', name='input_dis1_2')
    dis_fea1_2 = disembedding(input_dis1_2)

    input_dis2_2 = Input(shape=(word_length[2],), dtype='int32', name='input_dis2_2')
    dis_fea2_2 = disembedding(input_dis2_2)

    emb_merge_0 = Merge(mode='concat')([word_fea_0, pos_fea_0, dis_fea1_0, dis_fea2_0])
    emb_merge_1 = Merge(mode='concat')([word_fea_1, pos_fea_1, dis_fea1_1, dis_fea2_1])
    emb_merge_2 = Merge(mode='concat')([word_fea_2, pos_fea_2, dis_fea1_2, dis_fea2_2])



    input_shortest_word = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_shortest_word')
    shortest_word_fea = wordembedding(input_shortest_word)
    input_shortest_pos = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_shortest_pos')
    shortest_pos_fea = posembedding(input_shortest_pos)

    shortest_input_dis1 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='shortest_input_dis1')
    shortest_dis_fea1 = disembedding(shortest_input_dis1)

    shortest_input_dis2 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='shortest_input_dis2')
    shortest_dis_fea2 = disembedding(shortest_input_dis2)
    emb_merge_shortest = Merge(mode='concat')(
        [shortest_word_fea, shortest_pos_fea, shortest_dis_fea1, shortest_dis_fea2])


    if s["use_ontology"]:
        input_common = Input(shape=(s['shortest_part_length'],), dtype='int32', name='common_input')
        common_fea1 = ontoembedding(input_common)

        input_concat_0 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_concat_0')
        concat_fea_0 = ontoembedding(input_concat_0)
        input_concat_1 = Input(shape=(s['shortest_part_length'],), dtype='int32', name='input_concat_1')
        concat_fea_1 = ontoembedding(input_concat_1)

        #emb_merge_ancestors = Merge(mode='concat')([common_fea1, concat_fea_0, concat_fea_1,shortest_pos_fea, shortest_dis_fea1, shortest_dis_fea2])
        emb_merge_ancestors = Merge(mode='concat')([common_fea1,shortest_pos_fea, shortest_dis_fea1, shortest_dis_fea2])

            #[common_fea1, concat_fea_0, concat_fea_1])

    #attention layer
    att_emb_merge_0 = emb_AttentionLayer()([word_fea_0, entity_fea_0, entity_fea_1, emb_merge_0])

    att_emb_merge_1= emb_AttentionLayer()([word_fea_1, entity_fea_0, entity_fea_1, emb_merge_1])
    att_emb_merge_2 = emb_AttentionLayer()([word_fea_2, entity_fea_0, entity_fea_1, emb_merge_2])
    att_emb_merge_shortest = emb_AttentionLayer()([shortest_word_fea, entity_fea_0, entity_fea_1, emb_merge_shortest])

    if s["use_ontology"]:
        att_emb_merge_ancestors = emb_AttentionLayer()([common_fea1, entity_fea_0, entity_fea_1, emb_merge_ancestors])
        #att_emb_merge_ancestors = emb_AttentionLayer()([common_fea1, concat_fea_0, concat_fea_1, emb_merge_ancestors])

    #dropout layer
    emb_merge_0 = Dropout(s['emb_dropout'])(att_emb_merge_0)
    emb_merge_1 = Dropout(s['emb_dropout'])(att_emb_merge_1)
    emb_merge_2 = Dropout(s['emb_dropout'])(att_emb_merge_2)
    emb_merge_shortest = Dropout(s['emb_dropout'])(att_emb_merge_shortest)

    if s["use_ontology"]:
        emb_merge_ancestors = Dropout(s['emb_dropout'])(att_emb_merge_ancestors)

    #bottom RNNs
    left_gru_0 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                   activation='tanh',
                   inner_activation='sigmoid')(emb_merge_0)

    right_gru_0 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                    activation='tanh',
                    inner_activation='sigmoid',
                    go_backwards=True)(emb_merge_0)

    gru_merge_0 = Merge(mode='concat')([left_gru_0, right_gru_0])


    left_gru_1 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                     activation='tanh',
                     inner_activation='sigmoid')(emb_merge_1)

    right_gru_1 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                      activation='tanh',
                      inner_activation='sigmoid',
                      go_backwards=True)(emb_merge_1)

    gru_merge_1 = Merge(mode='concat')([left_gru_1, right_gru_1])

    left_gru_2 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                     activation='tanh',
                     inner_activation='sigmoid')(emb_merge_2)

    right_gru_2 = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                      activation='tanh',
                      inner_activation='sigmoid',
                      go_backwards=True)(emb_merge_2)

    gru_merge_2 = Merge(mode='concat')([left_gru_2, right_gru_2])

    left_gru_shortest = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                     activation='tanh',
                     inner_activation='sigmoid')(emb_merge_shortest)

    right_gru_shortest = LSTM(output_dim=s['first_hidden_layer'],
                   init='orthogonal',
                      activation='tanh',
                      inner_activation='sigmoid',
                      go_backwards=True)(emb_merge_shortest)

    gru_merge_shortest = Merge(mode='concat')([left_gru_shortest, right_gru_shortest])

    if s["use_ontology"]:
        left_gru_ancestors = LSTM(output_dim=s['first_hidden_layer'],
                                 init='orthogonal',
                                 activation='tanh',
                                 inner_activation='sigmoid')(emb_merge_ancestors)

        right_gru_ancestors = LSTM(output_dim=s['first_hidden_layer'],
                                  init='orthogonal',
                                  activation='tanh',
                                  inner_activation='sigmoid',
                                  go_backwards=True)(emb_merge_ancestors)

        gru_merge_ancestors = Merge(mode='concat')([left_gru_ancestors, right_gru_ancestors])


    gru_merge_0=Reshape((1,2*s['first_hidden_layer']))(gru_merge_0)
    gru_merge_1 = Reshape((1, 2*s['first_hidden_layer']))(gru_merge_1)
    gru_merge_2=Reshape((1,2*s['first_hidden_layer']))(gru_merge_2)
    gru_merge_shortest=Reshape((1,2*s['first_hidden_layer']))(gru_merge_shortest)



    if s["use_ontology"]:
        gru_merge_ancestors = Reshape((1, 2 * s['first_hidden_layer']))(gru_merge_ancestors)
        gru_merge_temp = Merge(mode='concat', concat_axis=1)(
            [gru_merge_0, entity_fea_0, gru_merge_1, gru_merge_shortest, entity_fea_1, gru_merge_2 , gru_merge_ancestors])
    else:
        gru_merge_temp = Merge(mode='concat', concat_axis=1)(
            [gru_merge_0, entity_fea_0, gru_merge_1, gru_merge_shortest, entity_fea_1, gru_merge_2])
    #top RNNs
    left_gru_top = LSTM(output_dim=s['first_hidden_layer'],
                            init='orthogonal',
                            # dropout_W=0.5, dropout_U=0.5,
                            activation='tanh',  # return_sequences=True,
                            inner_activation='sigmoid')(gru_merge_temp)

    right_gru_top = LSTM(output_dim=s['first_hidden_layer'],
                             init='orthogonal',
                             # dropout_W=0.5, dropout_U=0.5,
                             activation='tanh',  # return_sequences=True,
                             inner_activation='sigmoid',
                             go_backwards=True)(gru_merge_temp)

    gru_merge_top = Merge(mode='concat')([left_gru_top, right_gru_top])
    dense_drop_0=Dropout(s['dense_dropout'])(gru_merge_top)

    final_output_1=Dense(nb_class)(dense_drop_0)
    final_output=Activation('softmax')(final_output_1)

    if s["use_ontology"]:
        model = Model(input=[input_word_0,input_word_1,input_word_2,input_pos_0,input_pos_1,input_pos_2,input_dis1_0,input_dis1_1,input_dis1_2,
                         input_dis2_0,input_dis2_1,input_dis2_2,input_shortest_word, shortest_input_dis1,shortest_input_dis2,
                         input_shortest_pos,input_entity_0,input_entity_1, input_common, input_concat_0, input_concat_1], output=final_output)
    else:
        model = Model(
            input=[input_word_0, input_word_1, input_word_2, input_pos_0, input_pos_1, input_pos_2, input_dis1_0,
                   input_dis1_1, input_dis1_2,
                   input_dis2_0, input_dis2_1, input_dis2_2, input_shortest_word, shortest_input_dis1,
                   shortest_input_dis2,
                   input_shortest_pos, input_entity_0, input_entity_1],
            output=final_output)

    #opt = RMSprop(lr=s['lr'])
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    #opt = RMSprop(lr=s['lr'], rho=0.9, epsilon=1e-06)
    opt = Adam()
    #opt = SGD(lr=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=opt,\
                  metrics=['accuracy'])

    print('\n')

    model.summary()

    #model.load_weights("./model_weights.h5")

    inds = list(range(train_word_list[0].shape[0]))
    np.random.shuffle(inds)

    batch_num = len(inds) // s['mini_batch']

    totalloss=0
    #print len(inds),s['mini_batch'],batch_num

    print ('-----------------Begin of training-------------------')
    tnow = time.time()
    for epoch in range(s['epochs']):

        loss=acc=0

        print('learning epoch:'+str(epoch))

        for minibatch in range(batch_num):
            val_inputs = [train_word_list[0][inds[minibatch::batch_num]], \
                                      train_word_list[1][inds[minibatch::batch_num]], \
                                      train_word_list[2][inds[minibatch::batch_num]], \
                                      train_POS_list[0][inds[minibatch::batch_num]], \
                                      train_POS_list[1][inds[minibatch::batch_num]],
                                      train_POS_list[2][inds[minibatch::batch_num]],\
                                      train_distances_list[0][inds[minibatch::batch_num]], \
                                      train_distances_list[1][inds[minibatch::batch_num]],
                                      train_distances_list[2][inds[minibatch::batch_num]],\
                                      train_distances2_list[0][inds[minibatch::batch_num]],
                                      train_distances2_list[1][inds[minibatch::batch_num]],
                                      train_distances2_list[2][inds[minibatch::batch_num]],
                                      train_shortest_word[inds[minibatch::batch_num]],
                                      train_shortest_dis1[inds[minibatch::batch_num]],
                                      train_shortest_dis2[inds[minibatch::batch_num]],
                                      train_shortest_pos[inds[minibatch::batch_num]],
                                      train_entity_0[inds[minibatch::batch_num]],
                                      train_entity_1[inds[minibatch::batch_num]]]
            if s["use_ontology"]:
                val_inputs += [train_common[inds[minibatch::batch_num]],
                               train_concat0[inds[minibatch::batch_num]],
                                          train_concat1[inds[minibatch::batch_num]]]

            val=model.train_on_batch(val_inputs,
                                      train_y[inds[minibatch::batch_num]])

            if minibatch%20==0:
                    #print ('=', end = '')
                    print('=', end=' ')
            loss=loss+val[0]
        #training ended every  epoch
        totalloss=totalloss+loss
        print(('<<    ','training loss:',str(np.round(loss,5)) ))
        sys.stdout.flush()
        #model_arch = model.to_json()
        #model.save_weights('temp_model.h5', overwrite=True)
        #del model
        #model = model_from_json(model_arch)
        #model = model.load_weights('temp_model.h5')
        print((time.time() - tnow))
        tnow = time.time()
        print ('------------------Begin of testing------------------')
        test_inputs = [test_word_list[0],test_word_list[1],test_word_list[2],test_POS_list[0],test_POS_list[1],test_POS_list[2],
                                  test_distances_list[0],test_distances_list[1],test_distances_list[2],test_distances2_list[0],
                                  test_distances2_list[1],test_distances2_list[2],test_shortest_word,test_shortest_dis1,
                                   test_shortest_dis2,test_shortest_pos,test_entity_0,test_entity_1]

        if s["use_ontology"]:
            test_inputs += [test_common, test_concat0, test_concat1]

        pred_test = model.predict(test_inputs,
                                  batch_size=s['mini_batch'])#,test_POS


        precision, recall, F1=result_evaluation(answer_y, pred_test)

        #print'testing epochs:' + ' precision:' + str(np.round(precision, 5)) + ' recall:' + str(np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5))

        print('testing epochs:' + str(epoch)+' precision:' + str(np.round(precision, 5)) + ' recall:' + str(np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)))
        #print('testing epochs:' + str(epoch) + ' precision:' + str(np.round(precision2, 5)) + ' recall:' + str(
            np.round(recall2, 5)) + ' F1:' + str(np.round(F12, 5)))
        print((time.time() - tnow))
        tnow = time.time()
    print('-----------------End of DDI extraction----------------')

if __name__ == '__main__':
    main()