import sys
import logging
import os
logging.basicConfig(level=10)

import numpy as np
np.random.seed(1)
import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras import regularizers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
#from gensim.models.keyedvectors import KeyedVectors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from chebi_path import load_chebi
DATA_DIR = "data/"
vocab_size = 10000
embbed_size = 200
chebi_embbed_size = 100
LSTM_units = 200
sigmoid_units = 100
sigmoid_l2_reg = 0.000001
dropout1 = 0.5
#n_classes = 19
n_classes = 5
max_sentence_length = 5
max_ancestors_length = 10

n_epochs = 20
batch_size = 5
validation_split = 0.1

# https://github.com/keras-team/keras/issues/853#issuecomment-343981960

def write_plots(history):
    plt.figure()
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model F1')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_f1.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_loss.png")

def get_glove_vectors():
    embeddings_vectors = {} # words -> vector
    embedding_indexes = {}
    # load embeddings indexes: word -> coefs
    f = open(os.path.join(DATA_DIR, 'glove.6B.200d.txt'))
    #f = open(os.path.join(DATA_DIR, 'PubMed-and-PMC-w2v.txt'))
    for i, line in enumerate(f):
        #if i == 0:
        #    continue
        values = line.split()
        word = values[0].lower()
        #if word.isdigit():
        #    continue
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_vectors[word] = coefs
        #print(word, coefs)
        embedding_indexes[word] = i
        #print(i)
    print(len(embedding_indexes))
    f.close()
    print('Found %s word vectors.' % len(embeddings_vectors))

    # assemble the embedding_weights in one numpy array
    #n_symbols = len(embedding_indexes) + 1  # adding 1 to account for 0th index (for masking)
    n_symbols = i + 1  # hack because bio vectors are case snsitive
    embedding_weights = np.zeros((n_symbols, embbed_size))
    for word, index in embedding_indexes.items():
        #print(index, n_symbols)
        embedding_weights[index, :] = embeddings_vectors[word]

    return embedding_indexes, embedding_weights

def get_w2v():
    embeddings_vectors = {}  # words -> vector
    embedding_indexes = {}
    word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format



def preprocess_sequences(x_data, embeddings_index):
    data = []
    for i, seq in enumerate(x_data):
        #for w in seq:
            #if w.lower() not in embeddings_index:
            #    print("word not in index: {}".format(w.lower()))
        #print(seq)
        idxs = [embeddings_index.get(w.lower()) for w in seq if w.lower() in embeddings_index]
        if None in idxs:
            print(seq, idxs)
        #print(idxs)
        data.append(idxs)
    #print(data)
    data = pad_sequences(data, maxlen=max_sentence_length)
    return data

def preprocess_ids(x_data, id_to_index):
    # process a sequence of ontology:IDs, so a embedding index is not necessary
    data = []
    for i, seq in enumerate(x_data):
        # print(seq)
        idxs = [id_to_index[d] for d in seq if d]
        data.append(idxs)
    data = pad_sequences(data, maxlen=max_ancestors_length)
    return data


def get_model(embedding_matrix, id_to_index, words=True, ancestors=True):
    inputs = []
    pool_layers = []
    if words:
        words_input_left = Input(shape=(max_sentence_length,), name='left_words')
        words_input_right = Input(shape=(max_sentence_length,), name='right_words')

        inputs += [words_input_left, words_input_right]

        e_words_left = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                           trainable=False)
        e_words_left.build((None,))
        e_words_left.set_weights([embedding_matrix])
        e_words_left = e_words_left(words_input_left)

        e_words_left = Dropout(0.5)(e_words_left)

        e_words_right = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                            trainable=False)
        e_words_right.build((None,))
        e_words_right.set_weights([embedding_matrix])
        e_words_right = e_words_right(words_input_right)

        e_words_right = Dropout(0.5)(e_words_right)

        words_lstm_left = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_words_left)
        words_lstm_right = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                          kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_words_right)

        words_pool_left = GlobalMaxPooling1D()(words_lstm_left)
        words_pool_right = GlobalMaxPooling1D()(words_lstm_right)

        # we_hidden = Dense(sigmoid_units, activation='sigmoid',
        #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
        # we_hidden = Dropout(dropout1)(we_hidden)
        pool_layers += [words_pool_left, words_pool_right]

    if ancestors:
        ancestors_input_left = Input(shape=(max_ancestors_length,), name='left_ancestors')
        ancestors_input_right = Input(shape=(max_ancestors_length,), name='right_ancestors')
        inputs += [ancestors_input_left, ancestors_input_right]

        e_ancestors_left = Embedding(len(id_to_index), chebi_embbed_size, input_length=max_ancestors_length,
                                 trainable=True)
        e_ancestors_left.build((None,))
        e_ancestors_left = e_ancestors_left(ancestors_input_left)

        e_ancestors_left = Dropout(0.5)(e_ancestors_left)

        e_ancestors_right = Embedding(len(id_to_index), chebi_embbed_size, input_length=max_ancestors_length,
                                  trainable=True)
        e_ancestors_right.build((None,))
        # e_right.set_weights([embedding_matrix])
        e_ancestors_right = e_ancestors_right(ancestors_input_right)

        e_ancestors_right = Dropout(0.5)(e_ancestors_right)

        ancestors_lstm_left = LSTM(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size), return_sequences=True,
                               kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_ancestors_left)
        ancestors_lstm_right = LSTM(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size), return_sequences=True,
                                kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_ancestors_right)

        ancestors_pool_left = GlobalMaxPooling1D()(ancestors_lstm_left)
        ancestors_pool_right = GlobalMaxPooling1D()(ancestors_lstm_right)

        # we_hidden = Dense(sigmoid_units, activation='sigmoid',
        #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
        # we_hidden = Dropout(dropout1)(we_hidden)
        pool_layers += [ancestors_pool_left, ancestors_pool_right]

    concatenate = keras.layers.concatenate(pool_layers, axis=-1)

    final_hidden = Dense(sigmoid_units, activation='sigmoid',  # , kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg), )(concatenate)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=inputs, outputs=[output])
    # model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence_length))
    # model.add(Flatten())

    # model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  # optimizer=Adam(0.001),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model

def get_words_model(embedding_matrix):
    input = Input(shape=((max_sentence_length*2)-1,), name="input")

    emb = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                        trainable=False)
    emb.build((None,))
    emb.set_weights([embedding_matrix])
    emb = emb(input)


    lstm = LSTM(LSTM_units, input_shape=((max_sentence_length*2)-1, embbed_size), #, return_sequences=True,
                kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(emb)

    #pool_left = GlobalMaxPooling1D()(lstm_left)
    #pool_right = GlobalMaxPooling1D()(lstm_right)

    #concatenate = keras.layers.concatenate([pool_left, pool_right], axis=-1)
    # we_hidden = Dense(sigmoid_units, activation='sigmoid',
    #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    # we_hidden = Dropout(dropout1)(we_hidden)
    final_hidden = Dense(sigmoid_units, activation='sigmoid',  # , kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg), )(lstm)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=input, outputs=output)

    # model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  # optimizer=Adam(0.001),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model


def get_xu_model(embedding_matrix):
    #input_left = Input(shape=(max_sentence_length, embbed_size),  name='left_input')
    #input_right = Input(shape=(max_sentence_length, embbed_size),  name='right_input')

    input_left = Input(shape=(max_sentence_length,), name='left_input')
    input_right = Input(shape=(max_sentence_length,), name='right_input')

    e_left = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                       trainable=True)
    e_left.build((None,))
    #e_left.set_weights([embedding_matrix])
    e_left = e_left(input_left)

    e_left = Dropout(0.5)(e_left)

    e_right = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                        trainable=True)
    e_right.build((None,))
    #e_right.set_weights([embedding_matrix])
    e_right = e_right(input_right)

    e_right = Dropout(0.5)(e_right)

    lstm_left = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                     kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_left)
    lstm_right = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                      kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_right)

    pool_left = GlobalMaxPooling1D()(lstm_left)
    pool_right = GlobalMaxPooling1D()(lstm_right)

    concatenate = keras.layers.concatenate([pool_left, pool_right], axis=-1)
    #we_hidden = Dense(sigmoid_units, activation='sigmoid',
    #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    #we_hidden = Dropout(dropout1)(we_hidden)
    final_hidden = Dense(sigmoid_units, activation='sigmoid', #, kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=[input_left, input_right], outputs=[output])
    #model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence_length))
    #model.add(Flatten())

    #model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  #optimizer=Adam(0.001),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model

#https://github.com/fchollet/keras/issues/5400
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    # print(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
        p = true_positives / (predicted_positives + K.epsilon())
        return p

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
        r = true_positives / (possible_positives + K.epsilon())
        return r
    precision_v = precision(y_true, y_pred)
    recall_v = recall(y_true, y_pred)
    return (2.0*precision_v*recall_v)/(precision_v+recall_v)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        #print(dir(self.model))
        #print(len(self.validation_data))
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1],
                                                     self.validation_data[2], self.validation_data[3]]
                                                      ))).round()
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0],
        #                                             ))).round()
        val_targ = self.validation_data[4]
        #val_targ = self.validation_data[1]
        #probs = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]],
        #                                              ))
        _val_f1 = f1_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_recall = recall_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_precision = precision_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        s = "predicted not false: {}/{}".format(len([x for x in val_predict if np.argmax(x) != 0]),
                                                len([x for x in val_targ if x[0] < 0.5]))
        print("\n{} VAL_f1:{:6.3f} VAL_p:{:6.3f} VAL_r{:6.3f}\n".format(s, _val_f1, _val_precision, _val_recall),)
        #for i in range(len(self.validation_data[2])):
        #    if np.argmax(val_targ[i]) != np.argmax(val_predict[i]):
        #        print(i, np.argmax(val_targ[i]), np.argmax(val_predict[i]), probs[i])
        print("\n")
        #print(val_predict, val_targ)
        # print("true not false: {}".format()
        return


metrics = Metrics()



def get_ddi_data(dirs=["data/ddi2013Train/DrugBank/", "data/ddi2013Train/MedLine/"]):

    from parse_ddi import get_ddi_sdp_instances

    #dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    #instances = np.empty((0, max_sentence_length, embbed_size))
    #instances = np.empty((0, max_sentence_length))
    left_instances = []
    right_instances = []
    left_ancestors = []
    right_ancestors = []
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes, dir_ancestors = get_ddi_sdp_instances(dir)
        #dir_instances = np.array(dir_instances)
        #print(dir_instances)
        #dir_instances = sequence.pad_sequences(dir_instances, maxlen=max_sentence_length)
        dir_classes = np.array(dir_classes)

        labels += dir_labels
        #print(instances.shape, dir_instances.shape)
        #instances = np.concatenate((instances, dir_instances), axis=0)
        left_instances += dir_instances[0]
        right_instances += dir_instances[1]
        left_ancestors += dir_ancestors[0]
        right_ancestors += dir_ancestors[1]
        classes = np.concatenate((classes, dir_classes), axis=0)
    return labels, (left_instances, right_instances), classes, (left_ancestors, right_ancestors)





def main():

    if sys.argv[1] == "preprocessing":
        from parse_semeval8 import get_semeval8_sdp_instances
        #labels, X_train, classes = get_ddi_data(sys.argv[3:])
        train = True
        if "test" in sys.argv[3].lower():
            train= False
        if sys.argv[2] == "semeval8":
            labels, X_train, classes = get_semeval8_sdp_instances(sys.argv[4:], train=train)
            print(len(X_train))
            print(len(X_train[0]))
        elif sys.argv[2] == "ddi":
            labels, X_train, classes, X_train_ancestors = get_ddi_data(sys.argv[4:])
            print(len(X_train))
        np.save(sys.argv[3] + "_labels.npy", labels)
        np.save(sys.argv[3] + "_x_words.npy", X_train)
        np.save(sys.argv[3] + "_x_ancestors.npy", X_train_ancestors)
        np.save(sys.argv[3] + "_y.npy", classes)

    elif sys.argv[1] == "train":
        is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
        if os.path.isfile("model.json"):
            os.remove("model.json")
        if os.path.isfile("model.h5"):
            os.remove("model.h5")
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        emb_index, emb_matrix = get_glove_vectors()
        # print(emb_index)
        X_words_left = preprocess_sequences(X_words_train[0], emb_index)
        X_words_right = preprocess_sequences(X_words_train[1], emb_index)
        # skip root word
        #X_words_train = np.concatenate((X_words_left, X_words_right[..., 1:]), 1)

        X_ids_left = preprocess_ids(X_ancestors_train[0], id_to_index)
        X_ids_right = preprocess_ids(X_ancestors_train[1], id_to_index)

        #X_ancestors_train = np.concatenate((X_ids_left, X_ids_right[..., 1:]), 1)

        model = get_model(emb_matrix, id_to_index, words=True, ancestors=True)
        #model = get_words_model(emb_matrix)
        #model = get_xu_model(emb_matrix)

        inputs = {"left_ancestors": X_ids_left, "right_ancestors": X_ids_right,
                  "left_words":X_words_left, "right_words": X_words_right,
                 }
        #print(inputs)
        if len(sys.argv) > 3:
            X_test = np.load(sys.argv[3] + "_x_words.npy")
            Y_test = np.load(sys.argv[3] + "_y.npy")
            Y_test = to_categorical(Y_test, num_classes=n_classes)
            #Y_test = Y_test[...,1:]
            test_labels = np.load(sys.argv[3] + "_labels.npy")

            model.fit(X_words_train, Y_train, validation_data=(X_test, Y_test), epochs=n_epochs,
                      batch_size=batch_size, callbacks=[metrics], verbose=2)

        else:

            history = model.fit(inputs,
                      {"output": Y_train}, validation_split=validation_split, epochs=n_epochs,
                      batch_size=batch_size, verbose=2, callbacks=[metrics])
                                                                   #keras.callbacks.EarlyStopping(patience=3)])
            #history = model.fit({"input": X_words_train}, {"output": Y_train},
            #                    validation_split=validation_split, epochs=n_epochs,
            #                    batch_size=batch_size, verbose=2, callbacks=[metrics])
            write_plots(history)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    elif sys.argv[1] == "predict":
        is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
        emb_index, emb_matrix = get_glove_vectors()
        X_words_test = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_test = np.load(sys.argv[2] + "_x_ancestors.npy")
        #X_words_test_left = [one_hot(" ".join(d), vocab_size) for d in X_words_test[0]]
        #X_words_test_right = [one_hot(" ".join(d), vocab_size) for d in X_words_test[1]]
        #X_words_test = [pad_sequences(X_words_test_left, maxlen=max_sentence_length, padding='post'),
        #                pad_sequences(X_words_test_right, maxlen=max_sentence_length, padding='post')]

        X_words_test_left = preprocess_sequences(X_words_test[0], emb_index)
        X_words_test_right = preprocess_sequences(X_words_test[1], emb_index)
        #X_words_test = [X_words_test_left, X_words_test_right]
        X_words_test = np.concatenate((X_words_test_left, X_words_test_right[..., 1:]), 1)

        X_ids_left = preprocess_ids(X_ancestors_test[0], id_to_index)
        X_ids_right = preprocess_ids(X_ancestors_test[1], id_to_index)

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")


        test_labels = np.load(sys.argv[2] + "_labels.npy")

        #scores = loaded_model.predict(X_words_test)
        scores = loaded_model.predict({"left_words": X_words_test_left, "right_words": X_words_test_right,
                                       "left_ancestors": X_ids_left, "right_ancestors": X_ids_right})
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

if __name__ == "__main__":
    main()
