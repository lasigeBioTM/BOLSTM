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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parse_ddi import get_ddi_sdp_instances
from parse_semeval8 import get_semeval8_sdp_instances
GLOVE_DIR = "data/"
vocab_size = 10000
embbed_size = 300
LSTM_units = 300
sigmoid_units = 100
sigmoid_l2_reg = 0.000001
dropout1 = 0.5
#n_classes = 19
n_classes = 5
max_sentence_length = 10

n_epochs = 10
batch_size = 20
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
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_vectors[word] = coefs
        embedding_indexes[word] = i
    f.close()
    print('Found %s word vectors.' % len(embeddings_vectors))

    # assemble the embedding_weights in one numpy array
    n_symbols = len(embedding_indexes) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols, embbed_size))
    for word, index in embedding_indexes.items():
        embedding_weights[index, :] = embeddings_vectors[word]

    return embedding_indexes, embedding_weights

def preprocess_sequences(x_data, embeddings_index):
    data = []
    for i, seq in enumerate(x_data):
        #for w in seq:
            #if w.lower() not in embeddings_index:
            #    print("word not in index: {}".format(w.lower()))
        data.append([embeddings_index.get(w.lower()) for w in seq if w in embeddings_index])
    data = pad_sequences(data, maxlen=max_sentence_length)
    return data


def get_model(embedding_matrix):
    #input_left = Input(shape=(max_sentence_length, embbed_size),  name='left_input')
    #input_right = Input(shape=(max_sentence_length, embbed_size),  name='right_input')

    input_left = Input(shape=(max_sentence_length,), name='left_input')
    input_right = Input(shape=(max_sentence_length,), name='right_input')

    e_left = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                       trainable=False)
    e_left.build((None,))
    e_left.set_weights([embedding_matrix])
    e_left = e_left(input_left)

    e_left = Dropout(0.5)(e_left)

    e_right = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                        trainable=False)
    e_right.build((None,))
    e_right.set_weights([embedding_matrix])
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
                  #optimizer=Adam(),
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
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
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
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]],
                                                      ))).round()
        val_targ = self.validation_data[2]
        probs = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]],
                                                      ))
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
    #dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    #instances = np.empty((0, max_sentence_length, embbed_size))
    #instances = np.empty((0, max_sentence_length))
    left_instances = []
    right_instances = []
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes = get_ddi_sdp_instances(dir)
        #dir_instances = np.array(dir_instances)
        #print(dir_instances)
        #dir_instances = sequence.pad_sequences(dir_instances, maxlen=max_sentence_length)
        dir_classes = np.array(dir_classes)

        labels += dir_labels
        #print(instances.shape, dir_instances.shape)
        #instances = np.concatenate((instances, dir_instances), axis=0)
        left_instances += dir_instances[0]
        right_instances += dir_instances[1]
        classes = np.concatenate((classes, dir_classes), axis=0)
    return labels, (left_instances, right_instances), classes





def main():

    if sys.argv[1] == "preprocessing":
        #labels, X_train, classes = get_ddi_data(sys.argv[3:])
        train = True
        if "test" in sys.argv[3].lower():
            train= False
        if sys.argv[2] == "semeval8":
            labels, X_train, classes = get_semeval8_sdp_instances(sys.argv[4:], train=train)
            print(len(X_train))
            print(len(X_train[0]))
        elif sys.argv[2] == "ddi":
            labels, X_train, classes = get_ddi_data(sys.argv[4:])
            print(len(X_train))
        np.save(sys.argv[3] + "_labels.npy", labels)
        np.save(sys.argv[3] + "_x_words.npy", X_train)
        np.save(sys.argv[3] + "_y.npy", classes)

    elif sys.argv[1] == "train":

        if os.path.isfile("model.json"):
            os.remove("model.json")
        if os.path.isfile("model.h5"):
            os.remove("model.h5")
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        print(len(X_words_train))
        #print(X_words_train[0].shape, X_words_train[1].shape, Y_train.shape)
        #X_words_left = [one_hot(" ".join(d), vocab_size) for d in X_words_train[0]]
        #X_words_right = [one_hot(" ".join(d), vocab_size) for d in X_words_train[1]]
        #X_words_train = [pad_sequences(X_words_left, maxlen=max_sentence_length, padding='post'),
        #                 pad_sequences(X_words_right, maxlen=max_sentence_length, padding='post')]
        #print(X_words_train[0].shape, X_words_train[1].shape, Y_train.shape)
        emb_index, emb_matrix = get_glove_vectors()
        # print(emb_index)
        X_words_left = preprocess_sequences(X_words_train[0], emb_index)
        X_words_right = preprocess_sequences(X_words_train[1], emb_index)
        X_words_train = [X_words_left, X_words_right]
        model = get_model(emb_matrix)
        #Y_train = Y_train[...,1:]
        # print(Y_train)
        if len(sys.argv) > 3:
            X_test = np.load(sys.argv[3] + "_x_words.npy")
            Y_test = np.load(sys.argv[3] + "_y.npy")
            Y_test = to_categorical(Y_test, num_classes=n_classes)
            #Y_test = Y_test[...,1:]
            test_labels = np.load(sys.argv[3] + "_labels.npy")

            model.fit(X_words_train, Y_train, validation_data=(X_test, Y_test), epochs=n_epochs,
                      batch_size=batch_size, callbacks=[metrics], verbose=2)

        else:

            history = model.fit({"left_input":X_words_train[0], "right_input": X_words_train[1]},
                      {"output": Y_train}, validation_split=validation_split, epochs=n_epochs,
                      batch_size=batch_size, verbose=2, callbacks=[metrics])
                                                                   #keras.callbacks.EarlyStopping(patience=3)])
            write_plots(history)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    elif sys.argv[1] == "predict":
        emb_index, emb_matrix = get_glove_vectors()
        X_words_test = np.load(sys.argv[2] + "_x_words.npy")
        #X_words_test_left = [one_hot(" ".join(d), vocab_size) for d in X_words_test[0]]
        #X_words_test_right = [one_hot(" ".join(d), vocab_size) for d in X_words_test[1]]
        #X_words_test = [pad_sequences(X_words_test_left, maxlen=max_sentence_length, padding='post'),
        #                pad_sequences(X_words_test_right, maxlen=max_sentence_length, padding='post')]

        X_words_test_left = preprocess_sequences(X_words_test[0], emb_index)
        X_words_test_right = preprocess_sequences(X_words_test[1], emb_index)
        X_words_test = [X_words_test_left, X_words_test_right]

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")


        test_labels = np.load(sys.argv[2] + "_labels.npy")

        scores = loaded_model.predict(X_words_test)
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

if __name__ == "__main__":
    main()
