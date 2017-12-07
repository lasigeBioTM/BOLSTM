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


from parse_ddi import get_ddi_sdp_instances
from parse_semeval8 import get_semeval8_sdp_instances

vocab_size = 10000
embbed_size = 300
LSTM_units = 200
sigmoid_units = 100
n_classes = 19
max_sentence_length = 10

n_epochs = 30
batch_size = 10
validation_split = 0.1


def get_model():
    input_left = Input(shape=(max_sentence_length, embbed_size),  name='left_input')
    input_right = Input(shape=(max_sentence_length, embbed_size),  name='right_input')

    lstm_left = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True)(input_left)
    lstm_right = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True)(input_right)

    pool_left = GlobalMaxPooling1D()(lstm_left)
    pool_right = GlobalMaxPooling1D()(lstm_right)

    concatenate = keras.layers.concatenate([pool_left, pool_right], axis=-1)
    we_hidden = Dense(sigmoid_units, activation='sigmoid')(concatenate)
    #we_hidden = Dropout(0.3)(we_hidden)
    output = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.00001),
                   name='output')(we_hidden)
    model = Model(inputs=[input_left, input_right], outputs=[output])
    #model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence_length))
    #model.add(Flatten())

    #model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(1), metrics=['accuracy', precision, recall, f1])
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
        _val_f1 = f1_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_recall = recall_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_precision = precision_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        s = "predicted not false: {}/{}".format(len([x for x in val_predict if x[0] < 0.5]),
                                                len([x for x in val_targ if x[0] < 0.5]))
        print("\n{} VAL_f1:{:6.3f} VAL_p:{:6.3f} VAL_r{:6.3f} \n".format(s, _val_f1, _val_precision, _val_recall),)
        #print(val_predict, val_targ)
        # print("true not false: {}".format()
        return


metrics = Metrics()
#model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=[X_test,y_test],
#       verbose=1, callbacks=[metrics])


# model = Sequential()
# model.add(Dense(units=64, activation='relu', input_dim=100))
# model.add(Dense(units=10, activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
#
#
#
# model.fit(instances, classes, epochs=5, batch_size=32)



def get_ddi_data(dirs=["data/ddi2013Train/DrugBank/", "data/ddi2013Train/MedLine/"]):
    #dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    #instances = np.empty((0, max_sentence_length, embbed_size))
    #instances = np.empty((0, max_sentence_length))
    instances = []
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
        instances += dir_instances
        classes = np.concatenate((classes, dir_classes), axis=0)
    return labels, instances, classes





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
            labels, X_train, classes = get_ddi_data(sys.argv[3:])
        np.save(sys.argv[3] + "_labels.npy", labels)
        np.save(sys.argv[3] + "_x_words.npy", X_train)
        np.save(sys.argv[3] + "_y.npy", classes)

    elif sys.argv[1] == "train":
        model = get_model()
        if os.path.isfile("model.json"):
            os.remove("model.json")
        if os.path.isfile("model.h5"):
            os.remove("model.h5")
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        print(len(X_words_train))
        print(X_words_train[0].shape, X_words_train[1].shape, Y_train.shape)
        #X_words_train = [one_hot(" ".join(d), vocab_size) for d in X_words_train]
        X_words_train = [pad_sequences(X_words_train[0], maxlen=max_sentence_length, padding='post'),
                         pad_sequences(X_words_train[1], maxlen=max_sentence_length, padding='post')]
        print(X_words_train[0].shape, X_words_train[1].shape, Y_train.shape)


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

            model.fit({"left_input":X_words_train[0], "right_input": X_words_train[1]},
                      {"output": Y_train}, validation_split=validation_split, epochs=n_epochs,
                      batch_size=batch_size, verbose=1, callbacks=[metrics,
                                                                   keras.callbacks.EarlyStopping(patience=3)])

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    elif sys.argv[1] == "predict":

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        X_words_test = np.load(sys.argv[2] + "_x_words.npy")
        #X_words_test = [one_hot(" ".join(d), vocab_size) for d in X_words_test]
        X_words_test = [pad_sequences(X_words_test[0], maxlen=max_sentence_length, padding='post'),
                        pad_sequences(X_words_test[1], maxlen=max_sentence_length, padding='post')]
        test_labels = np.load(sys.argv[2] + "_labels.npy")

        scores = loaded_model.predict(X_words_test)
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

if __name__ == "__main__":
    main()
