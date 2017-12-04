import sys
import logging
logging.basicConfig(level=30)

import numpy as np
np.random.seed(1)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score

from parse_ddi import get_ddi_sdp_instances

embbed_size = 300
LSTM_units = 300
sigmoid_units = 100
n_classes = 2
max_sentence_length = 10

#https://github.com/fchollet/keras/issues/5400
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    print(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    precision_v = precision(y_true, y_pred)
    recall_v = recall(y_true, y_pred)
    if precision_v + recall_v == 0:
        return 0
    return (2.0*precision_v*recall_v)/(precision_v+recall_v)


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


def get_model():

    model = Sequential()
    # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size)))
    model.add(Dense(sigmoid_units, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model


def get_ddi_data(dirs=["data/ddi2013Train/DrugBank/", "data/ddi2013Train/MedLine/"]):
    dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    instances = np.empty((0, max_sentence_length, embbed_size))
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes = get_ddi_sdp_instances(dir)
        dir_instances = np.array(dir_instances)
        dir_instances = sequence.pad_sequences(dir_instances, maxlen=max_sentence_length)
        dir_classes = np.array(dir_classes)

        labels += dir_labels
        print(instances.shape, dir_instances.shape)
        instances = np.concatenate((instances, dir_instances), axis=0)
        classes = np.concatenate((classes, dir_classes), axis=0)
    return labels, instances, classes





def main():
    n_epochs = 20
    batch_size = 100
    validation_split = 0.1


    if sys.argv[1] == "preprocessing":
        labels, X_train, classes = get_ddi_data()
        np.save(sys.argv[2] + "_labels.npy", labels)
        np.save(sys.argv[2] + "_x.npy", X_train)
        np.save(sys.argv[2] + "_y.npy", classes)

    elif sys.argv[1] == "train":
        model = get_model()
        X_train = np.load(sys.argv[2] + "_x.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=None)
        # print(Y_train)
        if len(sys.argv) > 3:
            X_test = np.load(sys.argv[3] + "_x.npy")
            Y_test = np.load(sys.argv[3] + "_y.npy")
            Y_test = to_categorical(Y_test, num_classes=None)
            test_labels = np.load(sys.argv[3] + "_labels.npy")

            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=n_epochs, batch_size=batch_size)
        else:
            model.fit(X_train, Y_train, validation_split=validation_split, epochs=n_epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()
