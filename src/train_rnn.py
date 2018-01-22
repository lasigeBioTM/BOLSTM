import random
import sys
import logging
import os
logging.basicConfig(level=10)
import collections
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from gensim.models.keyedvectors import KeyedVectors
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from chebi_path import load_chebi
from models import get_model, get_xu_model, embbed_size, max_sentence_length, max_ancestors_length, n_classes,\
    words_channel, wordnet_channel, ancestors_channel

n_inputs = 0
if words_channel:
    n_inputs += 2
if wordnet_channel:
    n_inputs += 2
if ancestors_channel:
    n_inputs += 2

DATA_DIR = "data/"
n_epochs = 10
batch_size = 128
validation_split = 0.1
PRINTERRORS = True

# https://github.com/keras-team/keras/issues/853#issuecomment-343981960

def write_plots(history):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model eval')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_acc.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model_loss.png")

def get_glove_vectors():
    embeddings_vectors = {"": np.zeros(embbed_size, dtype='float32')} # words -> vector
    embedding_indexes = {"": 0}
    # load embeddings indexes: word -> coefs
    f = open(os.path.join(DATA_DIR, 'glove.6B.300d.txt'))
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
        embedding_indexes[word] = i+1
        #print(i)
    print(len(embedding_indexes))
    f.close()
    print('Found %s word vectors.' % len(embeddings_vectors))

    # assemble the embedding_weights in one numpy array
    #n_symbols = len(embedding_indexes) + 1  # adding 1 to account for 0th index (for masking)
    n_symbols = len(embedding_indexes) + 1
    embedding_weights = np.zeros((n_symbols, embbed_size))
    for word, index in embedding_indexes.items():
        #print(index, n_symbols)
        embedding_weights[index, :] = embeddings_vectors[word]

    return embedding_indexes, embedding_weights


def get_w2v():
    embeddings_vectors = {}  # words -> vector
    embedding_indexes = {}
    #word_vectors = KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.txt', binary=False)  # C text format
    word_vectors = KeyedVectors.load_word2vec_format('data/PubMed-w2v.bin', binary=True)  # C text format
    return word_vectors

def get_wordnet_indexes():
    embedding_indexes = {}
    with open("sst-light-0.4/DATA/WNSS_07.TAGSET", 'r') as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            if l.startswith("I-"):
                continue
            embedding_indexes[l.strip().split("-")[-1]] = i
            i += 1
    # print(embedding_indexes)
    return embedding_indexes


def preprocess_sequences_glove(x_data, embeddings_index):
    data = []
    for i, seq in enumerate(x_data):
        #for w in seq:
            #if w.lower() not in embeddings_index:
            #    print("word not in index: {}".format(w.lower()))
        #print(seq)
        #idxs = [embeddings_index.get(w.lower()) for w in seq if w.lower() in embeddings_index]
        idxs = [embeddings_index.get(w) for w in seq if w in embeddings_index]
        #idxs = [embeddings_index.vocab[w.lower()].index for w in seq if w.lower() in embeddings_index.vocab]
        if None in idxs:
            print(seq, idxs)
        # print(idxs)
        data.append(idxs)
    #print(data)
    data = pad_sequences(data, maxlen=max_sentence_length)
    return data


def preprocess_sequences(x_data, embeddings_index):
    data = []
    for i, seq in enumerate(x_data):
        #for w in seq:
            #if w.lower() not in embeddings_index:
            #    print("word not in index: {}".format(w.lower()))
        #print(seq)
        #idxs = [embeddings_index.get(w.lower()) for w in seq if w.lower() in embeddings_index]
        idxs = [embeddings_index.vocab[w.lower()].index for w in seq if w.lower() in embeddings_index.vocab]
        if None in idxs:
            print(seq, idxs)
        #print(idxs)
        data.append(idxs)
    #print(data)
    data = pad_sequences(data, maxlen=max_sentence_length, padding='post')
    return data

def preprocess_ids(x_data, id_to_index):
    # process a sequence of ontology:IDs, so a embedding index is not necessary
    data = []
    for i, seq in enumerate(x_data):
        # print(seq)
        idxs = [id_to_index[d.replace("_", ":")] for d in seq if d and d.startswith("CHEBI")]
        data.append(idxs)
    data = pad_sequences(data, maxlen=max_ancestors_length)
    return data



class Metrics(Callback):
    def __init__(self, labels, **kwargs):
        self.labels = labels
        super(Metrics, self).__init__()


    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        #print(dir(self.model))
        #print(len(self.validation_data))
        val_predict = (np.asarray(self.model.predict([self.validation_data[i] for i in range(n_inputs)],
                                                     ))).round()
        val_targ = self.validation_data[n_inputs]
        #val_targ = self.validation_data[1]
        #probs = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]],
        #                                              ))
        _val_f1 = f1_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_recall = recall_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _val_precision = precision_score(val_targ[...,1:], val_predict[...,1:], average='micro')
        _confusion_matrix = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        s = "predicted not false: {}/{}\n{}\n".format(len([x for x in val_predict if np.argmax(x) != 0]),
                                                len([x for x in val_targ if x[0] < 0.5]),
                                                    _confusion_matrix)
        print("\n{} VAL_f1:{:6.3f} VAL_p:{:6.3f} VAL_r{:6.3f}\n".format(s, _val_f1, _val_precision, _val_recall),)

        if PRINTERRORS:
            for i in range(len(val_targ)):
                 true_label = np.argmax(val_targ[i])
                 predicted = np.argmax(val_predict[i])
                 if predicted != true_label:
                     error_type = "wrong label"
                     if true_label == 0:
                         error_type = "FP"
                     elif predicted == 0:
                         error_type = "FN"
                     if error_type != "FN":
                         #print("{}: {}->{}; inputs: {}".format(error_type, true_label, predicted,
                         #                                  str([self.validation_data[j][i] for j in range(n_inputs)])))
                         print("{}: {}->{}; inputs: {}".format(error_type, true_label, predicted, self.labels[i]))
            #print(val_predict, val_targ)
            #print("true not false: {}".format()
        print()
        return






def get_ddi_data(dirs=["data/ddi2013Train/DrugBank/", "data/ddi2013Train/MedLine/"]):

    from parse_ddi import get_ddi_sdp_instances

    #dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    #instances = np.empty((0, max_sentence_length, embbed_size))
    #instances = np.empty((0, max_sentence_length))
    left_instances = []
    right_instances = []
    common_ancestors = []
    left_ancestors = []
    right_ancestors = []
    left_wordnet = []
    right_wordnet = []
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes, dir_common, dir_ancestors, dir_wordnet = get_ddi_sdp_instances(dir)
        #dir_instances = np.array(dir_instances)
        #print(dir_instances)
        #dir_instances = sequence.pad_sequences(dir_instances, maxlen=max_sentence_length)
        dir_classes = np.array(dir_classes)

        labels += dir_labels
        #print(instances.shape, dir_instances.shape)
        #instances = np.concatenate((instances, dir_instances), axis=0)
        left_instances += dir_instances[0]
        right_instances += dir_instances[1]
        common_ancestors += dir_common
        left_ancestors += dir_ancestors[0]
        right_ancestors += dir_ancestors[1]
        left_wordnet += dir_wordnet[0]
        right_wordnet += dir_wordnet[1]
        classes = np.concatenate((classes, dir_classes), axis=0)

    return labels, (left_instances, right_instances), classes, common_ancestors,\
           (left_ancestors, right_ancestors), (left_wordnet, right_wordnet)





def main():

    if sys.argv[1] == "preprocessing":
        from parse_semeval8 import get_semeval8_sdp_instances
        #labels, X_train, classes = get_ddi_data(sys.argv[3:])
        train = True
        if "test" in sys.argv[3].lower():
            train= False
        # TODO: generalize text pre-processing
        if sys.argv[2] == "semeval8":
            labels, X_train, classes, X_train_ancestors, X_train_wordnet = get_semeval8_sdp_instances(sys.argv[4:], train=train)
            print(len(X_train))
            print(len(X_train[0]))
        elif sys.argv[2] == "ddi":
            labels, X_train, classes, X_train_ancestors, X_train_subpaths, X_train_wordnet = get_ddi_data(sys.argv[4:])
            print(len(X_train))
            np.save(sys.argv[3] + "_x_ancestors.npy", X_train_ancestors)
            np.save(sys.argv[3] + "_x_subpaths.npy", X_train_subpaths)
        np.save(sys.argv[3] + "_labels.npy", labels)
        np.save(sys.argv[3] + "_x_words.npy", X_train)
        np.save(sys.argv[3] + "_x_wordnet.npy", X_train_wordnet)
        np.save(sys.argv[3] + "_y.npy", classes)

    elif sys.argv[1] == "train":
        if os.path.isfile("model.json"):
            os.remove("model.json")
        if os.path.isfile("model.h5"):
            os.remove("model.h5")
        train_labels = np.load(sys.argv[2] + "_labels.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        list_order = np.arange(len(Y_train))
        random.shuffle(list_order)

        Y_train = Y_train[list_order]
        train_labels = train_labels[list_order]
        print("train order:", list_order)
        # print(emb_index)
        inputs = {}
        if words_channel:
            #emb_index, emb_matrix = get_glove_vectors()
            word_vectors = get_w2v()
            w2v_layer = word_vectors.get_keras_embedding(train_embeddings=False)
            X_words_train = np.load(sys.argv[2] + "_x_words.npy")
            #X_words_left = preprocess_sequences_glove(X_words_train[0], emb_index)
            #X_words_right = preprocess_sequences_glove(X_words_train[1], emb_index)

            X_words_left = preprocess_sequences([["drug"] + x[1:] for x in X_words_train[0]], word_vectors)
            X_words_right = preprocess_sequences([x[:-1] + ["drug"] for x in X_words_train[1]], word_vectors)
            #X_words_left = preprocess_sequences(X_words_train[0], word_vectors)
            #X_words_right = preprocess_sequences(X_words_train[1], word_vectors)

            # skip root word

            #print(np.array(X_words_train[1]))
            #X_words_train = np.concatenate((np.array(X_words_train[0]), np.array(X_words_train[1])), 1)
            #X_words_train = [X_words_train[0][i] + X_words_train[1][i] for i in range(len(X_words_train[0]))]
            #X_words_train = preprocess_sequences(X_words_train, word_vectors)

            inputs["left_words"] = X_words_left[list_order]
            inputs["right_words"] = X_words_right[list_order]
            #inputs["words"] = X_words_train[list_order]
        else:
            emb_matrix = None
            w2v_layer = None

        if wordnet_channel:
            wn_index = get_wordnet_indexes()
            X_wordnet_train = np.load(sys.argv[2] + "_x_wordnet.npy")
            X_wn_left = preprocess_sequences_glove(X_wordnet_train[0], wn_index)
            X_wn_right = preprocess_sequences_glove(X_wordnet_train[1], wn_index)
            inputs["left_wordnet"] = X_wn_left[list_order]
            inputs["right_wordnet"] = X_wn_right[list_order]
        else:
            wn_index = None

        if ancestors_channel:
            is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
            X_subpaths_train = np.load(sys.argv[2] + "_x_subpaths.npy")
            X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
            X_ids_left = preprocess_ids(X_subpaths_train[0], id_to_index)
            X_ids_right = preprocess_ids(X_subpaths_train[1], id_to_index)
            X_ancestors = preprocess_ids(X_ancestors_train, id_to_index)
            #X_ancestors_train = np.concatenate((X_ids_left, X_ids_right[..., 1:]), 1)
            inputs["left_ancestors"] = X_ids_left[list_order]
            inputs["right_ancestors"] = X_ids_right[list_order]
            inputs["common_ancestors"] = X_ancestors[list_order]
        else:
            id_to_index = None

        model = get_model(w2v_layer, wn_index, id_to_index)

        #model = get_words_model(emb_matrix)
        #model = get_xu_model(emb_matrix)


        #print(inputs)
        if len(sys.argv) > 3:
            Y_labels = np.load(sys.argv[3] + "_labels.npy")
            Y_test = np.load(sys.argv[3] + "_y.npy")
            Y_test = to_categorical(Y_test, num_classes=n_classes)
            val_inputs = {}

            if words_channel:
                X_words_test = np.load(sys.argv[3] + "_x_words.npy")
                X_words_test_left = preprocess_sequences([["drug"] + x[1:]  for x in X_words_test[0]], word_vectors)
                X_words_test_right = preprocess_sequences([x[:-1] + ["drug"] for x in X_words_test[1]], word_vectors)
                val_inputs["left_words"] = X_words_test_left
                val_inputs["right_words"] = X_words_test_right

            if wordnet_channel:
                X_wordnet_test = np.load(sys.argv[3] + "_x_wordnet.npy")
                X_wn_test_left = preprocess_sequences_glove(X_wordnet_test[0], wn_index)
                X_wn_test_right = preprocess_sequences_glove(X_wordnet_test[1], wn_index)
                val_inputs["left_wordnet"] = X_wn_test_left
                val_inputs["right_wordnet"] = X_wn_test_right

            if ancestors_channel:
                X_subpaths_test = np.load(sys.argv[3] + "_x_subpaths.npy")
                X_ancestors_test = np.load(sys.argv[3] + "_x_ancestors.npy")
                X_ids_left = preprocess_ids(X_subpaths_test[0], id_to_index)
                X_ids_right = preprocess_ids(X_subpaths_test[1], id_to_index)
                X_ancestors = preprocess_ids(X_ancestors_test, id_to_index)
                # X_ancestors_train = np.concatenate((X_ids_left, X_ids_right[..., 1:]), 1)
                val_inputs["left_ancestors"] = X_ids_left
                val_inputs["right_ancestors"] = X_ids_right
                val_inputs["common_ancestors"] = X_ancestors

            #X_ancestors_test = np.load(sys.argv[3] + "_x_ancestors.npy")

            val_outputs = {"output": Y_test}

            test_labels = np.load(sys.argv[3] + "_labels.npy")

            #model.fit(X_words_train, Y_train, validation_data=(X_test, Y_test), epochs=n_epochs,
            #          batch_size=batch_size, callbacks=[metrics], verbose=2)
            metrics = Metrics(Y_labels)
            history = model.fit(inputs,
                                {"output": Y_train}, validation_data=(val_inputs, val_outputs), epochs=n_epochs,
                                batch_size=batch_size, verbose=2, callbacks=[metrics])
            write_plots(history)

        else:
            metrics = Metrics(train_labels)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, min_lr=0.001)
            history = model.fit(inputs,
                      {"output": Y_train}, validation_split=validation_split, epochs=n_epochs,
                      batch_size=batch_size, verbose=2, callbacks=[metrics, reduce_lr ])
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

        inputs = {}

        if words_channel:
            #emb_index, emb_matrix = get_glove_vectors()
            #emb_index, emb_matrix = None, None
            word_vectors = get_w2v()
            X_words_test = np.load(sys.argv[2] + "_x_words.npy")
            X_words_test_left = preprocess_sequences([["drug"] + x[1:] for x in X_words_test[0]], word_vectors)
            X_words_test_right = preprocess_sequences([x[:-1] + ["drug"] for x in X_words_test[1]], word_vectors)
            #X_words_test = [X_words_test[0][i] + X_words_test[1][i] for i in range(len(X_words_test[0]))]
            #X_words_test = preprocess_sequences(X_words_test, word_vectors)
            inputs["left_words"] = X_words_test_left
            inputs["right_words"] = X_words_test_right
            #inputs["words"] = X_words_test
            #X_words_test_left = preprocess_sequences_glove(X_words_test[0], emb_index)
            #X_words_test_right = preprocess_sequences_glove(X_words_test[1], emb_index)
            #X_words_test = np.concatenate((X_words_test_left, X_words_test_right[..., 1:]), 1)
            #inputs["left_words"] = X_words_test_left
            #inputs["right_words"] = X_words_test_right
        if wordnet_channel:
            wn_index = get_wordnet_indexes()
            X_wn_test = np.load(sys.argv[2] + "_x_wordnet.npy")
            X_wordnet_test_left = preprocess_sequences_glove(X_wn_test[0], wn_index)
            X_wordnet_test_right = preprocess_sequences_glove(X_wn_test[1], wn_index)
            inputs["left_wordnet"] = X_wordnet_test_left
            inputs["right_wordnet"] = X_wordnet_test_right
        if ancestors_channel:
            is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
            X_ancestors_test = np.load(sys.argv[2] + "_x_ancestors.npy")
            X_subpaths_test = np.load(sys.argv[2] + "_x_subpaths.npy")
            X_ids_left = preprocess_ids(X_subpaths_test[0], id_to_index)
            X_ids_right = preprocess_ids(X_subpaths_test[1], id_to_index)
            X_ancestors = preprocess_ids(X_ancestors_test, id_to_index)
            inputs["left_ancestors"] = X_ids_left
            inputs["right_ancestors"] = X_ids_right
            inputs["common_ancestors"] = X_ancestors

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
        scores = loaded_model.predict(inputs)
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

    elif sys.argv[1] == "showdata":
        limit = int(sys.argv[3])
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
        X_subpaths_train = np.load(sys.argv[2] + "_x_subpaths.npy")
        X_wordnet_train = np.load(sys.argv[2] + "_x_wordnet.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        train_labels = np.load(sys.argv[2] + "_labels.npy")


        print("labels:")
        print(train_labels[:limit])
        print()
        print("left words:")
        print(X_words_train[0][:limit])
        print("right words:")
        print(X_words_train[1][:limit])
        print()
        print("chebi ancestors:")
        #print(len(X_subpaths_train))
        #print(len(X_ancestors_train[0]))
        print(X_ancestors_train[:limit])
        print()
        print("chebi subpaths")
        print("left")
        print(X_subpaths_train[0][:limit])
        print("right")
        print(X_subpaths_train[1][:limit])
        print()

        print("wordnet:")
        print(X_wordnet_train[0][:limit])
        print(X_wordnet_train[1][:limit])
        print()

        print("classes")
        print(Y_train[:limit])

        print("class distribution")
        counter = collections.Counter(Y_train)
        print(counter)
        print(counter[1] + counter[2] + counter[3] + counter[4])



if __name__ == "__main__":
    main()
