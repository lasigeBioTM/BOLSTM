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
from keras.callbacks import Callback, LambdaCallback, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from chebi_path import load_chebi
from models import get_model, get_xu_model, embbed_size, max_sentence_length, max_ancestors_length, n_classes
    #words_channel, wordnet_channel, common_ancestors_channel, concat_ancestors_channel




DATA_DIR = "data/"
n_epochs = 500
batch_size = 128
validation_split = 0.2
PRINTERRORS = False

# https://github.com/keras-team/keras/issues/853#issuecomment-343981960

def write_plots(history, modelname):
    """
    Write plots regarding model training
    :param history: history object returned by fit function
    :param modelname: name of model to be used as part of filename
    """
    plt.figure()
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model eval')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("{}_acc.png".format(modelname))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("{}_loss.png".format(modelname))

def get_glove_vectors(filename='glove.6B.300d.txt'):
    """
    Open
    :param filename: file containing the word vectors trained with glove
    :return: index of each word and vectors
    """
    embeddings_vectors = {"": np.zeros(embbed_size, dtype='float32')} # words -> vector
    embedding_indexes = {"": 0}
    # load embeddings indexes: word -> coefs
    f = open(os.path.join(DATA_DIR, filename))
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


def get_w2v(filename='data/PubMed-w2v.bin'):
    """
    Open Word2Vec file using gensim package
    :return: word vectors in KeyedVectors gensim object
    """
    #word_vectors = KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.txt', binary=False)  # C text format
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True)  # C text format
    return word_vectors

def get_wordnet_indexes():
    """
    Get the wordnet classes considered by SST, ignoring BI tags
    :return: embedding_indexes: tag -> index
    """
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
    """
    Replace words in x_data with index of word in embeddings_index and pad sequence
    :param x_data: list of sequences of words (sentences)
    :param embeddings_index: word -> index in embedding matrix
    :return: matrix to be used as training data
    """
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
    """
    Replace words in x_data with index of word in embeddings_index and pad sequence
    :param x_data: list of sequences of words (sentences)
    :param embeddings_index: word -> index in embedding matrix
    :return: matrix to be used as training data
    """
    data = []
    for i, seq in enumerate(x_data):
        #for w in seq:
            #if w.lower() not in embeddings_index:
            #    print("word not in index: {}".format(w.lower()))
        #print(seq)
        #idxs = [embeddings_index.get(w.lower()) for w in seq if w.lower() in embeddings_index]
        #idxs = [embeddings_index.vocab[w.lower()].index for w in seq if w.lower() in embeddings_index.vocab]
        idxs= []
        for w in seq:
            if w.lower() in embeddings_index.vocab:
                idxs.append(embeddings_index.vocab[w.lower()].index)
        if None in idxs:
            print(seq, idxs)
        #print(idxs)
        data.append(idxs)
    #print(data)
    data = pad_sequences(data, maxlen=max_sentence_length, padding='post')
    return data

def preprocess_ids(x_data, id_to_index, maxlen):
    """
    process a sequence of ontology:IDs, so an embedding index is not necessary
    :param x_data:
    :param id_to_index:
    :param maxlen:
    :return: matrix to be used as training data
    """
    #
    data = []
    for i, seq in enumerate(x_data):
        # print(seq)
        idxs = [id_to_index[d.replace("_", ":")] for d in seq if d and d.startswith("CHEBI")]
        data.append(idxs)
    data = pad_sequences(data, maxlen=maxlen)
    return data



class Metrics(Callback):
    """
    Implementation of P, R and F1 metrics for fit function callback
    """
    def __init__(self, labels, words, n_inputs, **kwargs):
        self.labels = labels
        self.words_left = words[0]
        self.words_right = words[1]
        self.n_inputs = n_inputs
        self._val_f1 = 0
        self._val_recall = 0
        self._val_precision = 0
        super(Metrics, self).__init__()


    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        #print(dir(self.model))
        #print(len(self.validation_data))
        val_predict = (np.asarray(self.model.predict([self.validation_data[i] for i in range(self.n_inputs)],
                                                     ))).round()
        val_targ = self.validation_data[self.n_inputs]
        #val_targ = self.validation_data[1]
        #probs = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]],
        #                                              ))
        self._val_f1 = f1_score(val_targ[...,1:], val_predict[...,1:], average='macro')
        self._val_recall = recall_score(val_targ[...,1:], val_predict[...,1:], average='macro')
        self._val_precision = precision_score(val_targ[...,1:], val_predict[...,1:], average='macro')
        _confusion_matrix = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
        self.val_f1s.append(self._val_f1)
        self.val_f1s.append(self._val_f1)
        self.val_recalls.append(self._val_recall)
        self.val_precisions.append(self._val_precision)
        s = "predicted not false: {}/{}\n{}\n".format(len([x for x in val_predict if np.argmax(x) != 0]),
                                                len([x for x in val_targ if x[0] < 0.5]),
                                                    _confusion_matrix)
        print("\n{} VAL_f1:{:6.3f} VAL_p:{:6.3f} VAL_r{:6.3f}\n".format(s, self._val_f1,
                                                                        self._val_precision, self._val_recall),)

        return



def get_ddi_data(dirs=["data/ddi2013Train/DrugBank/", "data/ddi2013Train/MedLine/"]):
    """
    Generate data instances for the documents in the subdirectories of the corpus
    :param dirs: list of directories to be scanned for XML files
    :return: column vectors where each element corresponds to a label or a sequence of values of a particular
    data instance.
    """

    # import function to process a directory with XML DDI files
    from parse_ddi import get_ddi_sdp_instances

    #dirs = ["data/DDICorpus/Test/DDIExtraction/DrugBank/", "data/DDICorpus/Test/DDIExtraction/MedLine/"]
    labels = []
    #instances = np.empty((0, max_sentence_length, embbed_size))
    #instances = np.empty((0, max_sentence_length))
    left_instances = [] # word indexes
    right_instances = []
    common_ancestors = [] # ontology IDs
    left_ancestors = []
    right_ancestors = []
    left_wordnet = [] # wordnet IDs
    right_wordnet = []
    all_pos_gv = set() # anti positive governors
    all_neg_gv = set()
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes, dir_common, dir_ancestors, dir_wordnet, neg_gv, pos_gv = get_ddi_sdp_instances(dir)
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

        all_pos_gv.update(pos_gv)
        all_neg_gv.update(neg_gv)

    return labels, (left_instances, right_instances), classes, common_ancestors,\
           (left_ancestors, right_ancestors), (left_wordnet, right_wordnet)

def main():
    if sys.argv[1] == "preprocessing":
        # generate data instances and write to disk as numpy arrays
        # args: corpus_type (semeval8 or ddi) corpus_name1 (corpus_name2) (...)
        # e.g. python3 src/train_rnn.py preprocessing ddi temp/dditrain data/DDICorpus/Train/MedLine/ data/DDICorpus/Train/DrugBank/
        train = True
        if "test" in sys.argv[3].lower():
            train= False
        # TODO: generalize text pre-processing
        if sys.argv[2] == "semeval8":
            from parse_semeval8 import get_semeval8_sdp_instances
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
        # open numpy arrays with data and train model

        # number of input channels is determined by args after corpus_name and model_name
        n_inputs = 0
        if "words" in sys.argv[4:]:
            n_inputs += 2
        if "wordnet" in sys.argv[4:]:
            n_inputs += 2
        if "concat_ancestors" in sys.argv[4:]:
            n_inputs += 2
        if "common_ancestors" in sys.argv[4:]:
            n_inputs += 1

        # remove previous model files
        if os.path.isfile("models/{}.json".format(sys.argv[3])):
            os.remove("models/{}.json".format(sys.argv[3]))
        if os.path.isfile("models/{}.h5".format(sys.argv[3])):
            os.remove("models/{}.h5".format(sys.argv[3]))
        train_labels = np.load(sys.argv[2] + "_labels.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        # get random instance order (to shuffle docs types and labels)
        list_order = np.arange(len(Y_train))
        random.seed(1)
        random.shuffle(list_order)
        Y_train = Y_train[list_order]
        train_labels = train_labels[list_order]
        print("train order:", list_order)

        # store features in this dictionry according to args
        inputs = {}
        if "words" in sys.argv[4:]:
            #emb_index, emb_matrix = get_glove_vectors()
            word_vectors = get_w2v()
            w2v_layer = word_vectors.get_keras_embedding(train_embeddings=False)
            X_words_train = np.load(sys.argv[2] + "_x_words.npy")
            #X_words_left = preprocess_sequences_glove(X_words_train[0], emb_index)
            #X_words_right = preprocess_sequences_glove(X_words_train[1], emb_index)

            #X_words_left = preprocess_sequences([["drug"] + x[1:] for x in X_words_train[0]], word_vectors)
            #X_words_right = preprocess_sequences([x[:-1] + ["drug"] for x in X_words_train[1]], word_vectors)
            X_words_left = preprocess_sequences(X_words_train[0], word_vectors)
            X_words_right = preprocess_sequences(X_words_train[1], word_vectors)

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

        if "wordnet" in sys.argv[4:]:
            wn_index = get_wordnet_indexes()
            X_wordnet_train = np.load(sys.argv[2] + "_x_wordnet.npy")
            X_wn_left = preprocess_sequences_glove(X_wordnet_train[0], wn_index)
            X_wn_right = preprocess_sequences_glove(X_wordnet_train[1], wn_index)
            inputs["left_wordnet"] = X_wn_left[list_order]
            inputs["right_wordnet"] = X_wn_right[list_order]
        else:
            wn_index = None

        if "concat_ancestors" in sys.argv[4:] or "common_ancestors" in sys.argv[4:]:
            is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
            X_subpaths_train = np.load(sys.argv[2] + "_x_subpaths.npy")
            X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
            X_ids_left = preprocess_ids(X_subpaths_train[0], id_to_index, max_ancestors_length)
            X_ids_right = preprocess_ids(X_subpaths_train[1], id_to_index, max_ancestors_length)
            X_ancestors = preprocess_ids(X_ancestors_train, id_to_index, max_ancestors_length*2)
            #X_ancestors_train = np.concatenate((X_ids_left, X_ids_right[..., 1:]), 1)
            inputs["left_ancestors"] = X_ids_left[list_order]
            inputs["right_ancestors"] = X_ids_right[list_order]
            inputs["common_ancestors"] = X_ancestors[list_order]
        else:
            id_to_index = None

        model = get_model(w2v_layer, sys.argv[4:], wn_index, id_to_index)

        # alternative models
        #model = get_words_model(emb_matrix)
        #model = get_xu_model(emb_matrix)



        metrics = Metrics(train_labels, X_words_train, n_inputs)
        checkpointer = ModelCheckpoint(filepath="models/{}.h5".format(sys.argv[3]), verbose=1, save_best_only=True)
        history = model.fit(inputs,
                  {"output": Y_train}, validation_split=validation_split, epochs=n_epochs,
                  batch_size=batch_size, verbose=2, callbacks=[metrics, checkpointer])

                                                               #keras.callbacks.EarlyStopping(patience=3)])
        #history = model.fit({"input": X_words_train}, {"output": Y_train},
        #                    validation_split=validation_split, epochs=n_epochs,
        #                    batch_size=batch_size, verbose=2, callbacks=[metrics])
        write_plots(history, sys.argv[3])

        # serialize model to JSON
        model_json = model.to_json()
        with open("models/{}.json".format(sys.argv[3]), "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        #model.save_weights("{}.h5".format(sys.argv[3]))
        print("Saved model to disk")

    elif sys.argv[1] == "predict":
        # open numpy files according to the input channels specified, open model files and apply model to data
        inputs = {}

        if "words" in sys.argv[4:]:
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
        if "wordnet" in sys.argv[4:]:
            wn_index = get_wordnet_indexes()
            X_wn_test = np.load(sys.argv[2] + "_x_wordnet.npy")
            X_wordnet_test_left = preprocess_sequences_glove(X_wn_test[0], wn_index)
            X_wordnet_test_right = preprocess_sequences_glove(X_wn_test[1], wn_index)
            inputs["left_wordnet"] = X_wordnet_test_left
            inputs["right_wordnet"] = X_wordnet_test_right

        if "common_ancestors" in sys.argv[4:] or "concat_ancestors" in sys.argv[4:]:
            is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
            X_ancestors_test = np.load(sys.argv[2] + "_x_ancestors.npy")
            X_subpaths_test = np.load(sys.argv[2] + "_x_subpaths.npy")
            X_ids_left = preprocess_ids(X_subpaths_test[0], id_to_index, max_ancestors_length)
            X_ids_right = preprocess_ids(X_subpaths_test[1], id_to_index, max_ancestors_length)
            X_ancestors = preprocess_ids(X_ancestors_test, id_to_index, max_ancestors_length*2)
            inputs["left_ancestors"] = X_ids_left
            inputs["right_ancestors"] = X_ids_right
            inputs["common_ancestors"] = X_ancestors

        # load json and create model
        json_file = open('models/{}.json'.format(sys.argv[3]), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("models/{}.h5".format(sys.argv[3]))
        print("Loaded model {} from disk".format(sys.argv[3]))


        test_labels = np.load(sys.argv[2] + "_labels.npy")

        #scores = loaded_model.predict(X_words_test)
        scores = loaded_model.predict(inputs)
        # write results to file
        with open("{}_{}_results.txt".format(sys.argv[3], sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

    elif sys.argv[1] == "dummy_predict":
        test_labels = np.load(sys.argv[2] + "_labels.npy")
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], "3")) + "\n")

    elif sys.argv[1] == "showdata":
        if sys.argv[3].isdigit(): # limit this number of instances
            limit = int(sys.argv[3])
            target = None
        else:
            target = sys.argv[3] # print instanes with this entity
            limit = None
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
        X_subpaths_train = np.load(sys.argv[2] + "_x_subpaths.npy")
        X_wordnet_train = np.load(sys.argv[2] + "_x_wordnet.npy")
        Y_train = np.load(sys.argv[2] + "_y.npy")
        train_labels = np.load(sys.argv[2] + "_labels.npy")

        if limit:
            print("labels:")
            print(train_labels[:limit])
            print()
            print("left words:")
            print(X_words_train[0][:limit])
            print("right words:")
            print(X_words_train[1][:limit])
            print()
            print("chebi ancestors:")
            print(len(X_subpaths_train))
            print(len(X_ancestors_train[0]))
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
            analyze_entity_distances(train_labels, Y_train, X_words_train)
            print("class distribution")
            counter = collections.Counter(Y_train)
            print(counter)
            print(counter[1] + counter[2] + counter[3] + counter[4])
            #analyze_sdps(Y_train, X_words_train)
            # print([(X_words_train[0][i], X_words_train[1][i]) for i, l in enumerate(train_labels) if 'DDI-DrugBank.d769.s2.e1' in l])
            #analyze_lens(Y_train, X_words_train, X_wordnet_train, X_subpaths_train, X_ancestors_train)

        else:
            train_labels = [(i, t) for i, t in enumerate(train_labels) if target in t]
            for (i, l) in train_labels:
                print()
                print()
                print(l)
                print("left words:")
                print(X_words_train[0][i])
                print("right words:")
                print(X_words_train[1][i])
                #print()
                print("classes")
                print(Y_train[i])
                print("wordnet:")
                print(X_wordnet_train[0][i])
                print(X_wordnet_train[1][i])
                print()
                print("chebi ancestors:")
                #print(len(X_subpaths_train))
                #print(len(X_ancestors_train[0]))
                print(X_ancestors_train[i], len(X_ancestors_train[i]))
                print()
                print("chebi subpaths")
                print("left")
                print(X_subpaths_train[0][i], len(X_subpaths_train[0][i]))
                print("right")
                print(X_subpaths_train[1][i], len(X_subpaths_train[1][i]))
                print()







def analyze_sdps(Y_train, X_words_train):
    pos_sdps = {}
    neg_sdps = {}
    for i, p in enumerate(Y_train):
        sdp_len = len(X_words_train[0][i]) + len(X_words_train[1][i])
        if p != 0:
            pos_sdps[sdp_len] = pos_sdps.get(sdp_len, 0) + 1
        else:
            neg_sdps[sdp_len] = neg_sdps.get(sdp_len, 0) + 1
    #print("positive SDPs with length shorter than {}: {}".format(threshold, pos_short_sdps))
    #print(short_sdps/len([y for y in Y_train if y != 0]))
    #print("negative SDPs with length shorter than {}: {}".format(threshold, short_sdps - pos_short_sdps))
    #print((short_sdps - pos_short_sdps) / len([y for y in Y_train if y == 0]))
    print("pos sdps:", sum(pos_sdps.values()))
    od = collections.OrderedDict(sorted(pos_sdps.items()))
    for k, v in od.items():
        print(k, v, round(v / sum(pos_sdps.values()), 3), round(v/(v + neg_sdps.get(k, 0)), 3))
    print()
    print("neg sdps:", sum(neg_sdps.values()))
    od = collections.OrderedDict(sorted(neg_sdps.items()))
    for k, v in od.items():
        print(k, v, round(v / sum(neg_sdps.values()), 3), round(v/(v + pos_sdps.get(k, 0)), 3))
    print()


def analyze_entity_distances(train_labels, Y_train, X_words_train):
    entity_id_max = 10
    print()
    print("entity id distribution of entities in positive pairs")
    c = {}
    for i, p in enumerate(train_labels):
        if Y_train[i] == 0:
            continue
        eid1 = int(p[0].split("e")[-1])
        eid2 = int(p[1].split("e")[-1])
        if eid1 not in c:
            c[eid1] = 0
        if eid2 not in c:
            c[eid2] = 0
        c[eid1] += 1
        c[eid2] += 2
    for e in c:
        print(e, c[e], c[e] / sum(c.values()))
    print("percentage of entities with id>{}".format(entity_id_max))
    print(sum([c[e] / sum(c.values()) for e in c if e > entity_id_max]))
    print()

    print()
    print("entity id distribution of entities in all pairs")
    c = {}
    for i, p in enumerate(train_labels):
        eid1 = int(p[0].split("e")[-1])
        eid2 = int(p[1].split("e")[-1])
        if eid1 not in c:
            c[eid1] = 0
        if eid2 not in c:
            c[eid2] = 0
        c[eid1] += 1
        c[eid2] += 2
    for e in c:
        print(e, c[e], c[e] / sum(c.values()))
    print("percentage of entities with id>{}".format(entity_id_max))
    print(sum([c[e] / sum(c.values()) for e in c if e > entity_id_max]))
    print()

    digits1 = [x for x in X_words_train[0] if any([y.replace(".", "").isdigit() for y in x])]
    digits2 = [x for x in X_words_train[1] if any([y.replace(".", "").isdigit() for y in x])]
    print(digits1, digits2)
    print(len(digits1), len(digits2))
    print(len(X_words_train[0]), len(X_words_train[1]))

def analyze_lens(Y_train, X_words_train, X_wordnet_train, X_subpaths_train, X_ancestors_train):
    pos = 0
    neg = 0
    pos_word_left = 0
    pos_word_right = 0
    neg_word_left = 0
    neg_word_right = 0
    pos_wordnet_left = 0
    pos_wordnet_right = 0
    neg_wordnet_left = 0
    neg_wordnet_right = 0
    pos_chebi_left = 0
    pos_chebi_right = 0
    neg_chebi_left = 0
    neg_chebi_right = 0
    pos_common = 0
    neg_common = 0
    
    for i, y in enumerate(Y_train):
        if y == 0:
            neg += 1
            neg_word_left += len(X_words_train[0][i])
            neg_word_right += len(X_words_train[1][i])
            neg_wordnet_left += len(X_wordnet_train[0][i])
            neg_wordnet_right += len(X_wordnet_train[1][i])
            neg_chebi_left += len(X_subpaths_train[0][i])
            neg_chebi_right += len(X_subpaths_train[1][i])
            neg_common += len(X_ancestors_train[i])
        else:
            pos_word_left += len(X_words_train[0][i])
            pos_word_right += len(X_words_train[1][i])
            pos_wordnet_left += len(X_wordnet_train[0][i])
            pos_wordnet_right += len(X_wordnet_train[1][i])
            pos_chebi_left += len(X_subpaths_train[0][i])
            pos_chebi_right += len(X_subpaths_train[1][i])
            pos_common += len(X_ancestors_train[i])
            pos += 1
    pos_values = [pos_word_left, pos_word_right, pos_wordnet_left, pos_wordnet_right, pos_chebi_left, pos_chebi_right, pos_common]
    print("positive pairs\n: {}".format("\n".join([str(x/pos) for x in pos_values])))

    neg_values = [neg_word_left, neg_word_right, neg_wordnet_left, neg_wordnet_right, neg_chebi_left, neg_chebi_right,
                  neg_common]
    print("negitive pairs\n: {}".format("\n".join([str(x / neg) for x in neg_values])))


if __name__ == "__main__":
    main()
