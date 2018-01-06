import sys
import logging
import os
logging.basicConfig(level=10)
import collections
import numpy as np
np.random.seed(1)

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
from models import get_model, embbed_size, max_sentence_length, max_ancestors_length, n_classes,\
    words_channel, wordnet_channel, ancestors_channel

n_inputs = 0
if words_channel:
    n_inputs += 2
if wordnet_channel:
    n_inputs += 2
if ancestors_channel:
    n_inputs += 2

DATA_DIR = "data/"
n_epochs = 20
batch_size = 5
validation_split = 0.1

# https://github.com/keras-team/keras/issues/853#issuecomment-343981960

def write_plots(history):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model F1')
    plt.ylabel('F1')
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

def get_wordnet_indexes():
    embedding_indexes = {}
    with open("sst-light-0.4/DATA/WNSS_07.TAGSET", 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l.startswith("I-"):
                continue
            embedding_indexes[l.strip().split("-")[-1]] = i
    return embedding_indexes

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



class Metrics(Callback):
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


        # for i in range(len(val_targ)):
        #     true_label = np.argmax(val_targ[i])
        #     predicted = np.argmax(val_predict[i])
        #     if predicted != true_label:
        #         error_type = "wrong label"
        #         if true_label == 0:
        #             error_type = "FP"
        #         elif predicted == 0:
        #             error_type = "FN"
        #         print("{}: {}->{}; inputs: {}".format(error_type, true_label, predicted,
        #                                               str([self.validation_data[j][i] for j in range(n_inputs)])))
        #print(val_predict, val_targ)
        #print("true not false: {}".format()
        print()
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
    left_wordnet = []
    right_wordnet = []
    classes = np.empty((0,))

    for dir in dirs:
        dir_labels, dir_instances, dir_classes, dir_ancestors, dir_wordnet = get_ddi_sdp_instances(dir)
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
        left_wordnet += dir_wordnet[0]
        right_wordnet += dir_wordnet[1]
        classes = np.concatenate((classes, dir_classes), axis=0)

    return labels, (left_instances, right_instances), classes, (left_ancestors, right_ancestors), (left_wordnet, right_wordnet)





def main():

    if sys.argv[1] == "preprocessing":
        from parse_semeval8 import get_semeval8_sdp_instances
        #labels, X_train, classes = get_ddi_data(sys.argv[3:])
        train = True
        if "test" in sys.argv[3].lower():
            train= False
        # TODO: generalize text pre-processing
        if sys.argv[2] == "semeval8":
            labels, X_train, classes = get_semeval8_sdp_instances(sys.argv[4:], train=train)
            print(len(X_train))
            print(len(X_train[0]))
        elif sys.argv[2] == "ddi":
            labels, X_train, classes, X_train_ancestors, X_train_wordnet = get_ddi_data(sys.argv[4:])
            print(len(X_train))
        np.save(sys.argv[3] + "_labels.npy", labels)
        np.save(sys.argv[3] + "_x_words.npy", X_train)
        np.save(sys.argv[3] + "_x_ancestors.npy", X_train_ancestors)
        np.save(sys.argv[3] + "_x_wordnet.npy", X_train_wordnet)
        np.save(sys.argv[3] + "_y.npy", classes)

    elif sys.argv[1] == "train":
        if os.path.isfile("model.json"):
            os.remove("model.json")
        if os.path.isfile("model.h5"):
            os.remove("model.h5")

        Y_train = np.load(sys.argv[2] + "_y.npy")
        Y_train = to_categorical(Y_train, num_classes=n_classes)

        # print(emb_index)
        inputs = {}
        if words_channel:
            emb_index, emb_matrix = get_glove_vectors()
            X_words_train = np.load(sys.argv[2] + "_x_words.npy")
            X_words_left = preprocess_sequences(X_words_train[0], emb_index)
            X_words_right = preprocess_sequences(X_words_train[1], emb_index)
            # skip root word
            # X_words_train = np.concatenate((X_words_left, X_words_right[..., 1:]), 1)
            inputs["left_words"] = X_words_left
            inputs["right_words"] = X_words_right

        else:
            emb_matrix = None

        if wordnet_channel:
            wn_index = get_wordnet_indexes()
            X_wordnet_train = np.load(sys.argv[2] + "_x_wordnet.npy")
            X_wn_left = preprocess_sequences(X_wordnet_train[0], wn_index)
            X_wn_right = preprocess_sequences(X_wordnet_train[1], wn_index)
            inputs["left_wordnet"] = X_wn_left
            inputs["right_wordnet"] = X_wn_right

        if ancestors_channel:
            is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
            X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
            X_ids_left = preprocess_ids(X_ancestors_train[0], id_to_index)
            X_ids_right = preprocess_ids(X_ancestors_train[1], id_to_index)

            #X_ancestors_train = np.concatenate((X_ids_left, X_ids_right[..., 1:]), 1)
            inputs["left_ancestors"] = X_ids_left
            inputs["right_ancestors"] = X_ids_right
        else:
            id_to_index = None

        model = get_model(emb_matrix, id_to_index)

        #model = get_words_model(emb_matrix)
        #model = get_xu_model(emb_matrix)


        #print(inputs)
        if len(sys.argv) > 3:

            Y_test = np.load(sys.argv[3] + "_y.npy")
            Y_test = to_categorical(Y_test, num_classes=n_classes)
            val_inputs = {}

            if words_channel:
                X_words_test = np.load(sys.argv[3] + "_x_words.npy")
                X_words_test_left = preprocess_sequences(X_words_test[0], emb_index)
                X_words_test_right = preprocess_sequences(X_words_test[1], emb_index)
                val_inputs["left_words"] = X_words_test_left
                val_inputs["right_words"] = X_words_test_right

            if wordnet_channel:
                X_wordnet_test = np.load(sys.argv[3] + "_x_wordnet.npy")
                X_wn_test_left = preprocess_sequences(X_wordnet_test[0], wn_index)
                X_wn_test_right = preprocess_sequences(X_wordnet_test[1], wn_index)
                val_inputs["left_wordnet"] = X_wn_test_left
                val_inputs["right_wordnet"] = X_wn_test_right

            #X_ancestors_test = np.load(sys.argv[3] + "_x_ancestors.npy")

            val_outputs = {"output": Y_test}

            test_labels = np.load(sys.argv[3] + "_labels.npy")

            #model.fit(X_words_train, Y_train, validation_data=(X_test, Y_test), epochs=n_epochs,
            #          batch_size=batch_size, callbacks=[metrics], verbose=2)
            history = model.fit(inputs,
                                {"output": Y_train}, validation_data=(val_inputs, val_outputs), epochs=n_epochs,
                                batch_size=batch_size, verbose=2, callbacks=[metrics])
            write_plots(history)

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
        wn_index = get_wordnet_indexes()
        X_words_test = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_test = np.load(sys.argv[2] + "_x_ancestors.npy")
        X_wn_test = np.load(sys.argv[2] + "_x_wordnet.npy")
        #X_words_test_left = [one_hot(" ".join(d), vocab_size) for d in X_words_test[0]]
        #X_words_test_right = [one_hot(" ".join(d), vocab_size) for d in X_words_test[1]]
        #X_words_test = [pad_sequences(X_words_test_left, maxlen=max_sentence_length, padding='post'),
        #                pad_sequences(X_words_test_right, maxlen=max_sentence_length, padding='post')]

        X_words_test_left = preprocess_sequences(X_words_test[0], emb_index)
        X_words_test_right = preprocess_sequences(X_words_test[1], emb_index)
        # X_words_test = [X_words_test_left, X_words_test_right]
        # X_words_test = np.concatenate((X_words_test_left, X_words_test_right[..., 1:]), 1)

        X_wordnet_test_left = preprocess_sequences(X_wn_test[0], wn_index)
        X_wordnet_test_right = preprocess_sequences(X_wn_test[1], wn_index)

        #X_ids_left = preprocess_ids(X_ancestors_test[0], id_to_index)
        #X_ids_right = preprocess_ids(X_ancestors_test[1], id_to_index)

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
        scores = loaded_model.predict({
                                       "left_words": X_words_test_left, "right_words": X_words_test_right,
                                       #"left_ancestors": X_ids_left, "right_ancestors": X_ids_right
                                       #"left_wordnet": X_wordnet_test_left, "right_wordnet": X_wordnet_test_right
                                       })
        with open("{}_results.txt".format(sys.argv[2].split("/")[-1]), 'w') as f:
            for i, pair in enumerate(test_labels):
                f.write(" ".join((pair[0], pair[1], str(np.argmax(scores[i])))) + "\n")

    elif sys.argv[1] == "showdata":
        limit = int(sys.argv[3])
        X_words_train = np.load(sys.argv[2] + "_x_words.npy")
        X_ancestors_train = np.load(sys.argv[2] + "_x_ancestors.npy")
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
        print(len(X_ancestors_train))
        print(len(X_ancestors_train[0]))
        print(X_ancestors_train[0][:limit])
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
