import sys
import os
import numpy as np
import logging
import pickle as pkl
import gzip
import collections

from chebi_path import load_chebi
logging.getLogger().setLevel(logging.DEBUG)

from hierarchi_Rnns3 import load_data

DATA_DIR = "../data/"
SSTLIGHT_DIR = "/sst-light-0.4/"
MODELS_DIR = "/models/"
validation_split = 0.2
PRINTERRORS = False

def get_ddi_data(dirs, name_to_id, synonym_to_id, id_to_name):
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
    entities = []

    sentence_words = [] # word indexes
    sentence_pos = []
    sentence_dist1 = []
    sentence_dist2 = []

    sdp_words = []  # word indexes
    sdp_pos = []
    sdp_dist1 = []
    sdp_dist2 = []

    common_ancestors = []  # ontology IDs
    left_ancestors = []
    right_ancestors = []

    all_pos_gv = set() # anti positive governors
    all_neg_gv = set()
    classes = np.empty((0,))

    for dir in dirs:
        print(dir)
        if not os.path.isdir(dir):
            print("{} does not exist!".format(dir))
            sys.exit()

        dir_labels, dir_classes, dir_entities, \
            dir_sentence_words, dir_sentence_pos, dir_sentence_dist1, dir_sentence_dist2,\
            dir_sdp_words, dir_sdp_pos, dir_sdp_dist1, dir_sdp_dist2,\
            dir_common, dir_ancestors, \
            neg_gv, pos_gv = get_ddi_sdp_instances(dir, name_to_id, synonym_to_id, id_to_name)
        #dir_instances = np.array(dir_instances)
        #print(dir_instances)
        #dir_instances = sequence.pad_sequences(dir_instances, maxlen=max_sentence_length)
        #dir_classes = np.array(dir_classes)

        labels += dir_labels
        entities += dir_entities
        #print(instances.shape, dir_instances.shape)
        #instances = np.concatenate((instances, dir_instances), axis=0)
        sentence_words += dir_sentence_words
        sentence_pos += dir_sentence_pos
        sentence_dist1 += dir_sentence_dist1
        sentence_dist2 += dir_sentence_dist2

        sdp_words += dir_sdp_words
        sdp_pos += dir_sdp_pos
        sdp_dist1 += dir_sdp_dist1
        sdp_dist2 += dir_sdp_dist2

        classes = np.concatenate((classes, dir_classes), axis=0)

        common_ancestors += dir_common
        left_ancestors += dir_ancestors[0]
        right_ancestors += dir_ancestors[1]

        all_pos_gv.update(pos_gv)
        all_neg_gv.update(neg_gv)

    return labels, classes, entities,\
           sentence_words, sentence_pos, sentence_dist1, sentence_dist2,\
           sdp_words, sdp_pos, sdp_dist1, sdp_dist2,\
           common_ancestors, (left_ancestors, right_ancestors)

def main():
    #inputs = load_data({'train_file': "../data/train.pkl.gz",
    #                                                   'test_file': "../data/test.pkl.gz",
    #                                                   'wordvecfile': "../data/vec.pkl.gz",})
    is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi("{}/chebi.obo".format(DATA_DIR))

    # train_labels, Y_train, train_entities,\
    # train_sentence_words, train_sentence_pos, train_sentence_dist1, train_sentence_dist2,\
    # sdp_words, sdp_pos, sdp_dist1, sdp_dist2,\
    # X_train_ancestors, X_train_subpaths = get_ddi_data(["../data/DDICorpus/Train/MedLine","../data/DDICorpus/Train/DrugBank"], name_to_id, synonym_to_id, id_to_name)

    train_labels, Y_train, train_entities, \
    train_sentence_words, train_sentence_pos, train_sentence_dist1, train_sentence_dist2, \
    sdp_words, sdp_pos, sdp_dist1, sdp_dist2, \
    X_train_ancestors, X_train_subpaths = get_ddi_data(sys.argv[2:], name_to_id, synonym_to_id, id_to_name)

    #pfile = open('train_bolstm.pkl', 'wb')
    pfile = open(sys.argv[1], 'wb')

    output = {"train_labels_vec": Y_train,
              "train_entity": train_entities,
              "train_word": train_sentence_words,
              "train_POS": train_sentence_pos,
              "train_distances": train_sentence_dist1,
              "train_distances2": train_sentence_dist2,
              "train_shortest_word": sdp_words,
              "train_shortest_pos": sdp_pos,
              "train_shortest_dis1": sdp_dist1,
              "train_shortest_dis2": sdp_dist2,
              "train_ancestors": X_train_ancestors,
              "train_subpaths": X_train_subpaths}
    print(output.keys())
    print("instances:", len(train_sentence_words))
    print(collections.Counter(Y_train))
    pkl.dump(output, pfile)

    #print(train_labels)
    #pkl.dump(X_train_ancestors, pfile)
    #pkl.dump(X_train_subpaths, pfile)
    #pkl.dump(inputs["common_ancestors"], pfile)
    #pkl.dump(inputs["concat_ancestors"], pfile)

    pfile.close()

if __name__ == "__main__":
    main()