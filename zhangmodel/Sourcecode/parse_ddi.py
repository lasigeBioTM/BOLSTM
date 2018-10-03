import xml.etree.ElementTree as ET
import sys
import os
import logging
import pickle
import atexit

from subprocess import PIPE, Popen
from parse_text import process_sentence_spacy,\
                    parse_sentence_spacy, run_sst
from chebi_path import get_lowest_common_ascestor_path, load_chebi, get_all_shortest_paths_to_root, \
                        get_path_to_root, map_to_chebi, get_common_ancestors



#NORELATION = 0
#MECHANISM = 1
#EFFECT = 2
#ADVICE = 3
#INTTYPE = 4
NORELATION = 4
MECHANISM = 0
EFFECT = 1
ADVICE = 2
INTTYPE = 3

pairtypes = (MECHANISM, EFFECT, ADVICE, INTTYPE)
label_to_pairtype = {"effect": EFFECT, "mechanism": MECHANISM, "advise": ADVICE, "int": INTTYPE}
pairtype_tolabel = {v: k for k, v in label_to_pairtype.items()}
pairtype_tolabel[NORELATION] = "norelation"

global onto_index

def add_to_onto_index(orthid):
    if orthid not in onto_index:
        onto_index.append(orthid)
    return onto_index.index(orthid)

if os.path.isfile("../data/onto_indexes.pkl"):
    logging.info("loading onto...")
    onto_index = pickle.load(open("../data/onto_indexes.pkl", "rb"))
    loadedchebi = True
    logging.info("loaded onto index with %s entries", str(len(onto_index)))
else:
    onto_index = []
    loadedchebi = False
    logging.info("new onto_index dictionary")

def exit_handler():
    pass
    print('Saving indexes...!')
    pickle.dump(onto_index, open("../data/onto_indexes.pkl", "wb"))

atexit.register(exit_handler)

def get_ancestors(sentence_labels, sentence_entities, name_to_id, synonym_to_id, id_to_name):
    """
    obtain the path to lowest common ancestor of each entity of each pair and path from LCA to root
    :param sentence_labels: list of (e1, e2)
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :return: left and right paths to LCA
    """
    right_paths = []
    left_paths = []
    common_ancestors = []
    for p in sentence_labels:
        instance_ancestors = get_common_ancestors(sentence_entities[p[0]][2], sentence_entities[p[1]][2])
        left_path = get_path_to_root(sentence_entities[p[0]][2])
        right_path = get_path_to_root(sentence_entities[p[1]][2])
        # print("common ancestors:", sentence_entities[p[0]][1:], sentence_entities[p[1]][1:], instance_ancestors)
        instance_ancestors = [add_to_onto_index(i) for i in instance_ancestors if i.startswith("CHEBI")]
        left_path = [add_to_onto_index(i) for i in left_path if i.startswith("CHEBI")]
        right_path = [add_to_onto_index(i) for i in right_path if i.startswith("CHEBI")]
        common_ancestors.append(instance_ancestors)
        left_paths.append(left_path)
        right_paths.append(right_path)
    #return (left_paths, right_paths)
    return common_ancestors, (left_paths, right_paths)

def get_sentence_entities(base_dir, name_to_id, synonym_to_id, entity_id_max=10):
    entities = {} # sentence_id -> entities
    pair_entities = set() # list of entities not in interactions
    total_entities = 0
    total_pairs = 0
    for f in os.listdir(base_dir):
        logging.info("processing entities: {}".format(f))
        #if f != "L-Glutamine_ddi.xml":
        #    continue
        tree = ET.parse(base_dir + "/" + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            sentence_text = sentence.get("text")
            sentence_entities = {}
            all_pairs = sentence.findall('pair')
            total_pairs += len(all_pairs)
            pos_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in
                               all_pairs if p.get("ddi") == "true"}
            for p in pos_pairs:
                pair_entities.add(p[0])
                pair_entities.add(p[1])

            if len(all_pairs) > 0: # skip sentences without pairs

                for e in sentence.findall('entity'):
                    e_id = e.get("id")
                    if int(e_id.split("e")[-1]) > entity_id_max:
                        continue

                    sep_offsets = e.get("charOffset").split(";")
                    offsets = []
                    for o in sep_offsets:
                        start, end = o.split("-")
                        offsets.append(int(start))
                        offsets.append(int(end)+1)
                    e_text = e.get("text")
                    if offsets[0] == 0 and sentence_text[offsets[-1]] == ":":
                        logging.debug("skipped title: {} -> {}".format(e_text, sentence_text))
                    chebi_name, used_syn = map_to_chebi(e_text, name_to_id, synonym_to_id)
                    if chebi_name in name_to_id:
                        chebi_id = name_to_id[chebi_name]
                    else:
                        chebi_id = synonym_to_id[chebi_name][0]
                    #chebi_name = ""
                    #e_path = get_all_shortest_paths_to_root(chebi_name, is_a_graph, name_to_id, synonym_to_id, id_to_name)
                    #e_path = []
                    #sentence_entities[e.get("id")] = (offsets, e_text, e_path)
                    sentence_entities[e_id] = (offsets, e_text, chebi_id)
                entities[sentence_id] = sentence_entities
                total_entities += len(sentence_entities)
    print("total entities", total_entities)
    print("total pairs", total_pairs)
    return entities, pair_entities



def parse_ddi_sentences_spacy(base_dir, entities):

    parsed_sentences = {}
    # first iterate all documents, and preprocess all sentences
    token_seq = {}
    for f in os.listdir(base_dir):
        logging.info("parsing {}".format(f))
        tree = ET.parse(base_dir + "/" + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            if len(sentence.findall('pair')) > 0:  # skip sentences without pairs
                parsed_sentence = parse_sentence_spacy(sentence.get("text"), entities[sentence_id])
                parsed_sentences[sentence_id] = parsed_sentence
                tokens = []
                #for t in parsed_sentence:
                for t in parsed_sentence:
                    tokens.append(t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'))
                #sentence_file.write("{}\t{}\t.\n".format(sentence_id, "\t".join(tokens)))
                token_seq[sentence_id] = tokens
    #wordnet_tags = run_sst(token_seq)
    wordnet_tags = []
    return parsed_sentences, wordnet_tags



def get_ddi_sdp_instances(base_dir, name_to_id, synonym_to_id, id_to_name, parser="spacy"):
    """
    Parse DDI corpus, return vectors of SDP of each relation instance
    :param base_dir: directory containing semeval XML documents and annotations
    :return: labels (eid1, eid2), instances (vectors), classes (0/1), common ancestors, l/r ancestors, l/r wordnet
    """
    entities, positive_entities = get_sentence_entities(base_dir, name_to_id, synonym_to_id, entity_id_max=20)
    if parser == "spacy":
        parsed_sentences, wordnet_sentences = parse_ddi_sentences_spacy(base_dir, entities)
    #print(wordnet_sentences.keys())
    # print(sstoutput)
    dir_labels = []
    dir_entities = []
    dir_classes = []

    dir_sentence_words = []  # word indexes
    dir_sentence_pos = []
    dir_sentence_dist1 = []
    dir_sentence_dist2 = []

    dir_sdp_words = []  # word indexes
    dir_sdp_pos = []
    dir_sdp_dist1 = []
    dir_sdp_dist2 = []

    common_ancestors = []  # ontology IDs
    left_ancestors = []
    right_ancestors = []

    all_pos_gv = set()
    all_neg_gv = set()
    for f in os.listdir(base_dir):
        logging.info("generating instances: {}".format(f))
        tree = ET.parse(base_dir + "/" + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            sentence_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in
                              sentence.findall('pair') if p.get("ddi") == "true"}
            if len(sentence.findall('pair')) > 0: # skip sentences without pairs
                this_sentence_entities = entities[sentence_id]
                parsed_sentence = parsed_sentences[sentence_id]
                #wordnet_sentence = wordnet_sentences[sentence_id]
                wordnet_sentence = []
                # sentence_pairs: {(e1id, e2id): pairtype_label}
                if parser == "spacy":
                    sentence_labels, sentence_classes, sentence_entities, \
                        sentence_words, sentence_pos, \
                        sentence_dist1, sentence_dist2,\
                        sdp_words, sdp_pos, sdp_dist1, sdp_dist2, \
                        pos_gv, neg_gv = process_sentence_spacy(parsed_sentence,
                                                                               this_sentence_entities,
                                                                               sentence_pairs,
                                                                                     positive_entities,
                                                                               wordnet_sentence
                                                                                     )


                sentence_ancestors, sentence_subpaths = get_ancestors(sentence_labels, this_sentence_entities,
                                                   name_to_id, synonym_to_id, id_to_name)
                #sentence_ancestors = []
                #sentence_subpaths = [[],[]]

                dir_labels += sentence_labels
                dir_classes += sentence_classes
                dir_entities += sentence_entities

                dir_sentence_words += sentence_words
                dir_sentence_pos += sentence_pos
                dir_sentence_dist1 += sentence_dist1
                dir_sentence_dist2 += sentence_dist2

                dir_sdp_words += sdp_words
                dir_sdp_pos += sdp_pos
                dir_sdp_dist1 += sdp_dist1
                dir_sdp_dist2 += sdp_dist2
                for s in sentence_ancestors:
                    common_ancestors.append([add_to_onto_index(x) for x in s])
                for s in sentence_subpaths[0]:
                    left_ancestors.append([add_to_onto_index(x) for x in s])
                for s in sentence_subpaths[1]:
                    right_ancestors.append([add_to_onto_index(x) for x in s])

                all_pos_gv.update(pos_gv)
                all_neg_gv.update(neg_gv)

    return dir_labels, dir_classes, dir_entities, \
           dir_sentence_words, dir_sentence_pos, dir_sentence_dist1, dir_sentence_dist2,\
           dir_sdp_words, dir_sdp_pos, dir_sdp_dist1, dir_sdp_dist2, \
           common_ancestors, (left_ancestors, right_ancestors), \
           all_neg_gv, all_pos_gv

                    #

# print("Drugbank")
# labels, instances, classes = get_ddi_sdp_instances("data/ddi2013Train/DrugBank/")
# print(len(labels), len(instances), len(classes))
# # print(labels[:10], instances[:10], classes[:10])
# print("average path size", sum([len(a) for a in instances])/len(instances))
# print("# of true relations", classes.count(1))
# print("# of total relations", len(classes))
#
# print("Medline")
# labels, instances, classes = get_ddi_sdp_instances("data/ddi2013Train/MedLine/")
# print(len(labels), len(instances), len(classes))
# # print(labels[:10], instances[:10], classes[:10])
# print("average path size", sum([len(a) for a in instances])/len(instances))
# print("# of true relations", classes.count(1))
# print("# of total relations", len(classes))
