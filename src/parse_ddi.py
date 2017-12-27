import xml.etree.ElementTree as ET
import sys
import os
import logging

from parse_text import process_sentence
from chebi_path import get_lowest_common_ascestor_path, load_chebi, get_all_shortest_paths_to_root, map_to_chebi



NORELATION = 0
MECHANISM = 1
EFFECT = 2
ADVICE = 3
INTTYPE = 4

pairtypes = (MECHANISM, EFFECT, ADVICE, INTTYPE)
label_to_pairtype = {"effect": EFFECT, "mechanism": MECHANISM, "advise": ADVICE, "int": INTTYPE}


def get_ancestors(sentence_labels, sentence_entities, name_to_id, synonym_to_id, id_to_name):
    """
    obtain the path to lowest common ancestor of each entity of each pair
    :param sentence_labels: list of (e1, e2)
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :return: left and right paths to LCA
    """
    right_paths = []
    left_paths = []
    for p in sentence_labels:
        paths1 = sentence_entities[p[0]][2]
        paths2 = sentence_entities[p[1]][2]
        lca, left_path, right_path = get_lowest_common_ascestor_path(paths1, paths2, id_to_name)
        left_paths.append(left_path + [lca])
        right_paths.append(right_path + [lca])
    return (left_paths, right_paths)

def get_ddi_sdp_instances(base_dir):
    """
    Parse DDI corpus, return vectors of SDP of each relation instance
    :param base_dir: directory containing semeval XML documents and annotations
    :return: instances (vectors), classes (0/1) and labels (eid1, eid2)
    """
    is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
    left_instances = []
    right_instances = []
    left_ancestors = []
    right_ancestors = []
    classes = []
    labels = []

    for f in os.listdir(base_dir):
        logging.info(f)
        #if f != "L-Glutamine_ddi.xml":
        #    continue
        tree = ET.parse(base_dir + f)
        root = tree.getroot()
        for sentence in root:
            sentence_entities = {}
            sentence_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in
                              sentence.findall('pair') if p.get("ddi") == "true"}
            if len(sentence.findall('pair')) > 0: # skip sentences without pairs
                for e in sentence.findall('entity'):
                    sep_offsets = e.get("charOffset").split(";")
                    offsets = []
                    for o in sep_offsets:
                        start, end = o.split("-")
                        offsets.append(int(start))
                        offsets.append(int(end)+1)
                    e_text = e.get("text")
                    chebi_name = map_to_chebi(e_text, name_to_id, synonym_to_id)
                    #chebi_name = ""
                    e_path = get_all_shortest_paths_to_root(chebi_name, is_a_graph, name_to_id, synonym_to_id, id_to_name)
                    #e_path = []
                    sentence_entities[e.get("id")] = (offsets, e_text, e_path)


                # sentence_pairs: {(e1id, e2id): pairtype_label}

                sentence_labels, sentence_instances, sentence_classes = process_sentence(sentence.get("text"),
                                                                                      sentence_entities,
                                                                                      sentence_pairs)


                sentence_ancestors = get_ancestors(sentence_labels, sentence_entities,
                                                   name_to_id, synonym_to_id, id_to_name)

                labels += sentence_labels
                left_instances += sentence_instances[0]
                right_instances += sentence_instances[1]
                classes += sentence_classes
                left_ancestors += sentence_ancestors[0]
                right_ancestors += sentence_ancestors[1]
    return labels, (left_instances, right_instances), classes, (left_ancestors, right_ancestors)



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
